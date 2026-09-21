#!/usr/bin/env python
"""Paired-eval harness: run a repo's REAL LLaVA pipeline on TextVQA/GQA with greedy decode,
while recording the ACTUAL per-layer LLM visual-token schedule during prefill.

This does NOT modify any repo's core code. It imports the repo's modules, calls its real
load_pretrained_model / generate, and only adds a forward wrapper to count
`n_visual = seq_len - text_len` at each LLM layer's input during the prefill pass.

Usage (repo root on PYTHONPATH, repo's own python env):
  python paired_eval_run.py --method vispruner|sparsevlm \
      --repo <abs repo dir> --budget <budget-spec> \
      --question-file <official jsonl> --image-folder <dir> \
      --out <answers.jsonl> --record <per_question.jsonl> --runid <id>
"""
import os, sys, json, time, argparse, hashlib, warnings
import torch
warnings.filterwarnings("ignore")

def sha(s): return hashlib.sha256(s.encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--budget", required=True)          # e.g. vispruner:T:r | sparsevlm:RETAIN
    ap.add_argument("--question-file", required=True)
    ap.add_argument("--image-folder", required=True)
    ap.add_argument("--out", required=True)             # answers jsonl (repo eval format)
    ap.add_argument("--record", required=True)          # per-question jsonl
    ap.add_argument("--runid", required=True)
    ap.add_argument("--max-q", type=int, default=-1)    # -1 = all
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--cache-vision", action="store_true",
                    help="Cache CLIP vision features per image (identical inputs -> identical output). "
                         "Valid ONLY when every question is scored independently and image features depend only on the image. "
                         "A pure inference scheduling optimization; does not change per-question results.")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    method, budget = args.method, args.budget
    # MUST set RETAIN before importing llava: score.py reads it once at module import time.
    if method == "sparsevlm":
        os.environ["RETAIN_TOKN"] = budget

    from llava.model.builder import load_pretrained_model
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from PIL import Image

    CONTAINER = os.environ["LEGACY_LLAVA_7B"]

    # ---- load model via repo's REAL loader with repo's REAL budget args ----
    model_name = "llava-v1.5-7b"
    kw = {"torch_dtype": "fp16"}
    if method == "vispruner":
        t, r = budget.split(":")
        lt = load_pretrained_model(CONTAINER, None, model_name, visual_token_num=int(t), important_ratio=float(r), **kw)
        tokenizer, model, image_processor, ctx = lt
    elif method == "sparsevlm":
        # RETAIN_TOKN already set before llava import (score.py reads it at import)
        lt = load_pretrained_model(CONTAINER, None, model_name, **kw)
        tokenizer, model, image_processor, ctx = lt
    elif method == "noop":
        # true no-compression: repo's builder with dynamic_sparse=False loads the standard
        # LlavaLlamaForCausalLM on the same container (no pruner). Run under SparseVLMs repo.
        lt = load_pretrained_model(CONTAINER, None, model_name, dynamic_sparse=False, **kw)
        tokenizer, model, image_processor, ctx = lt
    elif method in ("random", "spatial"):
        # CONTROL pruner (independent; NOT any method's core): reuse VisPruner's working
        # flat-path machinery but replace its attention-based token selection with
        # random (seed) or spatial-uniform selection of `T` of the 576 patches.
        # budget syntax: "<T>:<seed>"
        t, seed = budget.split(":")
        t, seed = int(t), int(seed)
        import numpy as np
        rng_sel = np.random.RandomState(seed)
        lt = load_pretrained_model(CONTAINER, None, model_name, visual_token_num=t, important_ratio=1.0, **kw)
        tokenizer, model, image_processor, ctx = lt
        def control_encode_images(images):
            # images: [B,3,336,336]. Returns (feat[B,N,C], mask[B,N]) selecting T tokens.
            vis = model.get_model().get_vision_tower()(images, output_attentions=False)[0]  # [B,576,C]
            B, N, _ = vis.shape
            indices = np.arange(N)
            if method == "random":
                sel = rng_sel.choice(N, t, replace=False)
            else:  # spatial uniform
                side = 24
                per = int(round(np.sqrt(t)))
                per = max(1, per)
                rows = np.round(np.linspace(0, side - 1, per)).astype(int)
                cols = np.round(np.linspace(0, side - 1, per)).astype(int)
                sel = np.array([r * side + c for r in rows for c in cols])
                # if grid exceeds T (rounding up), trim; if short, pad with dense start
                if len(sel) > t: sel = np.sort(sel[:t])
                elif len(sel) < t:
                    pad = [i for i in indices if i not in sel][:t - len(sel)]
                    sel = np.sort(np.concatenate([sel, np.array(pad)]))
            mask = torch.zeros((B, N), dtype=torch.bool, device=vis.device)
            mask[:, sel] = True
            feat = model.get_model().mm_projector(vis.to(device=vis.device))
            return feat, mask
        model.get_model().encode_images = control_encode_images
    else:
        raise SystemExit(f"unknown method {method}")
    model.eval()

    # ---- optional identical-result vision caching (GQA reuses 398 images across 12578 qs) ----
    if args.cache_vision:
        import functools, hashlib as _hl
        vt = model.get_model().get_vision_tower()
        _orig_fwd = vt.forward
        _vcache = {}
        def _img_key(t):
            return t.detach().float().cpu().contiguous().numpy().tobytes()
        @functools.wraps(_orig_fwd)
        def _cached_forward(*a, **k):
            img = a[0] if a else k.get("images")
            oa = k.get("output_attentions", False)
            fb = _img_key(img)
            ck = (_hl.sha256(fb).hexdigest(), oa)
            if ck not in _vcache:
                _vcache[ck] = _orig_fwd(*a, **k)
            return _vcache[ck]
        vt.forward = _cached_forward
        print(f"[cache-vision] enabled; {len(_vcache)} unique visions cached", flush=True)

    # ---- layer forward wrapper: record n_visual per layer during prefill ----
    layers = model.model.layers
    # Prefill runs layers 0..31 once; we record `n_visual = seq_len - text_len` per layer
    # and stop (done=True) after 32 appends, so decode steps are ignored.
    recorder = {"on": False, "next": 0, "text_len": 0, "sched": [], "done": False}
    for idx, layer in enumerate(layers):
        orig = layer.forward
        def make(cur, o):
            def patch(*a, **k):
                if recorder["on"] and not recorder["done"] and cur == recorder["next"]:
                    h = k.get("hidden_states")
                    if h is None and a: h = a[0]
                    if h is not None and h.dim() == 3:
                        for hh in h:  # batch=1
                            recorder["sched"].append(int(hh.shape[0]) - recorder["text_len"])
                        recorder["next"] += 1
                    if recorder["next"] >= len(layers):
                        recorder["done"] = True
                return o(*a, **k)
            return patch
        layer.forward = make(idx, orig)

    conv_mode = "vicuna_v1"
    conv_base = conv_templates[conv_mode]

    lines = [json.loads(l) for l in open(args.question_file)]
    if args.max_q > 0: lines = lines[:args.max_q]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.record), exist_ok=True)
    ans_f = open(args.out, "w")
    rec_f = open(args.record, "w")

    t00 = time.time()
    for n, line in enumerate(lines):
        img_file = os.path.join(args.image_folder, line["image"])
        qs = line["text"]
        conv = conv_base.copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt_hash = sha(prompt)
        image = Image.open(img_file).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)

        text_len = int((input_ids != IMAGE_TOKEN_INDEX).sum())
        recorder["on"] = True; recorder["next"] = 0; recorder["sched"] = []; recorder["done"] = False; recorder["text_len"] = text_len

        ids = input_ids.cuda()
        img_t = image_tensor.unsqueeze(0).half().cuda()
        t0 = time.time()
        with torch.inference_mode():
            if method in ("vispruner", "random", "spatial"):
                out, vtn = model.generate(ids, images=img_t, image_sizes=[image.size],
                                          do_sample=False, temperature=0.0, top_p=None,
                                          num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True)
            else:
                out = model.generate(ids, images=img_t, image_sizes=[image.size],
                                     do_sample=False, temperature=0.0, top_p=None,
                                     num_beams=1, max_new_tokens=args.max_new_tokens, use_cache=True)
        elapsed = time.time() - t0
        recorder["on"] = False
        sched = list(recorder["sched"])
        tlb = (sum(sched) / (len(sched) * 576)) if len(sched) == 32 else None
        if method == "noop":
            seq = out["sequences"] if hasattr(out, "sequences") else out
            answer = tokenizer.batch_decode(seq[:, ids.shape[1]:], skip_special_tokens=True)[0].strip()
        else:
            answer = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

        ans_f.write(json.dumps({"question_id": line["question_id"], "prompt": qs,
                                "text": answer, "answer_id": "paireval", "model_id": model_name,
                                "metadata": {}}) + "\n"); ans_f.flush()
        rec_f.write(json.dumps({
            "run_id": args.runid, "method": method, "budget": budget,
            "question_id": line["question_id"], "image": line["image"],
            "prompt_hash": prompt_hash, "prompt": prompt,
            "gold": None, "answer": answer,
            "token_schedule": sched, "tlb": tlb,
            "prefill_n_image_at_l0": sched[0] if sched else None,
            "total_gen_time_s": round(elapsed, 4),
            "actual_visual_tokens_seen": sched,
        }) + "\n"); rec_f.flush()
        if (n + 1) % 50 == 0:
            print(f"[{args.runid}] {n+1}/{len(lines)}  last_sched={sched[:4]}... tlb={tlb if tlb is None else round(tlb,4)}  wall={(time.time()-t00)/60:.1f}min", flush=True)

    ans_f.close(); rec_f.close()
    print(f"[{args.runid}] DONE {len(lines)} questions in {(time.time()-t00)/60:.1f}min -> {args.out}")
    print(f"  last question sched (len {len(sched)}): {sched}")

if __name__ == "__main__":
    main()