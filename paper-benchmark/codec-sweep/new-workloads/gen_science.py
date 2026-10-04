#!/usr/bin/env python3
"""More data fields for the no-single-best-codec study, each written as flat
little-endian files (<array>[_tag].<dtype>) that codec_sweep / corpus_sweep
chunk directly. CPU only (CUDA_VISIBLE_DEVICES="" is fine).

    gen_science.py suitesparse --src DIR --out DIR      # *.tar.gz Matrix Market
    gen_science.py bf16ckpt    --out DIR [--model Qwen/Qwen2.5-1.5B]
    gen_science.py reluact     --out DIR [--model facebook/opt-1.3b]
                               [--seqs 13] [--seq-len 512]

suitesparse  every matrix in the full (both-triangle) CSR form a solver keeps:
             <name>_row_ptr.i64, <name>_col_idx.i32, <name>_values.f64
bf16ckpt     a natively bf16 LLM checkpoint's raw bytes, never converted,
             grouped by tensor kind across layers: embed, q, k, v, o, gate,
             up, down, norm, bias (.bf16)
reluact      post-ReLU feed-forward activations (input of every fc2) of a
             ReLU LLM over consecutive windows of a book: act_s<NN>.f16, one
             file per window, [layer][token][d_ffn]; mostly exact zeros
"""
import argparse
import glob
import os
import re
import tarfile
import urllib.request

import numpy as np


def cmd_suitesparse(a):
    import scipy.io
    for tgz in sorted(glob.glob(os.path.join(a.src, "*.tar.gz"))):
        name = os.path.basename(tgz)[:-7]
        with tarfile.open(tgz) as t:
            m = [x for x in t.getmembers() if x.name.endswith(f"{name}.mtx")][0]
            t.extract(m, a.src, filter="data")
        mtx = os.path.join(a.src, m.name)
        A = scipy.io.mmread(mtx).tocsr()  # symmetric files expand to full
        A.sum_duplicates()
        A.sort_indices()
        A.indptr.astype(np.int64).tofile(os.path.join(a.out, f"{name}_row_ptr.i64"))
        A.indices.astype(np.int32).tofile(os.path.join(a.out, f"{name}_col_idx.i32"))
        A.data.astype(np.float64).tofile(os.path.join(a.out, f"{name}_values.f64"))
        os.remove(mtx)
        print(f"{name}: {A.shape[0]} rows, {A.nnz} nonzeros", flush=True)


KINDS = [("embed", r"embed_tokens|lm_head"), ("q", r"q_proj\.weight"),
         ("k", r"k_proj\.weight"), ("v", r"v_proj\.weight"), ("o", r"o_proj\.weight"),
         ("gate", r"gate_proj"), ("up", r"up_proj"), ("down", r"down_proj"),
         ("bias", r"\.bias$"), ("norm", r"norm")]


def cmd_bf16ckpt(a):
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    path = snapshot_download(a.model, allow_patterns=["*.safetensors", "*.json"],
                             cache_dir=a.cache)
    groups = {k: [] for k, _ in KINDS}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt") as st:
            for name in st.keys():
                t = st.get_tensor(name)
                assert t.dtype == torch.bfloat16, (name, t.dtype)
                kind = next(k for k, pat in KINDS if re.search(pat, name))
                groups[kind].append(t.reshape(-1).view(torch.int16).numpy())
    for kind, parts in groups.items():
        if parts:
            arr = np.concatenate(parts)
            arr.tofile(os.path.join(a.out, f"{kind}.bf16"))
            print(f"{kind}: {arr.nbytes / 2**20:.0f} MiB", flush=True)


def cmd_reluact(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(a.threads)
    book = os.path.join(a.cache, "pg2600.txt")
    if not os.path.exists(book):
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", book)
    tok = AutoTokenizer.from_pretrained(a.model, cache_dir=a.cache)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32,
                                                 cache_dir=a.cache).eval()
    ids = tok(open(book, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    acts = []
    layers = model.model.decoder.layers
    hooks = [l.fc2.register_forward_pre_hook(lambda m, inp: acts.append(inp[0].detach()))
             for l in layers]
    for s in range(a.seqs):
        x = ids[s * a.seq_len:(s + 1) * a.seq_len].unsqueeze(0)
        acts.clear()
        with torch.no_grad():
            model(x)
        act = torch.stack([t.reshape(-1, t.shape[-1]) for t in acts]).half()  # [layer][tok][ffn]
        act.numpy().tofile(os.path.join(a.out, f"act_s{s:02d}.f16"))
        print(f"window {s}: {act.numel() * 2 / 2**20:.0f} MiB, zeros {(act == 0).float().mean():.3f}",
              flush=True)
    for h in hooks:
        h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["suitesparse", "bf16ckpt", "reluact"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=os.path.expanduser("~/np-data/new/src/suitesparse"))
    ap.add_argument("--cache", default=os.path.expanduser("~/np-data/new/src/dl"))
    ap.add_argument("--model", default="")
    ap.add_argument("--seqs", type=int, default=13)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    a.model = a.model or {"bf16ckpt": "Qwen/Qwen2.5-1.5B",
                          "reluact": "facebook/opt-1.3b"}.get(a.cmd, "")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    {"suitesparse": cmd_suitesparse, "bf16ckpt": cmd_bf16ckpt, "reluact": cmd_reluact}[a.cmd](a)


if __name__ == "__main__":
    main()
