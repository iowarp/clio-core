#!/usr/bin/env python3
"""Deep-learning workloads: the tensors a training / inference job writes.

    gen_dl.py train --out DIR [--epochs 8] [--cache DIR]
    gen_dl.py ckpt  --out DIR [--model EleutherAI/pythia-160m]
                    [--revisions step1000,step36000,step72000,step143000]
    gen_dl.py kv    --out DIR [--windows 32] [--cache DIR]

train  ResNet-18 trained on CIFAR-10 (Adam, batch 256). Every epoch writes
       weights / grad / adam_m / adam_v (f32, all parameters concatenated) and
       the post-ReLU activations of one fixed batch of 512 images (f32 and
       f16), which are about half zeros.
ckpt   a public LM's checkpoints across training (Pythia: same model, saved
       at several steps), each written as f32, f16 and bf16, plus a
       symmetric per-row int8 quantisation of the last one.
kv     GPT-2's key/value cache (f16) over 1024-token windows of a book.

Files: <out>/<tensor>_<tag>.<f32|f16|bf16|i8>, flat little-endian.
"""
import argparse
import os
import urllib.request

import numpy as np
import torch


def write(out, name, t):
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        arr, ext = t.view(torch.int16).numpy(), "bf16"
    else:
        arr = t.numpy()
        ext = {np.dtype("float32"): "f32", np.dtype("float16"): "f16",
               np.dtype("int8"): "i8"}[arr.dtype]
    arr.tofile(os.path.join(out, f"{name}.{ext}"))
    print(f"  {name}.{ext}: {arr.nbytes / 2**20:.0f} MiB", flush=True)


def flat(tensors):
    return torch.cat([t.reshape(-1).float() for t in tensors])


def cmd_train(a):
    import torchvision
    import torchvision.transforms as T
    dev = "cuda"
    tf = T.Compose([T.ToTensor(), T.Normalize((0.49, 0.48, 0.45), (0.25, 0.24, 0.26))])
    ds = torchvision.datasets.CIFAR10(a.cache, train=True, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, num_workers=8,
                                     drop_last=True)
    probe = torch.stack([ds[i][0] for i in range(512)]).to(dev)
    model = torchvision.models.resnet18(num_classes=10).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    acts = []
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
        params = list(model.parameters())
        st = [opt.state[p] for p in params]
        write(a.out, f"weights_e{ep:02d}", flat(params))
        write(a.out, f"grad_e{ep:02d}", flat(p.grad for p in params))
        write(a.out, f"adam_m_e{ep:02d}", flat(s["exp_avg"] for s in st))
        write(a.out, f"adam_v_e{ep:02d}", flat(s["exp_avg_sq"] for s in st))
        # post-ReLU activations of a fixed batch (torchvision ReLUs are reused
        # modules, so hook the functional output of every ReLU call)
        acts.clear()
        hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o.detach().reshape(-1).clone()))
                 for m in model.modules() if isinstance(m, torch.nn.ReLU)]
        model.eval()
        with torch.no_grad():
            model(probe)
        for h in hooks:
            h.remove()
        act = torch.cat(acts)
        write(a.out, f"act_e{ep:02d}", act)
        write(a.out, f"act16_e{ep:02d}", act.half())
        print(f"epoch {ep}: loss {loss.item():.3f}, act zeros {(act == 0).float().mean():.2f}",
              flush=True)


def cmd_ckpt(a):
    from transformers import AutoModelForCausalLM
    last = None
    for rev in a.revisions.split(","):
        m = AutoModelForCausalLM.from_pretrained(a.model, revision=rev, torch_dtype=torch.float32,
                                                 cache_dir=a.cache)
        w = flat(m.state_dict().values())
        tag = rev.replace("step", "s")
        write(a.out, f"ckpt_{tag}", w)
        write(a.out, f"ckpt_{tag}", w.half())
        write(a.out, f"ckpt_{tag}", w.bfloat16())
        last = m
        print(f"{a.model}@{rev}: {w.numel()} parameters", flush=True)
    q = []
    for t in last.state_dict().values():
        t = t.float()
        t2 = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
        s = t2.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127
        q.append(torch.round(t2 / s).clamp(-127, 127).to(torch.int8).reshape(-1))
    write(a.out, "ckpt_int8", torch.cat(q))


def cmd_kv(a):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = os.path.join(a.cache, "pg2600.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/2600/pg2600.txt", path)
    tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=a.cache)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 inference on the GPU; on the CPU run fp32 and store the cache as fp16
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        cache_dir=a.cache).to(dev).eval()
    ids = tok(open(path, encoding="utf-8").read(), return_tensors="pt").input_ids[0]
    for w in range(a.windows):
        x = ids[w * 1024:(w + 1) * 1024].unsqueeze(0).to(dev)
        with torch.no_grad():
            pkv = model(x, use_cache=True).past_key_values
        if hasattr(pkv, "layers"):          # transformers >= 4.5x DynamicCache
            pairs = [(l.keys, l.values) for l in pkv.layers]
        elif hasattr(pkv, "key_cache"):
            pairs = list(zip(pkv.key_cache, pkv.value_cache))
        else:                               # legacy tuple of (k, v)
            pairs = [(t[0], t[1]) for t in pkv]
        kv = torch.cat([torch.cat([k.reshape(-1), v.reshape(-1)]) for k, v in pairs]).half()
        write(a.out, f"kv_w{w:03d}", kv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "ckpt", "kv"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=os.path.expanduser("~/np-data/new/src/dl"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revisions", default="step1000,step36000,step72000,step143000")
    ap.add_argument("--windows", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    {"train": cmd_train, "ckpt": cmd_ckpt, "kv": cmd_kv}[a.cmd](a)


if __name__ == "__main__":
    main()
