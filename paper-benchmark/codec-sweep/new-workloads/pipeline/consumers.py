"""Consumer analyses for the producer-consumer pipeline. Each takes one frame's
producer arrays (CUDA tensors, by name) and returns the arrays the analysis
writes (CUDA tensors, by name). They run on the GPU with torch, except the
connected-component labelling, which uses scipy on the host (there is no
torch equivalent); the pipeline times the whole call either way.

  vpic     k-means (k = 8) on (log |rho_f|, |B|, |E|) -> labels; connected
           structures above the 90th |rho_f| percentile -> labels; 2-D PDF of
           (log |rho_f|, log |E|), 1024 x 1024
  lammps   coordination number (neighbours within 1.5 sigma) and the radial
           distribution (r < 2.5 sigma) from a GPU cell list; k-means on
           (speed, coordination); atom ids sorted by cell
  nanoaod  Z -> ee selection: events with >= 2 electrons, the leading pair
           opposite in charge, pT > 25 / 15 GeV; per-event selection mask,
           the pair's invariant mass for selected events, 1-D PDFs of the
           mass and leading pT and a 2-D PDF of (mass, pT)
  fem      conjugate gradient on A x = A 1 (exact solution: all ones),
           checkpointing x and the residual every 10 of 100 iterations, plus
           the residual-norm history
"""
import numpy as np
import torch
from scipy import ndimage

G = torch.Generator(device="cuda")


def _kmeans(x, k=8, iters=10, sample=200_000, seed=0):
    """Lloyd's k-means on the rows of x (standardised), fit on a sample."""
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    G.manual_seed(seed)
    idx = torch.randint(0, x.shape[0], (min(sample, x.shape[0]),), device=x.device, generator=G)
    s = x[idx]
    c = s[torch.randperm(s.shape[0], device=x.device, generator=G)[:k]].clone()
    for _ in range(iters):
        lab = torch.cdist(s, c).argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = s[m].mean(0)
    out = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)
    for a in range(0, x.shape[0], 1 << 22):
        out[a:a + (1 << 22)] = torch.cdist(x[a:a + (1 << 22)], c).argmin(1).to(torch.int32)
    return out


def _pdf2d(a, b, bins=1024):
    """Normalised 2-D histogram of two 1-D CUDA tensors, float64."""
    def edges(v):
        lo, hi = v.min(), v.max()
        return lo, (hi - lo).clamp_min(1e-30)
    la, sa = edges(a)
    lb, sb = edges(b)
    ia = ((a - la) / sa * (bins - 1)).long().clamp(0, bins - 1)
    ib = ((b - lb) / sb * (bins - 1)).long().clamp(0, bins - 1)
    h = torch.bincount(ia * bins + ib, minlength=bins * bins).double()
    return (h / h.sum()).reshape(bins, bins)


def vpic(f):
    """@param f {name: float32 CUDA tensor of 128^3 voxels}"""
    rho = f["rhof"].abs() + 1e-30
    b = torch.sqrt(f["cbx"] ** 2 + f["cby"] ** 2 + f["cbz"] ** 2)
    e = torch.sqrt(f["ex"] ** 2 + f["ey"] ** 2 + f["ez"] ** 2)
    lr = torch.log10(rho)
    km = _kmeans(torch.stack([lr, b, e], 1))
    thr = torch.quantile(rho[torch.randperm(rho.numel(), device=rho.device)[:1_000_000]], 0.9)
    side = round(rho.numel() ** (1 / 3))
    mask = (rho > thr).reshape(side, side, side).cpu().numpy()
    cc, _ = ndimage.label(mask)
    pdf = _pdf2d(lr, torch.log10(e + 1e-30))
    return {"kmeans": km, "cc_label": torch.from_numpy(cc.astype(np.int32)).cuda().reshape(-1),
            "pdf2d": pdf.reshape(-1)}


def _cell_pairs(x, L, cs, query, fn, block=1 << 21):
    """Visit every pair (query atom i, atom j) with j in i's 27 neighbouring
    cells of edge >= cs (periodic box L), calling fn(rows, r) per batch:
    rows = query positions in this batch (indices into query), r = distances
    (inf for empty slots, 0 for i == j)."""
    nc = max(3, int(L // cs))
    c = L / nc
    cell3 = torch.clamp((x / c).long(), max=nc - 1)
    key = (cell3[:, 0] * nc + cell3[:, 1]) * nc + cell3[:, 2]
    order = torch.argsort(key)
    counts = torch.bincount(key, minlength=nc ** 3)
    start = torch.cumsum(counts, 0) - counts
    xs = x[order]
    maxc = int(counts.max())
    offs = torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                        device=x.device)
    q3 = cell3[query]
    for a in range(0, query.numel(), block):
        qa = query[a:a + block]
        pa, ca = x[qa], q3[a:a + block]
        for o in offs:
            nb = torch.remainder(ca + o, nc)
            nk = (nb[:, 0] * nc + nb[:, 1]) * nc + nb[:, 2]
            st, ct = start[nk], counts[nk]
            for m in range(maxc):  # no data-dependent break: it would sync the host
                ok = m < ct
                d = xs[torch.where(ok, st + m, torch.zeros_like(st))] - pa
                d = d - L * torch.round(d / L)
                r = torch.sqrt((d * d).sum(1))
                fn(slice(a, a + qa.numel()), torch.where(ok, r, torch.full_like(r, float("inf"))))
    return order


def lammps(f, L, rc=1.5, rmax=2.5, rdf_sample=200_000, bins=4096):
    """@param f {"position", "velocity", "force": float64 CUDA tensors (N x 3)}
    @param L box edge (sigma)"""
    x = torch.remainder(f["position"].reshape(-1, 3), L)
    v = f["velocity"].reshape(-1, 3)
    n = x.shape[0]
    every = torch.arange(n, device=x.device)
    coord = torch.zeros(n, dtype=torch.int32, device=x.device)

    def count(sl, r):
        coord[sl] += ((r > 0) & (r < rc)).to(torch.int32)
    order = _cell_pairs(x, L, rc, every, count)
    G.manual_seed(1)
    samp = torch.randperm(n, device=x.device, generator=G)[:rdf_sample]
    hist = torch.zeros(bins, dtype=torch.float64, device=x.device)

    def rdf(sl, r):  # masked weights, not filtering: no host sync per step
        w = ((r > 0) & (r < rmax)).double()
        idx = (torch.nan_to_num(r, posinf=0.0) / rmax * bins).long().clamp(0, bins - 1)
        hist.add_(torch.bincount(idx, weights=w, minlength=bins))
    _cell_pairs(x, L, rmax, samp, rdf)
    speed = torch.linalg.norm(v, dim=1)
    km = _kmeans(torch.stack([speed, coord.double()], 1))
    return {"coordination": coord, "cell_sorted": order.to(torch.int32), "kmeans": km,
            "rdf": hist}


def nanoaod(cols):
    """@param cols {branch: CUDA tensor} with nElectron, Electron_pt/eta/phi/
    mass (float32) and Electron_charge (int32) for one event batch"""
    nel = cols["nElectron"].long()
    off = torch.cumsum(nel, 0) - nel
    pt, eta, phi, m = (cols[f"Electron_{k}"].double() for k in ("pt", "eta", "phi", "mass"))
    q = cols["Electron_charge"].long()
    two = nel >= 2
    i0, i1 = off, off + 1
    i0c, i1c = torch.where(two, i0, 0), torch.where(two, i1, 0)

    def p4(i):
        px = pt[i] * torch.cos(phi[i])
        py = pt[i] * torch.sin(phi[i])
        pz = pt[i] * torch.sinh(eta[i])
        e = torch.sqrt(px ** 2 + py ** 2 + pz ** 2 + m[i] ** 2)
        return e, px, py, pz
    e0, x0, y0, z0 = p4(i0c)
    e1, x1, y1, z1 = p4(i1c)
    mass = torch.sqrt(torch.clamp((e0 + e1) ** 2 - (x0 + x1) ** 2 - (y0 + y1) ** 2 - (z0 + z1) ** 2, min=0))
    sel = two & (q[i0c] * q[i1c] < 0) & (pt[i0c] > 25) & (pt[i1c] > 15)
    ms, lead = mass[sel].float(), pt[i0c][sel].float()

    def pdf1(v, lo, hi, bins=4096):
        h = torch.histc(v.double(), bins=bins, min=lo, max=hi)
        return h / h.sum().clamp_min(1)
    return {"selected": sel.to(torch.uint8), "mass": ms,
            "pdf_mass": pdf1(ms, 0, 500), "pdf_leadpt": pdf1(lead, 0, 500),
            "pdf2d_mass_pt": _pdf2d(ms.double(), lead.double()).reshape(-1)}


def fem(a, iters=100, every=10):
    """@param a {"row_ptr": int64, "col_idx": int32, "values": float64} CUDA
    tensors of one matrix. CG from x = 0 on A x = A 1."""
    rp, ci, va = a["row_ptr"], a["col_idx"].long(), a["values"]
    n = rp.numel() - 1
    A = torch.sparse_csr_tensor(rp, ci, va, size=(n, n))
    b = A @ torch.ones(n, dtype=torch.float64, device=va.device)
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rr = r @ r
    out, hist = {}, []
    for it in range(1, iters + 1):
        ap = A @ p
        alpha = rr / (p @ ap)
        x += alpha * p
        r -= alpha * ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
        hist.append(rr.sqrt())
        if it % every == 0:
            out[f"x_i{it:03d}"] = x.clone()
            out[f"r_i{it:03d}"] = r.clone()
    out["resnorm"] = torch.stack(hist)
    return out
