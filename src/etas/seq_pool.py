"""Dizi-havuzlu (hiyerarşik) verimlilik ön-bilgisi — yalnız geçmiş veriyle.

Gerekçe: büyük dizilerden sonraki fazla tahmin (CSEP N-testi), dizideki çok sayıda orta büyüklükte olayın her birine
genel verimlilik (ön-bilgi ortalaması 1) verilmesinden geliyor; gözlenen pencere içi tetikleme beklenenin %40–90'ı.
USGS dizi-özgü yaklaşımı (Page vd. 2016) ve mekânsal değişken verimlilik (Nandan vd. 2017) ile uyumlu olarak,
yeni kaynak j'nin ön-bilgi ortalaması komşu ve yakın geçmişteki kaynakların o ana kadarki toplu sonsalından alınır:
    m_j = (κ + Σ_k n_k(d_j)) / (κ + Σ_k E_k(d_j)),   k: |x_k − x_j| <= R, d_j − D <= t_k < d_j (d_j: j'nin gün başı)
ve g_j(t) = (ν + n_j(t⁻)) / (ν/m_j + E_j(t))  (seq_update.score_sparse prior_mean ile; kompanzatör kapalı biçim).
Kullanım: python3 src/etas/seq_pool.py configs/v3_referans.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
import seq_update as SU  # noqa: E402
import evaluate_masked as EM  # noqa: E402
from mc_field import hav  # noqa: E402


def daily_stats(P, oc, D):
    """Her kaynak k için gün başlarında (t_k'dan sonraki D güne kadar) n_k(d) ve E_k(d)."""
    ts = P["ts"]; nS = len(ts); tE = P["tO"][P["EVT"]]; SRC = P["SRC"]
    Pij = P["LIJ"].astype(float) / P["lam_tot"][P["EVT"]]
    d0 = np.floor(ts) + 1                      # kaynağın ertesi gün başı
    days = d0[:, None] + np.arange(D + 1)[None, :]   # nS × (D+1)
    Nk = np.zeros(days.shape); Ek = np.zeros(days.shape)
    order = np.lexsort((tE, SRC)); s_src, s_t, s_p = SRC[order], tE[order], Pij[order]
    bounds = np.r_[0, np.cumsum(np.bincount(SRC, minlength=nS))]
    for k in range(nS):
        a, b = bounds[k], bounds[k + 1]
        if b > a:
            cs = np.r_[0, np.cumsum(s_p[a:b])]
            Nk[k] = cs[np.searchsorted(s_t[a:b], days[k], side="left")]
        Ek[k] = oc.E_source(k, days[k])
    return d0, Nk, Ek


def pooled_prior(P, X, oc, R=30.0, D=90, kappa=10.0):
    ts = P["ts"]; nS = len(ts)
    las, los = X["las"], X["los"]
    d0, Nk, Ek = daily_stats(P, oc, D)
    pm = np.ones(nS); nn = np.zeros(nS, int)
    order = np.argsort(ts)
    for j in range(nS):
        dj = np.floor(ts[j])
        lo = np.searchsorted(ts, dj - D, side="left"); hi = np.searchsorted(ts, dj, side="left")
        if hi <= lo:
            continue
        k = np.arange(lo, hi)
        near = hav(las[k], los[k], las[j], los[j]) <= R
        k = k[near]
        if not len(k):
            continue
        col = (dj - d0[k]).astype(int)              # dj, k'nin ertesi gün başından kaç gün sonra
        ok = (col >= 0) & (col <= D); k, col = k[ok], col[ok]
        if not len(k):
            continue
        N = Nk[k, col].sum(); E = Ek[k, col].sum()
        pm[j] = (kappa + N) / (kappa + E); nn[j] = len(k)
    return pm, nn


if __name__ == "__main__":
    cfg = sys.argv[1]
    t0 = time.time()
    P = SU.prepare_sparse(cfg); X = EM.evaluate(cfg, expose=True)
    import obs_comp as OC
    oc = OC.ObsComp(X, SU._days(SU.T_END))
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    f = lambda res: [round((base[x["pencere"]]["LL"] + x["dLL"]) / base[x["pencere"]]["n_gozlenebilir"], 4) for x in res]
    print("havuzsuz (ν=0.3):", f(SU.score_sparse(P, 0.3, 3.5)), flush=True)
    out = {}
    for R, D in [(30.0, 90), (15.0, 60)]:
        for kappa in [5.0, 20.0, 50.0]:
            pm, nn = pooled_prior(P, X, oc, R, D, kappa)
            out[f"R{R:g}_D{D}_k{kappa:g}"] = pm
            print(f"R={R} D={D} κ={kappa}: ön-bilgi ort. medyanı {np.median(pm[nn > 0]):.2f} (komşulu kaynak %{100*(nn>0).mean():.0f})",
                  f(SU.score_sparse(P, 0.3, 3.5, prior_mean=pm)), f"({time.time()-t0:.0f} s)", flush=True)
    np.savez_compressed(ROOT / (json.load(open(cfg))["out_dir"]) / "havuz_onbilgi.npz", **out)
