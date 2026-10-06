"""Çok katlı doğrulamada iki üyeli topluluk (değişken-Mc varyant + sabit Mc) ağırlığının seçimi.
Kullanım: python3 src/etas/cv_topluluk.py <varyant> <m_p> <sig>   (ν = 0.3)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import logsumexp
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import seq_update as SU, seq_omori as SO, evaluate_masked as EM  # noqa
from cv_degerlendir import FOLDS  # noqa


def member(cfg, val, nxt, m_p, sig):
    SU.VAL, SU.TEST = val, nxt
    P = SU.prepare_sparse(cfg); Q = SO.prepare_omori(cfg)
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    out, pr = SO.score_omori(P, Q, 0.3, 0.0, sig, m_p, windows=(val,), return_parts=True)
    X = EM.evaluate(cfg, expose=True); ev = X["ev"]
    q = np.searchsorted(ev.t.values, P["tO"])
    lobs = -X["beta"] * (np.maximum(ev.mcf.values[q], 3.5) - 3.5)
    key = f"{val[0]}..{val[1]}"
    return P["tO"], np.log(P["lam_tot"]) + pr["dl"] + lobs, base[key]["beklenen"] + out[0]["dcomp"], base[key]["n_gozlenebilir"]


if __name__ == "__main__":
    var, m_p, sig = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    rows = []
    for f, (val, nxt) in FOLDS.items():
        c1 = f"configs/cv/{f}_{var}_mix.json"; c2 = f"configs/cv/{f}_sabit.json"
        t1, l1, C1, n = member(c1, val, nxt, m_p, sig); t2, l2, C2, _ = member(c2, val, nxt, m_p, sig)
        T0, T1 = SU._days(val[0]), SU._days(val[1])
        i1 = np.nonzero((t1 >= T0) & (t1 < T1))[0]; i2 = np.searchsorted(t2, t1[i1]); assert np.allclose(t2[i2], t1[i1])
        for w in np.round(np.arange(0, 1.01, 0.1), 1):
            ll = logsumexp(np.vstack([np.log(max(w, 1e-300)) + l1[i1], np.log(max(1 - w, 1e-300)) + l2[i2]]), axis=0).sum() - (w * C1 + (1 - w) * C2)
            rows.append(dict(kat=f, w_degisken=w, LL_olay=ll / len(i1)))
        print(f, "tamam", flush=True)
    d = pd.DataFrame(rows).pivot(index="w_degisken", columns="kat", values="LL_olay"); d["ort"] = d.mean(1)
    print(d.round(4).to_string())
    d.to_csv(ROOT / f"data/processed/etas/cv/topluluk_{var}_mp{m_p}_s{sig}.csv")
