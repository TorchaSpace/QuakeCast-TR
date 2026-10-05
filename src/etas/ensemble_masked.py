"""Model topluluğu (doğrusal havuz) — maskeli olabilirlikle, ağırlıklar doğrulamada seçilir.
λ_E = Σ_m w_m λ'_m (λ'_m: dizi-özgü güncellenmiş yoğunluk); Λ_E = Σ_m w_m Λ'_m.
Marzocchi vd. 2012; Taroni vd. 2014 (puan tabanlı ağırlık), Herrmann & Marzocchi 2023 (OEF-İtalya).
Kullanım: python3 src/etas/ensemble_masked.py ν cfg1.json cfg2.json ...
"""
import json, sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import seq_update as SU  # noqa: E402


def parts(cfg, nu):
    P = SU.prepare_sparse(cfg)
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    if nu > 0:
        out, pr = SU.score_sparse(P, nu, 3.5, return_parts=True); dl = pr["dl"]
    else:
        out = [dict(pencere=k, dcomp=0.0, dLL=0.0) for k in base]; dl = np.zeros(len(P["tO"]))
    oc = {r["pencere"]: r for r in out}
    import evaluate_masked as EM
    X = EM.evaluate(cfg, expose=True); ev = X["ev"]
    q = np.searchsorted(ev.t.values, P["tO"]); assert np.allclose(ev.t.values[q], P["tO"])
    lobs = -X["beta"] * (np.maximum(ev.mcf.values[q], 3.5) - 3.5)  # gözlenebilirlik çarpanı (modelin β'sı ile)
    return P["tO"], np.log(P["lam_tot"]) + dl + lobs, base, oc


def main(nu, cfgs):
    D = [parts(c, nu) for c in cfgs]
    tO = D[0][0]
    for d in D[1:]:
        assert len(d[0]) == len(tO) and np.allclose(d[0], tO)
    rows = []
    wins = [SU.VAL, SU.TEST]
    # ağırlık ızgarası (simpleks, 0.1 adım)
    K = len(cfgs); grid = [w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=K) if abs(sum(w) - 1) < 1e-9]
    res = {}
    for w in grid:
        r = []
        for w0, w1 in wins:
            key = f"{w0}..{w1}"; T0, T1 = SU._days(w0), SU._days(w1)
            iw = (tO >= T0) & (tO < T1)
            ref = D[0]
            # log λ'_m - log λ'_ref
            L = np.vstack([d[1][iw] for d in D])
            lse = np.log(np.maximum((np.array(w)[:, None] * np.exp(L - L[0])).sum(0), 1e-300))
            comp = np.array([d[2][key]["beklenen"] + d[3][key]["dcomp"] for d in D])
            LLref = ref[2][key]["LL"] + ref[3][key]["dLL"]
            LL = LLref + lse.sum() - (np.dot(w, comp) - comp[0])
            r.append(LL / ref[2][key]["n_gozlenebilir"])
        res[tuple(np.round(w, 1))] = r
    df = pd.DataFrame([dict(w=k, dogrulama=v[0], test=v[1]) for k, v in res.items()]).sort_values("dogrulama", ascending=False)
    print(df.head(8).round(4).to_string(index=False))
    print("tekil modeller:"); print(df[df.w.apply(lambda t: max(t) == 1.0)].round(4).to_string(index=False))
    return df


if __name__ == "__main__":
    main(float(sys.argv[1]), sys.argv[2:])
