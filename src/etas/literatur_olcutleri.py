"""Literatürle karşılaştırılabilir ölçütler (kendi sürümlerimize göre değil, dış referanslara göre).

1) Olay başına bilgi kazancı (IGPE, nat) — zamandan bağımsız Poisson referanslarına göre (referans_kiyas.py):
   sürekli güncelleme ve GÜNLÜK güncelleme (literatürdeki 1 günlük tahminlerle aynı tür); hedef M>=3.5 ve M>=3.95.
2) Standart ETAS (sabit Mc 3.5, tek tip arka plan) aynı çerçevede — verinin kendi kümelenme düzeyini gösterir.
3) CSEP N-testi %95 ve %99 güven düzeylerinde.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import seq_omori as SO, seq_update as SU, evaluate_masked as EM, referans_kiyas as RK, obs_comp as OC, evaluate_etas as E, longterm_background as LB  # noqa
WIN = [("dogrulama", SU.VAL), ("test", SU.TEST)]


def member(cfg, mode):
    """Üye model: O olayları için log λ (3.5 hedefi, gözlenebilirlik dahil), β, pencere kompanzatörleri."""
    P = SU.prepare_sparse(cfg); spec = json.load(open(cfg)); so = spec.get("seq_omori")
    base = {r["pencere"]: r for r in json.loads(str(P["base"]))}
    deltas = None
    if so is None:
        Q = SO.prepare_omori(cfg, deltas=np.array([0.0])) if "duy_mc35.json" in cfg else SO.prepare_omori(cfg)
    else:
        Q = SO.prepare_omori(cfg)
    if mode == "gunluk":
        D = SO.prepare_daily(cfg, deltas=Q["deltas"])
        res, dl = SO.score_daily(P, Q, D, spec.get("seq_nu", 0.3), (so or {}).get("mu", 0.0), (so or {}).get("sig", 0.2),
                                 (so or {}).get("m_p", 4.5), base_only=so is None)
    else:
        if so is None:
            res = [dict(pencere=k, dcomp=0.0, dLL=0.0) for k in base]; dl = np.zeros(len(P["tO"]))
        else:
            res, pr = SO.score_omori(P, Q, spec["seq_nu"], so["mu"], so["sig"], so["m_p"], return_parts=True); dl = pr["dl"]
    X = EM.evaluate(cfg, expose=True); ev = X["ev"]
    q = np.searchsorted(ev.t.values, P["tO"])
    mcf = ev.mcf.values[q]; mag = ev.magnitude.values[q]
    comp = {r["pencere"]: base[r["pencere"]]["beklenen"] + r["dcomp"] for r in res}
    return dict(tO=P["tO"], loglam=np.log(P["lam_tot"]) + dl, beta=X["beta"], mcf=mcf, mag=mag, comp=comp)


def ll(members, weights, tsel, key, thr):
    """Karışım log-olabilirliği; thr=3.5 ya da 3.95 hedef eşiği (3.95 için λ ve Λ e^{-β·0.45}... ölçeklenir)."""
    L = []; C = 0.0
    for M_, w in zip(members, weights):
        iw = np.searchsorted(M_["tO"], tsel); assert np.allclose(M_["tO"][iw], tsel)
        sh = -M_["beta"] * (thr - 3.45) if thr > 3.5 else 0.0
        lo = -M_["beta"] * (np.maximum(M_["mcf"][iw], thr) - thr)
        L.append(np.log(w) + M_["loglam"][iw] + sh + lo)
        C += w * M_["comp"][key] * np.exp(sh)
    from scipy.special import logsumexp
    return logsumexp(np.vstack(L), axis=0).sum() - C


def main():
    rk = json.load(open(ROOT / "data/processed/etas/referans_kiyas.json"))
    models = {"Standart ETAS (sabit Mc, tek tip)": [("configs/etas_duy_mc35.json", 1.0)],
              "QuakeCast-TR v3 (güncellemesiz)": [("configs/v3_referans.json", 1.0)],
              "QuakeCast-TR v6 topluluk": [("configs/v6_dizi_omori.json", 0.8), ("configs/v6_sabitMc_dizi_omori.json", 0.2)]}
    # referans B (uzun dönem) olay bazında: β = v3'ün β'sı
    X = EM.evaluate("configs/v3_referans.json", expose=True); ev = X["ev"]; obs = X["observed"]; betaR = X["beta"]
    fB, _, _ = LB.load_shape(); t = ev.t.values
    rows = []
    for mode in ["surekli", "gunluk"]:
        mem = {name: [(member(c, mode), w) for c, w in lst] for name, lst in models.items()}
        for wname, (w0, w1) in WIN:
            T0, T1 = SU._days(w0), SU._days(w1); key = f"{w0}..{w1}"
            R = (obs & (t >= SU._days("2014-01-01")) & (t < T0)).sum() / (T0 - SU._days("2014-01-01"))
            compB = rk[wname]["B_uzun_donem"]["beklenen"]
            for thr in [3.5, 3.95]:
                any_m = next(iter(mem.values()))[0][0]
                tO = any_m["tO"]
                iw = np.nonzero((tO >= T0) & (tO < T1) & (any_m["mag"] >= thr if thr > 3.5 else np.ones(len(tO), bool)))[0]
                ie = np.searchsorted(t, tO[iw])
                lamB = np.log(R * fB(ev.latitude.values[ie], ev.longitude.values[ie]))
                sh = -betaR * (thr - 3.45) if thr > 3.5 else 0.0
                loB = -betaR * (np.maximum(ev.mcf.values[ie], thr) - thr)
                LLB = (lamB + sh + loB).sum() - compB * np.exp(sh)
                for name, lst in mem.items():
                    LLm = ll([m for m, _ in lst], [w for _, w in lst], tO[iw], key, thr)
                    ig = (LLm - LLB) / len(iw)
                    rows.append(dict(kip=mode, pencere=wname, hedef=f"M>={thr}", N=len(iw), model=name, IGPE=round(ig, 3),
                                     olasilik_kazanci=round(float(np.exp(ig)), 1)))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(ROOT / "data/processed/etas/literatur_olcutleri.txt", sep="\t", index=False)
    # N-testi %95 / %99
    L = []
    for tag in ["v3_referans", "v6_dizi_omori", "v6_topluluk"]:
        d = pd.read_csv(ROOT / f"data/processed/etas/csep/{tag}.txt", sep="\t")
        f95 = ((d.delta1 < 0.025) | (d.delta2 < 0.025)).mean(); f99 = ((d.delta1 < 0.005) | (d.delta2 < 0.005)).mean()
        over99 = (d.delta2 < 0.005).mean()
        L.append(dict(model=tag, pencere=len(d), N_bas_95=round(100 * f95), N_bas_99=round(100 * f99), fazla_99=round(100 * over99)))
    print(pd.DataFrame(L).to_string(index=False))
    pd.DataFrame(L).to_csv(ROOT / "data/processed/etas/csep_N_95_99.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
