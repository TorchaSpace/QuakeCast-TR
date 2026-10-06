"""Gözlem maskeli eğitim dışı test: sabit Mc ve değişken Mc'li modelleri aynı gözlenebilir olay kümesinde puanlar.

Gözlenebilir test olayı: M >= max(3.5, Mc(t,x))  (Mc alanı: src/catalog/mc_field.py, Türkiye kalibrasyonu)
Gözlenebilir yoğunluk: λ_obs(t,x) = λ_{>=3.5}(t,x) · exp(-β (Mc(t,x) - 3.5))
  - değişken Mc'li modelde kaynaklar: Mc'lerinin üstündeki olaylar; görünmeyen tetikleyiciler için
    paketin sorumluluk çarpanı (xi) uygulanır.
  - sabit Mc modelinde kaynaklar: uydurulduğu gibi tüm M>=3.5 olaylar.
Kompanzatör = analitik tam integral − maskelenmiş (Mc>3.5) bölgede ∫∫ λ (1 − e^{−βΔ}), ana şok çevresinde
log-zaman × polar ızgarada sayısal integral.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
from etas.inversion import expected_aftershocks, responsibility_factor, upper_gamma_ext  # noqa: E402
import evaluate_etas as E  # noqa: E402
from mc_field import McField, hav  # noqa: E402

M_TEST = 3.5
T_ORIGIN = pd.Timestamp("2010-01-01")


def evaluate(mdir, windows=None, expose=False, comp="kaynak"):
    cfg, st, mdir = E.load_model(mdir)
    th = st["theta"]; beta = st["beta"]; dm = cfg["delta_m"]
    var = cfg["mc"] == "var"; mref = cfg["m_ref"] if var else cfg["mc"]
    cat = pd.read_csv(ROOT / os.environ.get("QC_KATALOG", "data/processed/etas_girdi_2010_M25.csv"), parse_dates=["time"])
    cat["magnitude"] = np.floor(cat.magnitude / dm + 0.5) * dm
    poly = cfg["shape_coords"]
    tn = (cat.time - T_ORIGIN).dt.total_seconds().values / 86400
    F = McField(pd.DataFrame({"time": tn, "latitude": cat.latitude, "longitude": cat.longitude, "magnitude": cat.magnitude}))
    cat["mcf"] = F(tn, cat.latitude.values, cat.longitude.values); cat["t"] = tn
    ev = cat[(cat.magnitude >= M_TEST - dm / 2) & (cat.time >= cfg["auxiliary_start"])]
    ev = ev[E.in_poly(ev.latitude.values, ev.longitude.values, poly)].sort_values("t").reset_index(drop=True)
    observed = ev.magnitude.values >= np.ceil(ev.mcf.values * 10 - 1e-6) / 10 - dm / 2
    # kaynaklar
    src_ok = observed if var else np.ones(len(ev), bool)
    S = ev[src_ok].reset_index(drop=True)
    k0, a, c, om, tau, d, g, rho = (10 ** th["log10_k0"], th["a"], 10 ** th["log10_c"], th["omega"],
                                   10 ** th["log10_tau"], 10 ** th["log10_d"], th["gamma"], th["rho"])
    theta_arr = np.array([th["log10_mu"], np.nan, th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho])
    if var:
        xi = responsibility_factor(theta_arr, beta, np.maximum(np.ceil(S.mcf.values * 10 - 1e-6) / 10 - mref, 0))
    else:
        xi = np.ones(len(S))
    prod = k0 * np.exp(a * (S.magnitude.values - mref)) * xi
    zone = d * np.exp(g * (S.magnitude.values - mref))
    ts, las, los = S.t.values, S.latitude.values, S.longitude.values
    fs_cols = []; omega_big = cfg.get("omega_big")
    if cfg.get("finite_source"):
        import finite_source as fs
        R = pd.read_csv(ROOT / cfg.get("ruptures", "data/processed/etas/rupturler.csv"), parse_dates=["time"])
        k_of = fs.match_sources(S.time.values, S.magnitude.values, R)
        fs_cols = [(j, R.iloc[k], np.exp(-fs.log_Z_ratio(R.L.values[k], zone[j], rho))) for j, k in enumerate(k_of) if k >= 0]
        zm = fs.zone_mask(ts, las, los, R, cfg.get("zone_km"), cfg.get("zone_days"), T_ORIGIN)
        tb_idx = sorted(set(np.nonzero(zm)[0]) | {j for j, _, _ in fs_cols})
    scale35 = np.exp(-beta * (M_TEST - mref))  # λ_{>=3.5} = λ_{>=mref} · e^{-β(3.5-mref)}
    mu_x, mu_tot = E.background_fn(cfg, st, mdir, pd.read_csv(ROOT / cfg["catalog"], parse_dates=["time"]))

    def lam35(t, lat, lon, srcs=None, bg=None, cols=None):
        """λ_{>=3.5}(t,x). cols (S indeksleri, srcs içinde) verilirse kaynak-bazlı katkı matrisi de döner."""
        si = np.arange(len(ts)) if srcs is None else np.asarray(srcs)
        out = np.empty(len(t))
        pos = {v: q for q, v in enumerate(si)}
        if cols is not None:
            cq = np.array([pos[v] for v in cols], int); outc = np.zeros((len(t), len(cq)))
        for s in range(0, len(t), 300):
            sl = slice(s, s + 300)
            dt = t[sl, None] - ts[None, si]
            msk = dt > 0
            dtp = np.where(msk, dt, 1.0)
            r2 = E.hav_sq(lat[sl, None], lon[sl, None], las[None, si], los[None, si])
            kap = np.ones(len(si))
            if fs_cols:
                for j, rr, kp in fs_cols:
                    if j in pos:
                        r2[:, pos[j]] = fs.dist2_to_rupture(lat[sl], lon[sl], rr); kap[pos[j]] = kp
            gij = kap[None] * prod[None, si] * np.exp(-dtp / tau) / (dtp + c) ** (1 + om) / (r2 + zone[None, si]) ** (1 + rho)
            if fs_cols and omega_big is not None:
                for j in tb_idx:
                    if j in pos:
                        gij[:, pos[j]] *= fs.big_time_ratio(dtp[:, pos[j]], om, omega_big, c, tau, upper_gamma_ext)
            gm = gij * msk
            out[sl] = gm.sum(1) + (mu_x(lat[sl], lon[sl]) if bg is None else bg[sl])
            if cols is not None:
                outc[sl] = gm[:, cq]
        return out * scale35 if cols is None else (out * scale35, outc * scale35)

    if expose:
        return dict(cfg=cfg, th=th, beta=beta, ev=ev, observed=observed, S=S, xi=xi, prod=prod, zone=zone, ts=ts,
                    las=las, los=los, fs_cols=fs_cols, tb_idx=(tb_idx if fs_cols else []), omega_big=omega_big,
                    scale35=scale35, mu_x=mu_x, mu_tot=mu_tot, lam35=lam35, F=F, mref=mref, poly=poly, a=a, rho=rho)

    res = []
    wins = windows if windows is not None else E.WINDOWS[:2]
    if comp == "kaynak":
        import obs_comp as OC
        T_last = max((pd.Timestamp(w1) - T_ORIGIN).days for _, w1 in wins)
        OCobj = OC.ObsComp(dict(S=S, F=F, th=th, ts=ts, xi=xi, scale35=scale35, mref=mref, omega_big=(omega_big if fs_cols else None),
                                tb_idx=(tb_idx if fs_cols else []), beta=beta, fs_cols=fs_cols, zone=zone, rho=rho, poly=poly,
                                las=las, los=los), T_last)
    for w0, w1 in wins:
        T0 = (pd.Timestamp(w0) - T_ORIGIN).days; T1 = (pd.Timestamp(w1) - T_ORIGIN).days
        idx = np.nonzero((ev.t.values >= T0) & (ev.t.values < T1) & observed)[0]
        lt = lam35(ev.t.values[idx], ev.latitude.values[idx], ev.longitude.values[idx])
        lobs = lt * np.exp(-beta * (np.maximum(ev.mcf.values[idx], M_TEST) - M_TEST))
        # tam kompanzatör (maskesiz)
        sel = ts < T1
        trig = expected_aftershocks([S.magnitude.values[sel], np.maximum(T0 - ts[sel], 0), T1 - ts[sel]],
                                    [[th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho], mref])
        if fs_cols and omega_big is not None:
            jb = np.array([j for j in tb_idx if ts[j] < T1], int)
            if len(jb):
                th8 = [th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho]
                Gb = fs.big_expected(S.magnitude.values[jb], np.maximum(T0 - ts[jb], 0), T1 - ts[jb], th8, mref, omega_big, expected_aftershocks)
                idx_sel = np.nonzero(sel)[0]; posmap = {v: q for q, v in enumerate(idx_sel)}
                for q, j in enumerate(jb):
                    trig[posmap[j]] = Gb[q]
        trig = (trig * xi[sel]).sum() * scale35
        bgi = (mu_tot if mu_tot is not None else 10 ** th["log10_mu"] * E.region_area(poly)) * (T1 - T0) * scale35
        # maske düzeltmesi
        corr = 0.0
        for k in (range(len(F.t)) if comp == "eski" else []):
            t_a, t_b = max(F.t[k], T0), min(F.t[k] + F.dur[k], T1)
            if t_b <= t_a or not E.in_poly(np.array([F.lat[k]]), np.array([F.lon[k]]), poly)[0]:
                continue
            tt = np.geomspace(max(t_a - F.t[k], 1e-4), t_b - F.t[k], 30); te = np.r_[tt[0] / 2, tt]
            rr = np.linspace(0, F.r[k], 25)[1:] - F.r[k] / 48; ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            R, A_ = np.meshgrid(rr, ang); dA = (F.r[k] / 24) * R * (2 * np.pi / 24)
            dlat = (R * np.cos(A_)) / 111.2; dlon = (R * np.sin(A_)) / (111.2 * np.cos(np.radians(F.lat[k])))
            glat = (F.lat[k] + dlat).ravel(); glon = (F.lon[k] + dlon).ravel(); dA = dA.ravel()
            bg_g = mu_x(glat, glon)
            near = np.nonzero((ts > F.t[k] - 730) & (ts < F.t[k] + F.dur[k]) &
                              (hav(las, los, F.lat[k], F.lon[k]) < F.r[k] + 300))[0]
            for i in range(len(tt)):
                tg = F.t[k] + tt[i]; dtt = te[i + 1] - te[i]
                mc_g = F(np.full(len(glat), tg), glat, glon)
                w = 1 - np.exp(-beta * (np.maximum(mc_g, M_TEST) - M_TEST))
                if w.max() <= 0:
                    continue
                corr += (lam35(np.full(len(glat), tg), glat, glon, srcs=near, bg=bg_g) * w * dA).sum() * dtt
        if comp == "kaynak":
            trig = OCobj.triggered(T0, T1)
            corr = OCobj.bg_masked(T0, T1, mu_x)
        LL = np.log(lobs).sum() - (trig + bgi - corr)
        res.append(dict(model=cfg["name"], pencere=f"{w0}..{w1}", n_gozlenebilir=len(idx), beklenen=round(trig + bgi - corr, 1),
                        maske_duzeltmesi=round(corr, 1), LL=round(LL, 1), LL_olay_basi=round(LL / len(idx), 4)))
    if windows is not None:
        return res
    tot = dict(model=cfg["name"], pencere=f"{E.WINDOWS[2][0]}..{E.WINDOWS[2][1]}",
               n_gozlenebilir=sum(r["n_gozlenebilir"] for r in res), beklenen=round(sum(r["beklenen"] for r in res), 1),
               maske_duzeltmesi=round(sum(r["maske_duzeltmesi"] for r in res), 1), LL=round(sum(r["LL"] for r in res), 1))
    tot["LL_olay_basi"] = round(tot["LL"] / tot["n_gozlenebilir"], 4)
    return res + [tot]


if __name__ == "__main__":
    rows = []
    for m in sys.argv[1:]:
        r = evaluate(m); rows += r
        print(pd.DataFrame(r).to_string(index=False), flush=True)
    out = ROOT / "data" / "processed" / "etas" / "test_sonuclari_maskeli.txt"
    old = pd.read_csv(out, sep="\t") if out.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["model", "pencere"], keep="last").to_csv(out, sep="\t", index=False)
