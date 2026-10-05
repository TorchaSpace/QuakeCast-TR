"""Uydurulmuş ETAS modellerinin eğitim dışı (psödo-prospektif) log-olabilirlik testi.

λ(t,x) = μ(x) + Σ_{t_i<t} k0·e^{a(m_i-mc)} · e^{-(t-t_i)/τ}/(t-t_i+c)^{1+ω} · 1/(r²+d·e^{γ(m_i-mc)})^{1+ρ}
- tek tip arka plan: μ(x) = mu (olay/gün/km²)
- serbest arka plan (paket flETAS formülü): μ(x) = Σ_j P_bg,j · N(x; x_j, bw²) / T_eğitim  (j: eğitim hedef olayları)
LL = Σ_{test olayları} log λ(t_j,x_j) − ∫∫ λ; tetiklenme integrali tüm düzlemde (paketle aynı yaklaşım),
arka plan integrali bölge içinde. Puan = olay başına LL; aynı Mc'deki modeller karşılaştırılabilir.
Kullanım: python3 src/etas/evaluate_etas.py <model_klasoru> [<model_klasoru> ...]
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
from etas.inversion import ETASParameterCalculation, expected_aftershocks  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R_EARTH = 6378.137
WINDOWS = [("2022-01-01", "2023-02-06"), ("2023-02-06", "2026-08-01"), ("2022-01-01", "2026-08-01")]


def hav_sq(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return (2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) ** 2


def in_poly(lat, lon, poly):
    from matplotlib.path import Path as MP
    return MP(np.array(poly)).contains_points(np.c_[lat, lon])


def region_area(poly):
    lats = [p[0] for p in poly]; lons = [p[1] for p in poly]
    la0, la1, lo0, lo1 = min(lats), max(lats), min(lons), max(lons)
    return (R_EARTH ** 2) * (np.radians(lo1 - lo0)) * (np.sin(np.radians(la1)) - np.sin(np.radians(la0)))


def load_model(mdir):
    if str(mdir).endswith(".json"):  # doğrudan yapılandırma dosyası
        cfg = json.loads((ROOT / mdir).read_text())
        md = ROOT / cfg["out_dir"]
        return cfg, json.loads((md / "durum.json").read_text()), md
    mdir = ROOT / mdir
    cfg = next(json.loads(p.read_text()) for p in (ROOT / "configs").glob("etas*.json")
               if json.loads(p.read_text())["out_dir"].rstrip("/") == str(mdir.relative_to(ROOT)))
    st = json.loads((mdir / "durum.json").read_text())
    return cfg, st, mdir


def background_fn(cfg, st, mdir, cat):
    th = st["theta"]; mu = 10 ** th["log10_mu"]
    if cfg.get("bg_adaptive"):
        return adaptive_background(cfg, st, mdir)
    if cfg.get("bg_shape"):
        import longterm_background as LB
        f, area_s, _ = LB.load_shape()
        A = region_area(cfg["shape_coords"])
        return (lambda lat, lon: mu * A * f(lat, lon)), float(mu * A)
    if not cfg.get("free_background"):
        return (lambda lat, lon: np.full(len(lat), mu)), None
    if cfg.get("bg_mix"):  # serbest arka plan + uzun dönem haritası karışımı
        import longterm_background as LB
        w = json.loads((mdir / "arka_plan_karisim.json").read_text())["w"]
        cfg2 = dict(cfg); cfg2.pop("bg_mix")
        g_x, rate = background_fn(cfg2, st, mdir, cat)
        f, _, _ = LB.load_shape()
        return (lambda lat, lon: (1 - w) * g_x(lat, lon) + w * rate * f(lat, lon)), rate
    # eğitim hedef olaylarının konumları: paketin hazırlığıyla aynı filtre
    cols = ["time", "latitude", "longitude", "magnitude"] + (["mc_current"] if cfg["mc"] == "var" else [])
    meta = dict(catalog=cat[cols].copy(), m_ref=cfg.get("m_ref"), auxiliary_start=cfg["auxiliary_start"],
                timewindow_start=cfg["timewindow_start"], timewindow_end=cfg["timewindow_end"], mc=cfg["mc"],
                delta_m=cfg["delta_m"], coppersmith_multiplier=cfg["coppersmith_multiplier"], shape_coords=cfg["shape_coords"],
                theta_0=th, free_background=True, bw_sq=cfg.get("bw_sq", 2))
    import logging; logging.basicConfig(level=logging.ERROR)
    c = ETASParameterCalculation(meta); c.prepare()
    te = c.target_events
    pb = pd.read_csv(mdir / "P_background.csv", index_col=0).iloc[:, 0].reindex(te.index).fillna(0).values
    lat_j, lon_j = te["latitude"].values, te["longitude"].values
    bw2 = cfg.get("bw_sq", 2); T = c.timewindow_length

    def mu_x(lat, lon):
        out = np.empty(len(lat))
        for s in range(0, len(lat), 200):
            r2 = hav_sq(lat[s:s + 200, None], lon[s:s + 200, None], lat_j[None], lon_j[None])
            out[s:s + 200] = (np.exp(-0.5 * r2 / bw2) / (2 * np.pi * bw2) * pb[None]).sum(1) / T
        return out
    return mu_x, float(pb.sum() / T)


def adaptive_background(cfg, st, mdir):
    """select_background.py ile seçilmiş uyarlamalı çekirdek + tek tip taban."""
    import select_background as SB
    par = json.loads((mdir / "arka_plan_uyarlamali.json").read_text())
    lat_j, lon_j, P, T = SB.training_targets(cfg, st, mdir)
    h = SB.bandwidths(lat_j, lon_j, int(par["k"]), par["h_min"])
    A = region_area(cfg["shape_coords"]); w = par["w"]; rate = P.sum() / T

    def mu_x(lat, lon):
        out = np.empty(len(lat))
        for s in range(0, len(lat), 200):
            K = SB.kernel_matrix(np.asarray(lat[s:s + 200]), np.asarray(lon[s:s + 200]), lat_j, lon_j, h)
            out[s:s + 200] = rate * ((1 - w) * (K * P[None]).sum(1) / P.sum() + w / A)
        return out
    return mu_x, float(rate)


def evaluate(mdir):
    cfg, st, mdir = load_model(mdir)
    th = st["theta"]; mc = cfg["mc"]; dm = cfg["delta_m"]
    cat = pd.read_csv(ROOT / cfg["catalog"], parse_dates=["time"])
    cat["magnitude"] = np.floor(cat.magnitude / dm + 0.5) * dm
    poly = cfg["shape_coords"]
    ev = cat[(cat.magnitude >= mc - dm / 2) & (cat.time >= cfg["auxiliary_start"])].sort_values("time").reset_index(drop=True)
    ev = ev[in_poly(ev.latitude.values, ev.longitude.values, poly)].reset_index(drop=True)
    t = (ev.time - pd.Timestamp("2010-01-01")).dt.total_seconds().values / 86400
    m = ev.magnitude.values; lat = ev.latitude.values; lon = ev.longitude.values
    k0, a, c, om, tau, d, g, rho = (10 ** th["log10_k0"], th["a"], 10 ** th["log10_c"], th["omega"],
                                   10 ** th["log10_tau"], 10 ** th["log10_d"], th["gamma"], th["rho"])
    prod = k0 * np.exp(a * (m - mc)); zone = d * np.exp(g * (m - mc))
    mu_x, mu_tot = background_fn(cfg, st, mdir, cat)
    area = region_area(poly)
    res = []
    for w0, w1 in WINDOWS:
        T0 = (pd.Timestamp(w0) - pd.Timestamp("2010-01-01")).days; T1 = (pd.Timestamp(w1) - pd.Timestamp("2010-01-01")).days
        idx = np.nonzero((t >= T0) & (t < T1))[0]
        lam = np.empty(len(idx))
        for s in range(0, len(idx), 300):
            jj = idx[s:s + 300]
            dt = t[jj, None] - t[None, :]
            msk = dt > 0
            r2 = hav_sq(lat[jj, None], lon[jj, None], lat[None], lon[None])
            dtp = np.where(msk, dt, 1.0)
            gij = prod[None] * np.exp(-dtp / tau) / (dtp + c) ** (1 + om) / (r2 + zone[None]) ** (1 + rho)
            lam[s:s + 300] = (gij * msk).sum(1)
        bg = mu_x(lat[idx], lon[idx])
        lam_tot = lam + bg
        # kompanzatör: tetiklenme (tüm düzlem) + arka plan
        src = np.nonzero(t < T1)[0]
        to_start = np.maximum(T0 - t[src], 0.0); to_end = T1 - t[src]
        trig = expected_aftershocks([m[src], to_start, to_end], [[th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho], mc]).sum()
        bgi = (mu_tot if mu_tot is not None else 10 ** th["log10_mu"] * area) * (T1 - T0)
        LL = np.log(lam_tot).sum() - trig - bgi
        res.append(dict(model=cfg["name"], pencere=f"{w0}..{w1}", n_test=len(idx), beklenen=round(trig + bgi, 1),
                        LL=round(LL, 1), LL_olay_basi=round(LL / max(len(idx), 1), 4),
                        arka_plan_payi_test=round(float((bg / lam_tot).mean()), 3)))
    return res


if __name__ == "__main__":
    rows = []
    for mdir in sys.argv[1:]:
        rows += evaluate(mdir)
        print(pd.DataFrame(rows[-3:]).to_string(index=False), flush=True)
    out = ROOT / "data" / "processed" / "etas" / "test_sonuclari.txt"
    old = pd.read_csv(out, sep="\t") if out.exists() else pd.DataFrame()
    new = pd.concat([old, pd.DataFrame(rows)]).drop_duplicates(["model", "pencere"], keep="last")
    new.to_csv(out, sep="\t", index=False)
