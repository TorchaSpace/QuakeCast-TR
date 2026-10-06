"""Fay tabanlı arka plan haritası (GEM Küresel Aktif Faylar — EMME/SHARE derlemeleri) ve 3 bileşenli karışım.

Harita: her fay ~1 km aralıkla örneklenir; nokta ağırlığı = tercih edilen net kayma hızı (mm/yıl) × sismojenik genişlik
(alt−üst sismojenik derinlik / sin(eğim)) ≈ birim uzunluk başına moment hızı. Kuvvet yasası çekirdeği
K_h(r) = h / (2π (r² + h²)^{3/2}) ile 0.05° ızgaraya yayılır; bölgede ∫ f dA = 1 (uzun_donem.npz ile aynı biçim).
Karışım: μ(x) = rate · [(1 − w_u − w_f) · G_15km(x) + w_u · f_uzun(x) + w_f · f_fay(x)], (w_u, w_f) eğitim hedeflerinde
P_bg-ağırlıklı LOO ile seçilir (select_background.mix_with_longterm'in genellemesi).
Dondurulmuş modüller değiştirilmez: değerlendirmede evaluate_etas.background_fn çalışma anında sarmalanır.
Kaynak: Styron & Pagani (2020), The GEM Global Active Faults Database, Earthquake Spectra 36(1_suppl):160–180.
"""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_etas as E  # noqa: E402
import longterm_background as LB  # noqa: E402
GEO = ROOT / "data/raw/faylar/gem_active_faults_harmonized.geojson"
BOX = (35.0, 44.0, 25.0, 46.0); G = 0.05


def first(s, default):
    m = re.findall(r"-?\d+\.?\d*", str(s or ""))
    return float(m[0]) if m else default


def fault_points(step_km=1.0):
    d = json.load(open(GEO)); pts = []
    for f in d["features"]:
        g = f["geometry"]
        if g is None:
            continue
        parts = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        p = f["properties"]
        sr = first(p.get("net_slip_rate"), np.nan)
        if not np.isfinite(sr) or sr <= 0:
            continue
        dip = first(p.get("average_dip"), 60.0); lo_d = first(p.get("lower_seis_depth"), 15.0); up_d = first(p.get("upper_seis_depth"), 0.0)
        width = max(lo_d - up_d, 5.0) / max(np.sin(np.radians(dip)), 0.3)
        for part in parts:
            a = np.array(part)[:, :2]
            if not (((a[:, 1] >= BOX[0] - 1) & (a[:, 1] <= BOX[1] + 1) & (a[:, 0] >= BOX[2] - 1) & (a[:, 0] <= BOX[3] + 1)).any()):
                continue
            for i in range(len(a) - 1):
                (x0, y0), (x1, y1) = a[i], a[i + 1]
                L = np.hypot((x1 - x0) * 111.2 * np.cos(np.radians(y0)), (y1 - y0) * 111.2)
                n = max(int(np.ceil(L / step_km)), 1); t = (np.arange(n) + 0.5) / n
                for tt in t:
                    pts.append((y0 + tt * (y1 - y0), x0 + tt * (x1 - x0), sr * width * L / n))
    return np.array(pts)


def build_map(h=10.0, power=1.0):
    from scipy.spatial import cKDTree
    P = fault_points()
    glat = np.arange(BOX[0], BOX[1], G) + G / 2; glon = np.arange(BOX[2], BOX[3], G) + G / 2
    LA, LO = np.meshgrid(glat, glon, indexing="ij"); lat0 = 39.5
    gx = np.c_[LO.ravel() * 111.2 * np.cos(np.radians(lat0)), LA.ravel() * 111.2]
    px = np.c_[P[:, 1] * 111.2 * np.cos(np.radians(lat0)), P[:, 0] * 111.2]
    w = P[:, 2] ** power
    tree = cKDTree(gx); dens = np.zeros(len(gx))
    for i in range(len(px)):
        idx = tree.query_ball_point(px[i], 8 * h)
        if idx:
            r2 = ((gx[idx] - px[i]) ** 2).sum(1)
            dens[idx] += w[i] * h / (2 * np.pi * (r2 + h * h) ** 1.5)
    dens = dens.reshape(LA.shape)
    ca = (111.2 * G) * (111.2 * G * np.cos(np.radians(glat)))[:, None] * np.ones((1, len(glon)))
    dens = dens / (dens * ca).sum()
    out = ROOT / f"data/processed/arka_plan/fay_h{h:g}_p{power:g}.npz"
    np.savez_compressed(out, lat=glat, lon=glon, dens=dens, area=float(ca.sum()))
    return out, len(P)


def load(path):
    z = np.load(path); glat, glon, dens = z["lat"], z["lon"], z["dens"]
    def f(lat, lon):
        i = np.clip(((np.asarray(lat) - BOX[0]) / G).astype(int), 0, len(glat) - 1)
        j = np.clip(((np.asarray(lon) - BOX[2]) / G).astype(int), 0, len(glon) - 1)
        return dens[i, j]
    return f, (glat, glon, dens)


def select_weights(mdir_s, fay_path):
    import select_background as SB
    cfg, st, mdir = E.load_model(mdir_s)
    lat, lon, P, T = SB.training_targets(cfg, st, mdir)
    bw2 = cfg.get("bw_sq", 2)
    r2 = E.hav_sq(lat[:, None], lon[:, None], lat[None], lon[None])
    Gm = np.exp(-0.5 * r2 / bw2) / (2 * np.pi * bw2); np.fill_diagonal(Gm, 0)
    g = (Gm * P[None]).sum(1) / (P.sum() - P)
    fu, _, _ = LB.load_shape(); ff, _ = load(fay_path)
    a, b = fu(lat, lon), ff(lat, lon)
    rows = []
    for wu in np.round(np.arange(0, 0.81, 0.1), 2):
        for wf in np.round(np.arange(0, 0.81, 0.05), 2):
            if wu + wf <= 1.0:
                rows.append(dict(w_uzun=wu, w_fay=wf, LOO_LL=float((P * np.log((1 - wu - wf) * g + wu * a + wf * b + 1e-300)).sum() / P.sum())))
    df = pd.DataFrame(rows).sort_values("LOO_LL", ascending=False)
    return df


def patch(fay_path, w_uzun, w_fay):
    """evaluate_etas.background_fn'i 'bg_mix3' yapılandırmaları için sarmalar (çalışma anında; dosya değişmez)."""
    orig = E.background_fn
    ff, _ = load(fay_path)
    def bf(cfg, st, mdir, cat):
        if not cfg.get("bg_mix3"):
            return orig(cfg, st, mdir, cat)
        cfg2 = dict(cfg); cfg2.pop("bg_mix3"); cfg2.pop("bg_mix", None)
        g_x, rate = orig(cfg2, st, mdir, cat)
        fu, _, _ = LB.load_shape()
        return (lambda lat, lon: (1 - w_uzun - w_fay) * g_x(lat, lon) + rate * (w_uzun * fu(lat, lon) + w_fay * ff(lat, lon))), rate
    E.background_fn = bf


if __name__ == "__main__":
    for h, pw in [(10.0, 1.0), (5.0, 1.0), (20.0, 1.0), (10.0, 0.5)]:
        p, n = build_map(h, pw); print("harita", p.name, "nokta", n, flush=True)
