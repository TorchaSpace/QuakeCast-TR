"""Uyarlamalı arka plan yoğunluğu: Helmstetter vd. (2007) tipi uyarlamalı çekirdek + tek tip taban.

μ(x) = (N_bg/T) · [ (1-w) Σ_j P_j K_{h_j}(x - x_j) / Σ_j P_j  +  w / A ]
K_h(r) = h / (2π (r² + h²)^{3/2})   (kuvvet yasası kuyruklu, normalize)
h_j = max(h_min, j'nin k'inci en yakın eğitim olayına uzaklığı)
P_j: EM'in eğitim hedeflerine verdiği arka plan olasılığı. (k, h_min, w) eğitim verisinde P-ağırlıklı
birini-dışarıda-bırak log-olabilirliğiyle seçilir (test verisi kullanılmaz).
Kullanım: python3 src/etas/select_background.py <model_klasoru>  -> <model_klasoru>/arka_plan_uyarlamali.json
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_etas as E  # noqa


def training_targets(cfg, st, mdir):
    from etas.inversion import ETASParameterCalculation
    import logging; logging.basicConfig(level=logging.ERROR)
    cat = pd.read_csv(ROOT / cfg["catalog"], parse_dates=["time"])
    cols = ["time", "latitude", "longitude", "magnitude"] + (["mc_current"] if cfg["mc"] == "var" else [])
    meta = dict(catalog=cat[cols].copy(), m_ref=cfg.get("m_ref"), auxiliary_start=cfg["auxiliary_start"],
                timewindow_start=cfg["timewindow_start"], timewindow_end=cfg["timewindow_end"], mc=cfg["mc"],
                delta_m=cfg["delta_m"], coppersmith_multiplier=cfg["coppersmith_multiplier"], shape_coords=cfg["shape_coords"],
                theta_0=st["theta"], free_background=True, bw_sq=cfg.get("bw_sq", 2))
    c = ETASParameterCalculation(meta); c.prepare()
    te = c.target_events
    pb = pd.read_csv(mdir / "P_background.csv", index_col=0).iloc[:, 0].reindex(te.index).fillna(0).values
    return te.latitude.values, te.longitude.values, pb, c.timewindow_length


def kernel_matrix(lat_q, lon_q, lat_j, lon_j, h):
    r2 = E.hav_sq(lat_q[:, None], lon_q[:, None], lat_j[None], lon_j[None])
    return h[None] / (2 * np.pi * (r2 + h[None] ** 2) ** 1.5)


def bandwidths(lat, lon, k, hmin):
    r2 = E.hav_sq(lat[:, None], lon[:, None], lat[None], lon[None])
    np.fill_diagonal(r2, np.inf)
    dk = np.sqrt(np.partition(r2, k - 1, axis=1)[:, k - 1])
    return np.maximum(dk, hmin)


def main(mdir_s):
    cfg, st, mdir = E.load_model(mdir_s)
    lat, lon, P, T = training_targets(cfg, st, mdir)
    A = E.region_area(cfg["shape_coords"])
    r2 = E.hav_sq(lat[:, None], lon[:, None], lat[None], lon[None])
    best = None; rows = []
    for k in [1, 2, 3, 5, 8, 12, 20]:
        rr = r2.copy(); np.fill_diagonal(rr, np.inf)
        dk = np.sqrt(np.partition(rr, k - 1, axis=1)[:, k - 1])
        for hmin in [0.5, 2, 5, 10]:
            h = np.maximum(dk, hmin)
            K = h[None] / (2 * np.pi * (r2 + h[None] ** 2) ** 1.5)
            np.fill_diagonal(K, 0)
            dens = (K * P[None]).sum(1) / (P.sum() - P)  # LOO
            for w in [0.0, 0.001, 0.01, 0.03, 0.1, 0.2]:
                ll = (P * np.log((1 - w) * dens + w / A + 1e-300)).sum() / P.sum()
                rows.append(dict(k=k, h_min=hmin, w=w, LOO_LL=ll))
                if best is None or ll > best["LOO_LL"]:
                    best = dict(k=k, h_min=hmin, w=w, LOO_LL=ll)
    # referans: paketin sabit 15 km Gauss çekirdeği ile LOO
    bw2 = cfg.get("bw_sq", 2)
    G = np.exp(-0.5 * r2 / bw2) / (2 * np.pi * bw2); np.fill_diagonal(G, 0)
    ref = (P * np.log((G * P[None]).sum(1) / (P.sum() - P) + 1e-300)).sum() / P.sum()
    best.update(referans_gauss_LOO_LL=ref, N_bg=float(P.sum()), T_gun=float(T), alan_km2=A)
    (mdir / "arka_plan_uyarlamali.json").write_text(json.dumps(best, indent=1))
    pd.DataFrame(rows).sort_values("LOO_LL", ascending=False).to_csv(mdir / "arka_plan_secim.txt", sep="\t", index=False)
    print(json.dumps(best, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
