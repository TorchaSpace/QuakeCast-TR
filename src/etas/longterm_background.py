"""Uzun dönem sismisite haritası (ETASbg için arka plan şekli) — sadece eğitim öncesi veriden (1900-2013).

Han, Mizrahi & Wiemer (2025, NHESS) ETASbg: μ(x) = ι · μ_k, μ_k uzun dönem oranlar (ortalaması 1'e normalize).
Biz ESHM20 yerine kendi kataloğumuzdan uzun dönem haritası kuruyoruz (İtalya OEF arka plan modelinde de tarihsel
yumuşatılmış sismisite en faydalı bileşen bulunmuştur):
  - olaylar: 1900-1969 M>=5.0 ve 1970-2013 M>=4.0 (birleşik katalog, deprem), 34-45K / 24-47D (kenar etkisi için geniş)
  - Gardner & Knopoff (1974) pencereleriyle artçı ayıklama
  - ağırlık: w_i = 10^{b (Mc_dönem - 4)} / dönem_süresi  (b = 1.0)  -> M>=4 eşdeğeri yıllık oran
  - uyarlamalı kuvvet yasası çekirdeği K_h(r) = h / (2π (r²+h²)^{3/2}), h_i = max(h_min, k'inci komşu uzaklığı) + tek tip taban w
  - (k, h_min, w): 2014-2021 eğitim hedeflerinin arka plan olasılığı (P_bg) ağırlıklı log-olabilirliği ile seçilir
Çıktı: data/processed/arka_plan/uzun_donem.npz  (0.05° ızgara, çalışma alanı üzerinde A·f ortalaması 1)
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_etas as E  # noqa

GRID = 0.05
BOX = (35.0, 44.0, 25.0, 46.0)


def gardner_knopoff(t, lat, lon, m):
    """Bağımsız (ana şok) olayların maskesi. t gün."""
    order = np.argsort(-m); keep = np.ones(len(m), bool)
    for i in order:
        if not keep[i]:
            continue
        L = 10 ** (0.1238 * m[i] + 0.983)
        T = 10 ** (0.032 * m[i] + 2.7389) if m[i] >= 6.5 else 10 ** (0.5409 * m[i] - 0.547)
        dt = t - t[i]
        cand = (np.abs(dt) <= T) & (m <= m[i]) & keep
        cand[i] = False
        if cand.any():
            d = np.sqrt(E.hav_sq(lat[cand], lon[cand], lat[i], lon[i]))
            idx = np.nonzero(cand)[0][d <= L]
            # öncü şoklar dahil pencere içindekiler bağımlı sayılır (simetrik pencere, basitleştirme)
            keep[idx] = False
    return keep


def load_events():
    d = pd.read_csv(ROOT / "data/processed/katalog_birlesik.txt", sep="\t", low_memory=False,
                    usecols=["time", "lat", "lon", "Mw", "olay_turu"])
    d = d[(d.olay_turu == "deprem") & d.Mw.notna()]
    d["time"] = pd.to_datetime(d.time, format="ISO8601")
    d = d[d.lat.between(34, 45) & d.lon.between(24, 47) & (d.time < "2014-01-01")]
    p1 = (d.time < "1970-01-01") & (d.Mw >= 4.95); p2 = (d.time >= "1970-01-01") & (d.Mw >= 3.95)
    d = d[p1 | p2].copy()
    d["mc_donem"] = np.where(d.time < "1970-01-01", 5.0, 4.0)
    d["sure_yil"] = np.where(d.time < "1970-01-01", 70.0, 44.0)
    t = (d.time - pd.Timestamp("1900-01-01")).dt.total_seconds().values / 86400
    keep = gardner_knopoff(t, d.lat.values, d.lon.values, d.Mw.values)
    d = d[keep].reset_index(drop=True)
    d["w"] = 10 ** (1.0 * (d.mc_donem - 4.0)) / d.sure_yil
    return d


def density_fn(ev, k, hmin, w_unif, area):
    lat, lon, wt = ev.lat.values, ev.lon.values, ev.w.values
    r2 = E.hav_sq(lat[:, None], lon[:, None], lat[None], lon[None]); np.fill_diagonal(r2, np.inf)
    h = np.maximum(np.sqrt(np.partition(r2, k - 1, axis=1)[:, k - 1]), hmin)

    def f(qlat, qlon):
        out = np.empty(len(qlat))
        for s in range(0, len(qlat), 500):
            rr = E.hav_sq(np.asarray(qlat[s:s + 500])[:, None], np.asarray(qlon[s:s + 500])[:, None], lat[None], lon[None])
            K = h[None] / (2 * np.pi * (rr + h[None] ** 2) ** 1.5)
            out[s:s + 500] = (1 - w_unif) * (K * wt[None]).sum(1) / wt.sum() + w_unif / area
        return out
    return f


def main(model_dir="data/processed/etas/mcvar_sonlu_tau1y"):
    import select_background as SB
    ev = load_events()
    cfg, st, md = E.load_model(model_dir)
    tlat, tlon, P, T = SB.training_targets(cfg, st, md)
    area = E.region_area([[BOX[0], BOX[2]], [BOX[1], BOX[2]], [BOX[1], BOX[3]], [BOX[0], BOX[3]]])
    best = None; rows = []
    for k in [1, 2, 3, 5, 8]:
        for hmin in [2.0, 5.0, 10.0, 20.0]:
            f0 = density_fn(ev, k, hmin, 0.0, area)
            base = f0(tlat, tlon)
            for w in [0.01, 0.03, 0.1, 0.2, 0.3]:
                dens = (1 - w) * base + w / area
                ll = (P * np.log(dens)).sum() / P.sum()
                rows.append(dict(k=k, h_min=hmin, w=w, LL=ll))
                if best is None or ll > best["LL"]:
                    best = dict(k=k, h_min=hmin, w=w, LL=ll)
    # referans: tek tip yoğunluk
    best["LL_tek_tip"] = float(np.log(1 / area))
    f = density_fn(ev, best["k"], best["h_min"], best["w"], area)
    glat = np.arange(BOX[0] + GRID / 2, BOX[1], GRID); glon = np.arange(BOX[2] + GRID / 2, BOX[3], GRID)
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    dens = f(LA.ravel(), LO.ravel()).reshape(LA.shape)
    cell_area = (111.2 * GRID) * (111.2 * GRID * np.cos(np.radians(LA)))
    dens = dens / (dens * cell_area).sum()  # bölge integrali 1
    out = ROOT / "data/processed/arka_plan"
    np.savez(out / "uzun_donem.npz", lat=glat, lon=glon, dens=dens, area=area)
    pd.DataFrame(rows).sort_values("LL", ascending=False).to_csv(out / "uzun_donem_secim.txt", sep="\t", index=False)
    best.update(n_olay=int(len(ev)), n_1900_1969=int((ev.mc_donem == 5).sum()), n_1970_2013=int((ev.mc_donem == 4).sum()))
    (out / "uzun_donem.json").write_text(json.dumps(best, indent=1))
    print(json.dumps(best, indent=1))


def load_shape():
    z = np.load(ROOT / "data/processed/arka_plan/uzun_donem.npz")
    glat, glon, dens, area = z["lat"], z["lon"], z["dens"], float(z["area"])

    def f(lat, lon):
        i = np.clip(((np.asarray(lat) - BOX[0]) / GRID).astype(int), 0, len(glat) - 1)
        j = np.clip(((np.asarray(lon) - BOX[2]) / GRID).astype(int), 0, len(glon) - 1)
        return dens[i, j]
    return f, area, (glat, glon, dens)


if __name__ == "__main__":
    main(*sys.argv[1:])
