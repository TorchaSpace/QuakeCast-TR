"""Literatürle karşılaştırılabilir ölçüt: zamandan bağımsız Poisson referanslarına göre olay başına bilgi kazancı (IGPE).

Aynı maskeli çerçeve (gözlenebilir hedefler M >= max(3.5, Mc(t,x)), aynı gözlenebilirlik çarpanı ve kompanzatör).
Referanslar (oran R = pencere başlangıcına kadarki 2014+ gözlenebilir M>=3.5 olayların ortalama günlük sayısı):
  A  tek tip Poisson: R / alan                                     (Mizrahi vd. 2021'deki STHPP ile aynı tür)
  B  uzun dönem (1900–2013, ayıklanmış) yumuşatılmış harita × R   (ETASbg tipi; test verisi içermez)
  C  pencere başlangıcına kadarki tüm M>=3.5 olayların uyarlamalı çekirdekle yumuşatılması × R
     (ayıklanmamış; Helmstetter vd. 2007 tipi güçlü zamandan bağımsız referans; k doğrulamada seçilir)
IGPE = (LL_model − LL_ref) / N  [nat/olay]; olasılık kazancı = e^IGPE.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import evaluate_masked as EM, evaluate_etas as E, obs_comp as OC, longterm_background as LB  # noqa
from mc_field import hav  # noqa

VAL = ("2022-01-01", "2024-01-01"); TEST = ("2024-01-01", "2026-08-01")
G = 0.05; BOX = (35.0, 44.0, 25.0, 46.0)


def grid_density(lat, lon, k, h_min=2.0, floor=0.05):
    """Uyarlamalı kuvvet yasası çekirdeği (h_i = k'inci komşu uzaklığı) ızgara yoğunluğu; bölgede toplamı 1, birim km^-2."""
    glat = np.arange(BOX[0], BOX[1], G) + G / 2; glon = np.arange(BOX[2], BOX[3], G) + G / 2
    from scipy.spatial import cKDTree
    xy = np.c_[lat * 111.2, lon * 111.2 * np.cos(np.radians(lat))]
    d, _ = cKDTree(xy).query(xy, k=k + 1); h = np.maximum(d[:, -1], h_min)
    LA, LO = np.meshgrid(glat, glon, indexing="ij"); dens = np.zeros(LA.shape)
    for s in range(0, len(lat), 400):
        r2 = hav(LA[..., None], LO[..., None], lat[None, None, s:s + 400], lon[None, None, s:s + 400]) ** 2
        hh = h[None, None, s:s + 400]
        dens += (hh / (2 * np.pi * (r2 + hh ** 2) ** 1.5)).sum(-1)
    cell = (111.2 * G) * (111.2 * G * np.cos(np.radians(LA)))
    p = dens * cell; p /= p.sum()
    p = (1 - floor) * p + floor * cell / cell.sum()
    return p / cell  # km^-2


def lookup(D):
    def f(la, lo):
        i = np.clip(((np.asarray(la) - BOX[0]) / G).astype(int), 0, D.shape[0] - 1)
        j = np.clip(((np.asarray(lo) - BOX[2]) / G).astype(int), 0, D.shape[1] - 1)
        return D[i, j]
    return f


def main(ref_cfg="configs/v3_referans.json"):
    X = EM.evaluate(ref_cfg, expose=True)
    ev, obs, beta, F = X["ev"], X["observed"], X["beta"], X["F"]
    oc = OC.ObsComp(X, EM.evaluate.__globals__["pd"].Timestamp(TEST[1]).value / 8.64e13 - EM.T_ORIGIN.value / 8.64e13)
    A = E.region_area(X["poly"]); fB, _, _ = LB.load_shape()
    t = ev.t.values; d = lambda s: (pd.Timestamp(s) - EM.T_ORIGIN).days
    out = {}
    for name, (w0, w1) in [("dogrulama", VAL), ("test", TEST)]:
        T0, T1 = d(w0), d(w1)
        hist = obs & (t >= d("2014-01-01")) & (t < T0)
        R = hist.sum() / (T0 - d("2014-01-01"))
        iw = np.nonzero(obs & (t >= T0) & (t < T1))[0]
        lo_i = -beta * (np.maximum(ev.mcf.values[iw], 3.5) - 3.5)
        refs = {"A_tek_tip": lambda la, lo: np.full(len(la), R / A), "B_uzun_donem": lambda la, lo: R * fB(la, lo)}
        h2 = obs & (t < T0)
        for k in [2, 5, 10]:
            refs[f"C_yumusatilmis_k{k}"] = (lambda D: (lambda la, lo: R * lookup(D)(la, lo)))(
                grid_density(ev.latitude.values[h2], ev.longitude.values[h2], k))
        res = {}
        for rn, mu in refs.items():
            lam = mu(ev.latitude.values[iw], ev.longitude.values[iw])
            comp = R * (T1 - T0) - oc.bg_masked(T0, T1, mu)
            res[rn] = dict(LL=float(np.log(lam).sum() + lo_i.sum() - comp), n=len(iw), beklenen=round(float(comp), 1))
        out[name] = res
        print(name, {k: (round(v["LL"] / v["n"], 4), v["beklenen"]) for k, v in res.items()}, "N", len(iw), flush=True)
    json.dump(out, open(ROOT / "data/processed/etas/referans_kiyas.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
