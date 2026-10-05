"""Zamana ve mekâna bağlı tamlık büyüklüğü alanı Mc(t, x).

Büyük depremlerden hemen sonra ağ küçük artçıları kaçırır (kısa süreli artçı eksikliği, STAI).
Yaklaşım: Mc(t) = c0 + c1 * M_ana - B * log10(t - t_ana [gün])
  - Türkiye kalibrasyonu (varsayılan; data/processed/mc/mc_artci_sonrasi.txt, M>=6 ana şoklar 2010-2026,
    t<3 gün, MAXC+0.2): c0=-2.97, c1=0.83, B=0.50 (artık σ=0.39)
  - Helmstetter vd. (2006), Kaliforniya: c0=-4.5, c1=1, B=0.75 (aynı veride σ=0.42, Mc'yi ort. 0.29 düşük tahmin ediyor)
ana şokun yırtılma uzunluğunun (Wells & Coppersmith 1994, L = 10^(-2.44 + 0.59 M) km) iki katı
(en az 20 km) içinde uygulanır; Mc(t, x) = max(taban, tüm ana şoklardan gelen değerler).
"""
TURKIYE = (-2.97, 0.83, 0.50)
HELMSTETTER = (-4.5, 1.0, 0.75)
import numpy as np
import pandas as pd

R_EARTH = 6378.137


def rupture_radius(m):
    return np.maximum(20.0, 2.0 * 10 ** (-2.44 + 0.59 * np.asarray(m)))


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class McField:
    def __init__(self, mainshocks: pd.DataFrame, base=3.5, m_min_main=5.5, coef=TURKIYE):
        """mainshocks: time (gün, sayısal), latitude, longitude, magnitude sütunları"""
        ms = mainshocks[mainshocks.magnitude >= m_min_main]
        self.t = ms.time.values.astype(float); self.lat = ms.latitude.values; self.lon = ms.longitude.values
        self.m = ms.magnitude.values; self.r = rupture_radius(self.m); self.base = base
        self.c0, self.c1, self.B = coef
        # Mc'nin tabanın üstünde kaldığı süre (gün)
        self.dur = 10 ** ((self.c0 + self.c1 * self.m - base) / self.B)

    def __call__(self, t, lat, lon):
        t = np.asarray(t, float); lat = np.asarray(lat, float); lon = np.asarray(lon, float)
        mc = np.full(t.shape, self.base)
        for k in range(len(self.t)):
            dt = t - self.t[k]
            sel = (dt > 0) & (dt < self.dur[k])
            if not sel.any():
                continue
            d = hav(lat[sel], lon[sel], self.lat[k], self.lon[k])
            val = np.where(d <= self.r[k], self.c0 + self.c1 * self.m[k] - self.B * np.log10(dt[sel]), self.base)
            mc[sel] = np.maximum(mc[sel], val)
        return mc
