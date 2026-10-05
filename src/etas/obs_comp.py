"""Gözlenebilir kompanzatör (maskeli değerlendirme için, kaynak-merkezli).

Eski yöntem (evaluate_masked, 'eski'): tam analitik integral − her ana şok diskinde kutupsal ızgarada ∫∫λ·w.
İki sorunu vardı:
  1) Örtüşen maskeler (ör. 6 Şubat 2023: M7.8, M6.6, M7.6, M6.0 ... hepsi M>=5.5 → ayrı disk) aynı (t,x)
     noktasını birden çok kez çıkarıyordu (çift sayım).
  2) Bir maskenin ana şokundan saatler sonra olan büyük kaynakların (ör. M7.6 Elbistan) çok sivri erken
     artçı yoğunluğu, eski maskenin kaba log-zaman ızgarasında örnekleniyordu (büyük hata).
Yeni yöntem:
  tetiklenmiş kısım: her kaynak j için gözlenebilir pay o_j(s) = E_x~f_j[ 1{x∈bölge} · e^{-β(max(Mc(t_j+s,x),3.5)-3.5)} ]
     (x, j'nin kendi mekânsal çekirdeğinden örneklenir — nokta ya da kırık-çizgi kaynak), zaman integrali analitik:
     E_j(t) = ∫_0^{t-t_j} o_j(s) dG_j(s),  G_j = beklenen artçı (expected_aftershocks / büyük-kaynak ω_b) · ξ_j · ölçek
     o_j log-zaman bölmelerinde sabit kabul edilir; maske etkisi olmayan kaynaklarda o_j = bölge içi pay (sabit).
  arka plan kısmı: μ(x)·w(t,x) maske disklerinde; her (t,x) yalnızca onu kapsayan EN GEÇ başlamış maskeye sayılır.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
import evaluate_etas as E  # noqa: E402
import finite_source as fs  # noqa: E402
import simulate_forecast as SF  # noqa: E402
from etas.inversion import expected_aftershocks  # noqa: E402
from mc_field import hav  # noqa: E402

M_TEST = 3.5


class ObsComp:
    def __init__(self, X, T_end, n_samp=160, nbins=60, seed=0):
        self.X = X; S = X["S"]; F = X["F"]; th = X["th"]
        self.ts = X["ts"]; self.m = S.magnitude.values; self.xi = X["xi"]; self.sc = X["scale35"]
        self.th8 = [th["log10_k0"], th["a"], th["log10_c"], th["omega"], th["log10_tau"], th["log10_d"], th["gamma"], th["rho"]]
        self.mref = X["mref"]; self.ob = X["omega_big"]; self.beta = X["beta"]; self.F = F; self.T_end = T_end
        n = len(S); self.tb = np.zeros(n, bool)
        if self.ob is not None:
            self.tb[list(X["tb_idx"])] = True
        rng = np.random.default_rng(seed)
        fsmap = {j: rr for j, rr, _ in X["fs_cols"]}
        zone = X["zone"]; rho = X["rho"]; poly = X["poly"]
        las, los = X["las"], X["los"]
        # mekânsal örnekler
        LAT = np.empty((n, n_samp)); LON = np.empty((n, n_samp))
        for j in range(n):
            if j in fsmap:
                rr = fsmap[j]; LAT[j], LON[j] = SF.sample_finite(rr, n_samp, zone[j], rho, rr["L"], rng)
            else:
                LAT[j], LON[j] = SF.sample_point_source(np.full(n_samp, las[j]), np.full(n_samp, los[j]), np.full(n_samp, zone[j]), rho, rng)
        inp = E.in_poly(LAT.ravel(), LON.ravel(), poly).reshape(n, n_samp)
        self.fin = inp.mean(1)
        # maskeden etkilenen kaynaklar
        self.aff = {}
        for j in range(n):
            if self.ts[j] >= T_end:
                continue
            ks = [k for k in range(len(F.t)) if F.t[k] + F.dur[k] > self.ts[j] and F.t[k] < T_end
                  and hav(LAT[j], LON[j], F.lat[k], F.lon[k]).min() <= F.r[k]]
            if not ks:
                continue
            span = T_end - self.ts[j]
            edges = np.r_[0.0, np.geomspace(1e-5, span, nbins)]
            mid = np.r_[edges[1] / 2, np.sqrt(edges[1:-1] * edges[2:])]
            tq = self.ts[j] + mid
            mc = np.full((len(mid), n_samp), 3.5)
            for k in ks:
                dt = tq - F.t[k]; ok = (dt > 0) & (dt < F.dur[k])
                if not ok.any():
                    continue
                d = hav(LAT[j], LON[j], F.lat[k], F.lon[k]) <= F.r[k]
                val = F.c0 + F.c1 * F.m[k] - F.B * np.log10(np.where(ok, dt, 1.0))
                mc = np.where(ok[:, None] & d[None, :], np.maximum(mc, val[:, None]), mc)
            o = (inp[j][None, :] * np.exp(-self.beta * (np.maximum(mc, M_TEST) - M_TEST))).mean(1)
            Ge = self.G(np.full(len(edges), j), edges)
            cum = np.r_[0.0, np.cumsum(o * np.diff(Ge))]
            self.aff[j] = (edges, o, cum)

    def G(self, jj, dt):
        jj = np.asarray(jj, int); dt = np.maximum(np.asarray(dt, float), 0)
        out = np.zeros(len(jj)); pos = dt > 0
        if pos.any():
            j = jj[pos]; d = dt[pos]
            g = expected_aftershocks([self.m[j], np.zeros(len(j)), d], [self.th8, self.mref])
            if self.ob is not None and self.tb[j].any():
                b = self.tb[j]
                g[b] = fs.big_expected(self.m[j][b], np.zeros(b.sum()), d[b], self.th8, self.mref, self.ob, expected_aftershocks)
            out[pos] = g * self.xi[j] * self.sc
        return out

    def E(self, jj, t):
        """Gözlenebilir kümülatif beklenen doğrudan artçı sayısı (j kaynağı, t mutlak zaman); jj ve t aynı boyda."""
        jj = np.asarray(jj, int); t = np.asarray(t, float)
        dt = t - self.ts[jj]
        out = self.fin[jj] * self.G(jj, dt)
        for q in np.nonzero([j in self.aff for j in jj])[0]:
            j = jj[q]
            if dt[q] <= 0:
                out[q] = 0.0; continue
            edges, o, cum = self.aff[j]
            b = min(np.searchsorted(edges, dt[q]) - 1, len(o) - 1)
            out[q] = cum[b] + o[b] * (self.G([j], [dt[q]])[0] - self.G([j], [edges[b]])[0])
        return out

    def E_source(self, j, t):
        """Tek kaynak j için E_j(t), t dizisi (vektörel)."""
        t = np.asarray(t, float); dt = t - self.ts[j]
        out = np.zeros(len(t)); pos = dt > 0
        if not pos.any():
            return out
        Gt = self.G(np.full(pos.sum(), j), dt[pos])
        if j not in self.aff:
            out[pos] = self.fin[j] * Gt
            return out
        edges, o, cum = self.aff[j]
        b = np.minimum(np.searchsorted(edges, dt[pos]) - 1, len(o) - 1)
        Ge = self.G(np.full(len(edges), j), edges)
        out[pos] = cum[b] + o[b] * (Gt - Ge[b])
        return out

    def triggered(self, T0, T1):
        j = np.nonzero(self.ts < T1)[0]
        return (self.E(j, np.full(len(j), T1)) - self.E(j, np.full(len(j), T0))).sum()

    def bg_masked(self, T0, T1, mu_x, nb=40):
        """∫∫ μ(x) w(t,x) maske disklerinde, en geç başlayan kapsayan maskeye atanarak (çift sayım yok)."""
        F = self.F; corr = 0.0
        poly = self.X["poly"]
        for k in range(len(F.t)):
            t_a, t_b = max(F.t[k], T0), min(F.t[k] + F.dur[k], T1)
            if t_b <= t_a or not E.in_poly(np.array([F.lat[k]]), np.array([F.lon[k]]), poly)[0]:
                continue
            ed = np.geomspace(max(t_a - F.t[k], 1e-5), t_b - F.t[k], nb + 1)
            mid = np.sqrt(ed[:-1] * ed[1:]); dtt = np.diff(ed)
            rr = np.linspace(0, F.r[k], 25)[1:] - F.r[k] / 48; ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            R_, A_ = np.meshgrid(rr, ang); dA = ((F.r[k] / 24) * R_ * (2 * np.pi / 24)).ravel()
            glat = (F.lat[k] + (R_ * np.cos(A_)) / 111.2).ravel()
            glon = (F.lon[k] + (R_ * np.sin(A_)) / (111.2 * np.cos(np.radians(F.lat[k])))).ravel()
            bg = mu_x(glat, glon) * self.sc
            later = [q for q in range(len(F.t)) if (F.t[q] > F.t[k] or (F.t[q] == F.t[k] and q > k))
                     and F.t[q] < t_b and F.t[q] + F.dur[q] > t_a and hav(F.lat[q], F.lon[q], F.lat[k], F.lon[k]) < F.r[q] + F.r[k]]
            dl = {q: hav(glat, glon, F.lat[q], F.lon[q]) <= F.r[q] for q in later}
            for i in range(nb):
                tg = F.t[k] + mid[i]
                mc_g = self.F(np.full(len(glat), tg), glat, glon)
                w = 1 - np.exp(-self.beta * (np.maximum(mc_g, M_TEST) - M_TEST))
                for q in later:
                    if F.t[q] < tg < F.t[q] + F.dur[q]:
                        w = np.where(dl[q], 0.0, w)
                corr += (bg * w * dA).sum() * dtt[i]
        return corr
