"""ETAS simülasyonuyla olasılıksal deprem tahmini (QuakeCast-TR çıktı formatı).

Verilen modelden, T0 anından itibaren H gün için N_SIM sentetik gelecek katalog üretir:
  1) arka plan olayları: hız N_bg/T (eğitim), konumlar arka plan yoğunluğundan (eğitim hedefleri P_bg ağırlıklı
     + çekirdek ofseti; uyarlamalı modelde kuvvet yasası çekirdeği ve tek tip taban),
  2) mevcut (T0 öncesi) olayların ufuk içindeki doğrudan artçıları (sonlu kaynaklar kırık çizgi boyunca),
  3) tüm yeni olayların kaskad artçıları (dallanma süreci, nokta kaynak),
  büyüklükler GR (β, M_ref, üstten M_max=8.0 ile kesik).
Çıktılar (data/processed/tahmin/<etiket>/):
  bolge_olasiliklari.txt : bölge x eşik (M>=4, 5, 6): P(en az 1 olay), beklenen sayı, %5-%95 aralığı
  harita_M4.txt          : 0.2° hücrelerde 30 günde M>=4 olasılığı ve beklenen sayı
  ozet.txt
Not: belirsizlik aralıkları simülasyon (stokastik) belirsizliğidir; parametre belirsizliği ayrıca eklenmelidir.
Kullanım: python3 src/etas/simulate_forecast.py <config.json> <T0 YYYY-MM-DD> [H=30] [N_SIM=2000]
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
from etas.inversion import expected_aftershocks  # noqa
import evaluate_etas as E  # noqa
import residuals as RS  # noqa

M_MAX = 8.0
T_ORIGIN = pd.Timestamp("2010-01-01")
# Yaklaşık bölge kutuları (enlem_min, enlem_max, boylam_min, boylam_max)
REGIONS = {
    "Marmara": (40.0, 41.3, 26.5, 30.5),
    "Kuzey Ege - Çanakkale": (39.0, 40.5, 25.0, 27.0),
    "Batı Anadolu (Ege grabenleri)": (37.0, 39.5, 26.5, 30.5),
    "Güneybatı Ege (Bodrum-Datça)": (35.5, 37.0, 26.0, 29.0),
    "Doğu Anadolu Fay Zonu": (36.0, 39.0, 35.8, 40.5),
    "Kuzey Anadolu Fay Zonu (orta-doğu)": (39.3, 41.5, 30.5, 41.0),
    "Van - Doğu sınırı": (38.0, 40.5, 41.5, 45.0),
    "Doğu Akdeniz - Kıbrıs": (35.0, 36.0, 30.0, 36.0),
    "Tüm çalışma alanı": (35.0, 44.0, 25.0, 46.0),
}
THRESH = [4.0, 5.0, 6.0]


class TimeKernel:
    """f(t) ∝ e^{-t/τ} (t+c)^{-1-ω}; ters CDF ile [A,B] aralığında örnekleme."""
    def __init__(self, c, om, tau):
        self.grid = np.r_[0, np.geomspace(1e-7, 1e7, 6000)]
        f = np.exp(-self.grid / tau) / (self.grid + c) ** (1 + om)
        F = np.r_[0, np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(self.grid))]
        self.F = F / F[-1]

    def cdf(self, t):
        return np.interp(t, self.grid, self.F)

    def sample(self, A, B, rng):
        u = rng.uniform(self.cdf(A), self.cdf(B))
        return np.interp(u, self.F, self.grid)


def offset_latlon(lat, lon, dx, dy):
    return (lat + np.degrees(dy / 6378.137), lon + np.degrees(dx / (6378.137 * np.cos(np.radians(lat)))))


def sample_point_source(lat, lon, D, rho, rng):
    u = rng.uniform(size=len(lat))
    r = np.sqrt(D * ((1 - u) ** (-1 / rho) - 1))
    ang = rng.uniform(0, 2 * np.pi, len(lat))
    return offset_latlon(lat, lon, r * np.cos(ang), r * np.sin(ang))


def sample_finite(rr, n, D, rho, L, rng):
    """Kırık çizgi kaynağı: Z0/Z olasılıkla uç noktalarda nokta-kaynak, kalanında hat boyunca 1B Student-t ofset."""
    import finite_source as fs
    V = np.array([[float(a) for a in q.split(",")] for q in rr["kirik_cizgi"].split(";")]) if isinstance(rr.get("kirik_cizgi", ""), str) and rr.get("kirik_cizgi", "") \
        else np.array([[rr["lat1"], rr["lon1"]], [rr["lat2"], rr["lon2"]]])
    p_line = 1 - np.exp(-fs.log_Z_ratio(L, D, rho))
    on_line = rng.uniform(size=n) < p_line
    lat = np.empty(n); lon = np.empty(n)
    # hat parçası: uzunluk ağırlıklı segment seçimi
    lat0, lon0 = V[:, 0].mean(), V[:, 1].mean()
    vx, vy = fs._xy(V[:, 0], V[:, 1], lat0, lon0)
    seg = np.hypot(np.diff(vx), np.diff(vy)); k = on_line.sum()
    si = rng.choice(len(seg), size=k, p=seg / seg.sum()); t = rng.uniform(size=k)
    px = vx[si] + t * (vx[si + 1] - vx[si]); py = vy[si] + t * (vy[si + 1] - vy[si])
    nx, ny = -(vy[si + 1] - vy[si]) / seg[si], (vx[si + 1] - vx[si]) / seg[si]
    nu = 1 + 2 * rho; off = rng.standard_t(nu, size=k) * np.sqrt(D / nu)
    la, lo = offset_latlon(np.full(k, lat0), np.full(k, lon0), px + off * nx, py + off * ny)
    lat[on_line], lon[on_line] = la, lo
    # uç noktalar
    m = (~on_line).sum(); e = rng.integers(0, 2, m)
    ends = V[[0, -1]][e]
    lat[~on_line], lon[~on_line] = sample_point_source(ends[:, 0], ends[:, 1], np.full(m, D), rho, rng)
    return lat, lon


class Simulator:
    """Model bileşenlerini bir kez hazırlar; run(T0, H, N_SIM) ile sentetik katalog üretir."""
    def __init__(self, cfg_path, seed=1):
        self.rng = rng = np.random.default_rng(seed)
        C = RS.components(cfg_path)
        self.C = C
        cfg, th = C["cfg"], C["th"]
        self.cfg, self.th, self.beta, self.mref = cfg, th, C["beta"], C["mref"]
        k0, a, c, om, tau, d, g, rho = C["kp"]
        self.kp = C["kp"]
        self.TK = TimeKernel(c, om, tau)
        self.pars = [[th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho], self.mref]
        import select_background as SB
        md = ROOT / cfg["out_dir"]
        self.shape = None
        if cfg.get("bg_shape"):
            import longterm_background as LB
            f, area_s, (glat, glon, dens) = LB.load_shape()
            A = E.region_area(cfg["shape_coords"])
            self.rate_bg = 10 ** th["log10_mu"] * A
            ca = (111.2 * LB.GRID) * (111.2 * LB.GRID * np.cos(np.radians(glat)))[:, None] * np.ones((1, len(glon)))
            pc = (dens * ca).ravel(); self.shape = (glat, glon, pc / pc.sum(), LB.GRID)
            self.P = np.ones(1); self.latj = self.lonj = np.zeros(1)
        else:
            self.latj, self.lonj, self.P, Ttr = SB.training_targets(cfg, json.loads((md / "durum.json").read_text()), md)
            self.rate_bg = self.P.sum() / Ttr
        self.mix = None
        if cfg.get("bg_mix"):
            import longterm_background as LB
            f, area_s, (glat, glon, dens) = LB.load_shape()
            ca = (111.2 * LB.GRID) * (111.2 * LB.GRID * np.cos(np.radians(glat)))[:, None] * np.ones((1, len(glon)))
            pc = (dens * ca).ravel()
            self.mix = (json.loads((md / "arka_plan_karisim.json").read_text())["w"], (glat, glon, pc / pc.sum(), LB.GRID))
        if cfg.get("bg_adaptive"):
            bp = json.loads((md / "arka_plan_uyarlamali.json").read_text())
            self.h = SB.bandwidths(self.latj, self.lonj, int(bp["k"]), bp["h_min"]); self.w_unif = bp["w"]
        else:
            self.h = None; self.w_unif = 0.0; self.bw = np.sqrt(cfg.get("bw_sq", 2))
        poly = cfg["shape_coords"]
        self.box = (min(p[0] for p in poly), max(p[0] for p in poly), min(p[1] for p in poly), max(p[1] for p in poly))
        self.R = None
        if cfg.get("finite_source"):
            self.R = pd.read_csv(ROOT / cfg.get("ruptures", "data/processed/etas/rupturler.csv"), parse_dates=["time"])
        # dizi-özgü verimlilik (Bayesçi Gamma çarpanı; seq_update.py)
        self.seq = None
        if cfg.get("seq_nu"):
            import seq_update as SU
            self.seq = SU.Posterior(cfg_path, cfg["seq_nu"])

    def sample_bg(self, n):
        rng = self.rng
        if self.shape is not None:
            glat, glon, pc, G = self.shape
            k = rng.choice(len(pc), size=n, p=pc)
            i, j = np.divmod(k, len(glon))
            return glat[i] + rng.uniform(-G / 2, G / 2, n), glon[j] + rng.uniform(-G / 2, G / 2, n)
        j = rng.choice(len(self.P), size=n, p=self.P / self.P.sum())
        if self.h is not None:
            u = rng.uniform(size=n); r = self.h[j] * np.sqrt(1 / (1 - u) ** 2 - 1)
        else:
            r = self.bw * np.sqrt(-2 * np.log(rng.uniform(size=n)))
        ang = rng.uniform(0, 2 * np.pi, n)
        lat, lon = offset_latlon(self.latj[j], self.lonj[j], r * np.cos(ang), r * np.sin(ang))
        uni = rng.uniform(size=n) < self.w_unif
        la0, la1, lo0, lo1 = self.box
        lat[uni] = rng.uniform(la0, la1, uni.sum()); lon[uni] = rng.uniform(lo0, lo1, uni.sum())
        if self.mix is not None:
            w, (glat, glon, pc, G) = self.mix
            sel = rng.uniform(size=n) < w; m = sel.sum()
            k = rng.choice(len(pc), size=m, p=pc); i, j = np.divmod(k, len(glon))
            lat[sel] = glat[i] + rng.uniform(-G / 2, G / 2, m); lon[sel] = glon[j] + rng.uniform(-G / 2, G / 2, m)
        return lat, lon

    def sample_mag(self, n):
        u = self.rng.uniform(size=n)
        return self.mref - np.log(1 - u * (1 - np.exp(-self.beta * (M_MAX - self.mref)))) / self.beta

    def run(self, T0s, H=30.0, N_SIM=2000):
        rng = self.rng; C = self.C; S = C["S"]; xi = C["xi"]
        k0, a, c, om, tau, d, g, rho = self.kp; mref = self.mref; pars = self.pars; TK = self.TK
        T0 = (pd.Timestamp(T0s) - T_ORIGIN).total_seconds() / 86400; T1 = T0 + H
        past = S.t.values < T0
        Sp = S[past].reset_index(drop=True); xip = xi[past]
        n_exp = expected_aftershocks([Sp.magnitude.values, T0 - Sp.t.values, T1 - Sp.t.values], pars) * xip
        keep = n_exp > 1e-7
        Sp, n_exp = Sp[keep].reset_index(drop=True), n_exp[keep]
        D_p = d * np.exp(g * (Sp.magnitude.values - mref))
        fsmap = {}
        import finite_source as fs
        if self.R is not None:
            for j, kk in enumerate(fs.match_sources(Sp.time.values, Sp.magnitude.values, self.R)):
                if kk >= 0:
                    fsmap[j] = self.R.iloc[kk]
        ob = self.cfg.get("omega_big")
        if ob is not None and fsmap:
            zmp = fs.zone_mask(Sp.t.values, Sp.latitude.values, Sp.longitude.values, self.R, self.cfg.get("zone_km"), self.cfg.get("zone_days"), T_ORIGIN)
            jb = np.array(sorted(set(fsmap) | set(np.nonzero(zmp)[0])), int)
            Gb = fs.big_expected(Sp.magnitude.values[jb], T0 - Sp.t.values[jb], T1 - Sp.t.values[jb], pars[0], mref, ob, expected_aftershocks)
            n_exp = n_exp.copy(); n_exp[jb] = Gb * C["xi"][past][keep][jb]
            if not hasattr(self, "TKb"):
                self.TKb = TimeKernel(c, ob, tau)
        if self.seq is not None:
            if T0 > self.seq.t_last + 1:
                print(f"UYARI: dizi hazırlığı {self.seq.t_last:.0f}. güne kadar; T0 sonrası olaylar sayılmıyor", flush=True)
            jj = np.nonzero(past)[0][keep]
            A_, B_ = self.seq.at(T0, jj)
            Gm = rng.gamma(A_[None, :], 1.0 / B_[None, :], (N_SIM, len(jj)))
            lam_mat = n_exp[None, :] * Gm
        else:
            lam_mat = np.broadcast_to(n_exp[None, :], (N_SIM, len(n_exp)))
        nb = rng.poisson(self.rate_bg * H, N_SIM)
        sid = np.repeat(np.arange(N_SIM), nb); n = len(sid)
        la, lo = self.sample_bg(n)
        gen = pd.DataFrame(dict(sim=sid, t=rng.uniform(T0, T1, n), lat=la, lon=lo, m=self.sample_mag(n)))
        cnt = rng.poisson(lam_mat)
        sim_i, src_i = np.nonzero(cnt); reps = cnt[sim_i, src_i]
        sim_e = np.repeat(sim_i, reps); src_e = np.repeat(src_i, reps); n = len(src_e)
        tt = Sp.t.values[src_e] + TK.sample(T0 - Sp.t.values[src_e], T1 - Sp.t.values[src_e], rng)
        if ob is not None and fsmap:
            bsel = np.isin(src_e, jb)
            if bsel.any():
                tt[bsel] = Sp.t.values[src_e[bsel]] + self.TKb.sample(T0 - Sp.t.values[src_e[bsel]], T1 - Sp.t.values[src_e[bsel]], rng)
        la = np.empty(n); lo = np.empty(n)
        pt = np.array([s not in fsmap for s in src_e], bool)
        la[pt], lo[pt] = sample_point_source(Sp.latitude.values[src_e[pt]], Sp.longitude.values[src_e[pt]], D_p[src_e[pt]], rho, rng)
        for j, rr in fsmap.items():
            sel = src_e == j
            if sel.any():
                la[sel], lo[sel] = sample_finite(rr, int(sel.sum()), D_p[j], rho, float(rr["L"]), rng)
        gen = pd.concat([gen, pd.DataFrame(dict(sim=sim_e, t=tt, lat=la, lon=lo, m=self.sample_mag(n)))], ignore_index=True)
        allev = [gen]; ngen = 0
        zk, zd = self.cfg.get("zone_km"), self.cfg.get("zone_days")
        while len(gen) and ngen < 60:
            ne = expected_aftershocks([gen.m.values, np.zeros(len(gen)), T1 - gen.t.values], pars)
            zg = np.zeros(len(gen), bool)
            if ob is not None and self.R is not None and zk:
                zg = fs.zone_mask(gen.t.values, gen.lat.values, gen.lon.values, self.R, zk, zd, T_ORIGIN)
                if zg.any():
                    ne[zg] = fs.big_expected(gen.m.values[zg], np.zeros(zg.sum()), T1 - gen.t.values[zg], pars[0], mref, ob, expected_aftershocks)
            if self.seq is not None:  # yeni olayların verimlilik çarpanı ~ Gamma(ν, ν)
                ne = ne * rng.gamma(self.seq.nu, 1.0 / self.seq.nu, len(ne))
            k = rng.poisson(ne)
            if k.sum() == 0:
                break
            par = np.repeat(np.arange(len(gen)), k)
            tt = gen.t.values[par] + TK.sample(np.zeros(len(par)), T1 - gen.t.values[par], rng)
            if zg.any():
                pz = zg[par]
                if pz.any():
                    if not hasattr(self, "TKb"):
                        self.TKb = TimeKernel(c, ob, tau)
                    tt[pz] = gen.t.values[par][pz] + self.TKb.sample(np.zeros(pz.sum()), T1 - gen.t.values[par][pz], rng)
            Dn = d * np.exp(g * (gen.m.values[par] - mref))
            la, lo = sample_point_source(gen.lat.values[par], gen.lon.values[par], Dn, rho, rng)
            gen = pd.DataFrame(dict(sim=gen.sim.values[par], t=tt, lat=la, lon=lo, m=self.sample_mag(len(par))))
            allev.append(gen); ngen += 1
        ev = pd.concat(allev, ignore_index=True)
        info = dict(ngen=ngen, bg=self.rate_bg * H, direct=float(n_exp.sum()))
        return ev, info


def main(cfg_path, T0s, H=30.0, N_SIM=2000, seed=1):
    t_start = time.time()
    SIM = Simulator(cfg_path, seed)
    ev, info = SIM.run(T0s, H, N_SIM)
    cfg, mref = SIM.cfg, SIM.mref; ngen = info["ngen"]
    # --- çıktılar ---
    out = ROOT / "data/processed/tahmin" / f"{Path(cfg_path).stem}_{T0s}_{int(H)}g"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (a0, a1, b0, b1) in REGIONS.items():
        inr = ev.lat.between(a0, a1) & ev.lon.between(b0, b1)
        for thr in THRESH:
            cnts = ev[inr & (ev.m >= thr)].groupby("sim").size().reindex(range(N_SIM), fill_value=0).values
            rows.append(dict(bolge=name, esik=f"M>={thr:.0f}", olasilik=round((cnts > 0).mean(), 4),
                             beklenen=round(cnts.mean(), 3), q05=int(np.percentile(cnts, 5)), q95=int(np.percentile(cnts, 95))))
    tab = pd.DataFrame(rows); tab.to_csv(out / "bolge_olasiliklari.txt", sep="\t", index=False)
    e4 = ev[ev.m >= 4.0]
    gl = np.floor(e4.lat / 0.2) * 0.2; go = np.floor(e4.lon / 0.2) * 0.2
    cell = e4.assign(gl=gl.round(1), go=go.round(1)).groupby(["gl", "go"])
    mp = pd.DataFrame({"beklenen_M4": cell.size() / N_SIM, "olasilik_M4": cell.sim.nunique() / N_SIM}).reset_index()
    mp.rename(columns={"gl": "enlem", "go": "boylam"}).to_csv(out / "harita_M4.txt", sep="\t", index=False)
    L = [f"QuakeCast-TR olasılıksal tahmin — model: {cfg['name']}",
         f"Başlangıç: {T0s}, ufuk: {int(H)} gün, simülasyon sayısı: {N_SIM}, kaskad nesli: {ngen}, süre {time.time()-t_start:.0f} s",
         f"Arka plan hızı: {info['bg']:.1f} olay (M>={mref}) / {int(H)} gün; mevcut olaylardan beklenen doğrudan artçı: {info['direct']:.1f}",
         "", tab.to_string(index=False), "",
         "Not: aralıklar yalnızca stokastik (simülasyon) belirsizliği yansıtır; bölgeler yaklaşık enlem-boylam kutularıdır."]
    (out / "ozet.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], float(a[3]) if len(a) > 3 else 30.0, int(a[4]) if len(a) > 4 else 2000)
