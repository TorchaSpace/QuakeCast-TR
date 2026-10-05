"""Referans ETAS modelinin artık analizi (eğitim dışı 2022-01-01 .. 2026-08-01).

1) Dönüştürülmüş zaman: τ_i = Λ(t_i) (tüm bölge kompanzatörü); ardışık farklar Exp(1) olmalı -> KS testi,
   ayrıca Λ(t) - N(t) eğrisi (kümülatif fazla/eksik tahmin).
2) Mekânsal: 1°x1° hücrelerde gözlenen vs beklenen (tetiklenme beklentisi kaynağın hücresine,
   arka plan μ(x) ızgarada) -> Pearson artıkları; en kötü hücreler.
3) En düşük λ'lı (en "sürpriz") test olayları.
Çıktılar: data/processed/etas/artiklar/
"""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.environ.get("ETAS_LIB", str(Path.home() / "etasrepo")))
sys.path.insert(0, str(ROOT / "src" / "etas")); sys.path.insert(0, str(ROOT / "src" / "catalog"))
from etas.inversion import expected_aftershocks, responsibility_factor  # noqa
import evaluate_etas as E  # noqa
from mc_field import McField  # noqa

T_ORIGIN = pd.Timestamp("2010-01-01"); M_TEST = 3.5
W0, W1 = "2022-01-01", "2026-08-01"


def components(mdir):
    cfg, st, mdir = E.load_model(mdir)
    th = st["theta"]; beta = st["beta"]; dm = cfg["delta_m"]
    var = cfg["mc"] == "var"; mref = cfg["m_ref"] if var else cfg["mc"]
    cat = pd.read_csv(ROOT / "data/processed/etas_girdi_2010_M25.csv", parse_dates=["time"])
    cat["magnitude"] = np.floor(cat.magnitude / dm + 0.5) * dm
    tn = (cat.time - T_ORIGIN).dt.total_seconds().values / 86400
    F = McField(pd.DataFrame({"time": tn, "latitude": cat.latitude, "longitude": cat.longitude, "magnitude": cat.magnitude}))
    cat["mcf"] = F(tn, cat.latitude.values, cat.longitude.values); cat["t"] = tn
    ev = cat[(cat.magnitude >= M_TEST - dm / 2) & (cat.time >= cfg["auxiliary_start"])]
    ev = ev[E.in_poly(ev.latitude.values, ev.longitude.values, cfg["shape_coords"])].sort_values("t").reset_index(drop=True)
    ev["gozlenebilir"] = ev.magnitude.values >= np.ceil(ev.mcf.values * 10 - 1e-6) / 10 - dm / 2
    S = ev[ev.gozlenebilir].reset_index(drop=True) if var else ev.copy()
    a, g, rho, om = th["a"], th["gamma"], th["rho"], th["omega"]
    k0, c, tau, d = 10 ** th["log10_k0"], 10 ** th["log10_c"], 10 ** th["log10_tau"], 10 ** th["log10_d"]
    tharr = np.array([th["log10_mu"], np.nan, th["log10_k0"], a, th["log10_c"], om, th["log10_tau"], th["log10_d"], g, rho])
    xi = responsibility_factor(tharr, beta, np.maximum(np.ceil(S.mcf.values * 10 - 1e-6) / 10 - mref, 0)) if var else np.ones(len(S))
    sc = np.exp(-beta * (M_TEST - mref))
    mu_x, mu_tot = E.background_fn(cfg, st, mdir, pd.read_csv(ROOT / cfg["catalog"], parse_dates=["time"]))
    fs_cols = []
    if cfg.get("finite_source"):
        import finite_source as fs
        R = pd.read_csv(ROOT / cfg.get("ruptures", "data/processed/etas/rupturler.csv"), parse_dates=["time"])
        zone = d * np.exp(g * (S.magnitude.values - mref))
        for j, k in enumerate(fs.match_sources(S.time.values, S.magnitude.values, R)):
            if k >= 0:
                fs_cols.append((j, R.iloc[k], float(np.exp(-fs.log_Z_ratio(R.L.values[k], zone[j], rho)))))
    return dict(cfg=cfg, th=th, beta=beta, ev=ev, S=S, xi=xi, sc=sc, mu_x=mu_x, mu_tot=mu_tot, mref=mref,
                kp=(k0, a, c, om, tau, d, g, rho), fs_cols=fs_cols)


def lam_at(C, t, lat, lon):
    S = C["S"]; k0, a, c, om, tau, d, g, rho = C["kp"]
    prod = k0 * np.exp(a * (S.magnitude.values - C["mref"])) * C["xi"]; zone = d * np.exp(g * (S.magnitude.values - C["mref"]))
    out = np.empty(len(t))
    for s in range(0, len(t), 300):
        sl = slice(s, s + 300)
        dt = t[sl, None] - S.t.values[None]; msk = dt > 0; dtp = np.where(msk, dt, 1.0)
        r2 = E.hav_sq(lat[sl, None], lon[sl, None], S.latitude.values[None], S.longitude.values[None])
        kap = np.ones(len(S))
        if C.get("fs_cols"):
            import finite_source as fs
            for j, rr, kp in C["fs_cols"]:
                r2[:, j] = fs.dist2_to_rupture(lat[sl], lon[sl], rr); kap[j] = kp
        out[sl] = (kap[None] * prod[None] * np.exp(-dtp / tau) / (dtp + c) ** (1 + om) / (r2 + zone[None]) ** (1 + rho) * msk).sum(1)
    return (out + C["mu_x"](lat, lon)) * C["sc"]


def main(mdir):
    out = ROOT / "data/processed/etas/artiklar" / Path(mdir).name.replace(".json", ""); out.mkdir(parents=True, exist_ok=True)
    C = components(mdir); th = C["th"]; S = C["S"]; ev = C["ev"]
    T0 = (pd.Timestamp(W0) - T_ORIGIN).days; T1 = (pd.Timestamp(W1) - T_ORIGIN).days
    te = ev[(ev.t >= T0) & (ev.t < T1) & ev.gozlenebilir].reset_index(drop=True)
    # 1) dönüştürülmüş zaman
    bg_rate = (C["mu_tot"] if C["mu_tot"] is not None else 10 ** th["log10_mu"] * E.region_area(C["cfg"]["shape_coords"])) * C["sc"]
    pars = [[th["log10_k0"], th["a"], th["log10_c"], th["omega"], th["log10_tau"], th["log10_d"], th["gamma"], th["rho"]], C["mref"]]
    Lam = np.empty(len(te))
    for i, ti in enumerate(te.t.values):
        sel = S.t.values < ti
        ea = expected_aftershocks([S.magnitude.values[sel], np.maximum(T0 - S.t.values[sel], 0), ti - S.t.values[sel]], pars)
        Lam[i] = (ea * C["xi"][sel]).sum() * C["sc"] + bg_rate * (ti - T0)
    dtau = np.diff(np.r_[0, Lam])
    ks = stats.kstest(dtau, "expon")
    cum = pd.DataFrame({"zaman": pd.to_datetime(te.time.values), "Lambda": Lam, "N": np.arange(1, len(te) + 1)})
    cum.to_csv(out / "donusmus_zaman.txt", sep="\t", index=False)
    # aylık beklenen vs gözlenen
    cum["ay"] = cum.zaman.dt.to_period("M")
    mon = cum.groupby("ay").agg(gozlenen=("N", "size"), Lambda_son=("Lambda", "max"))
    mon["beklenen"] = mon.Lambda_son.diff().fillna(mon.Lambda_son.iloc[0])
    # 2) mekânsal hücreler
    te["lam"] = lam_at(C, te.t.values, te.latitude.values, te.longitude.values)
    sel = S.t.values < T1
    ea = expected_aftershocks([S.magnitude.values[sel], np.maximum(T0 - S.t.values[sel], 0), T1 - S.t.values[sel]], pars) * C["xi"][sel] * C["sc"]
    Ssel = S[sel].assign(ea=ea)
    cells = {}
    for (la, lo), v in Ssel.groupby([np.floor(Ssel.latitude), np.floor(Ssel.longitude)]).ea.sum().items():
        cells[(la, lo)] = cells.get((la, lo), 0) + v
    glat, glon = np.meshgrid(np.arange(35.05, 44, 0.1), np.arange(25.05, 46, 0.1), indexing="ij")
    mu_g = C["mu_x"](glat.ravel(), glon.ravel()) * C["sc"] * (T1 - T0)
    cellarea = (111.2 * 0.1) * (111.2 * 0.1 * np.cos(np.radians(glat.ravel())))
    for la, lo, v in zip(np.floor(glat.ravel()), np.floor(glon.ravel()), mu_g * cellarea):
        cells[(la, lo)] = cells.get((la, lo), 0) + v
    obs = te.groupby([np.floor(te.latitude), np.floor(te.longitude)]).size()
    rows = [dict(enlem=k[0], boylam=k[1], beklenen=v, gozlenen=int(obs.get(k, 0))) for k, v in cells.items()]
    cm = pd.DataFrame(rows); cm["pearson"] = (cm.gozlenen - cm.beklenen) / np.sqrt(cm.beklenen.clip(lower=0.5))
    cm["poisson_p_alt"] = stats.poisson.cdf(cm.gozlenen, cm.beklenen)
    cm["poisson_p_ust"] = stats.poisson.sf(cm.gozlenen - 1, cm.beklenen)
    cm.sort_values("pearson").to_csv(out / "hucre_artiklari.txt", sep="\t", index=False)
    # 3) sürpriz olaylar: λ'nın bölge ortalama hızına oranı
    te["surpriz"] = -np.log(te.lam)
    te.sort_values("surpriz", ascending=False).head(40)[["time", "latitude", "longitude", "magnitude", "lam", "surpriz"]].to_csv(out / "surpriz_olaylar.txt", sep="\t", index=False)
    L = [f"Artık analizi — model: {C['cfg']['name']}  (test {W0}..{W1}, gözlenebilir M>={M_TEST}: {len(te)} olay)", "",
         f"Toplam beklenen (maske düzeltmesiz): {Lam[-1]:.0f}  gözlenen: {len(te)}",
         f"Dönüştürülmüş zaman aralıkları ~ Exp(1)? KS istatistiği {ks.statistic:.3f}, p={ks.pvalue:.2g}  (ortalama {dtau.mean():.3f}, std {dtau.std():.3f})",
         "", "Aylık gözlenen / beklenen (en büyük sapmalar):"]
    mon["oran"] = mon.gozlenen / mon.beklenen
    L += ["  " + s for s in mon.assign(fark=mon.gozlenen - mon.beklenen).reindex(mon.assign(f=(mon.gozlenen - mon.beklenen).abs()).sort_values("f", ascending=False).index[:10]).round(1).to_string().splitlines()]
    L += ["", "Mekânsal (1° hücre) en çok EKSİK tahmin edilen hücreler (gözlenen >> beklenen):"]
    L += ["  " + s for s in cm.sort_values("pearson", ascending=False).head(10).round(3).to_string(index=False).splitlines()]
    L += ["", "En çok FAZLA tahmin edilen hücreler (gözlenen << beklenen):"]
    L += ["  " + s for s in cm.sort_values("pearson").head(10).round(3).to_string(index=False).splitlines()]
    L += ["", f"Hücrelerin %{100*((cm.poisson_p_alt<0.025)|(cm.poisson_p_ust<0.025)).mean():.0f}'i Poisson %95 aralığı dışında (beklenen ~%5)"]
    L += ["", "En sürpriz 15 olay (en düşük λ):"] + ["  " + s for s in te.sort_values("surpriz", ascending=False).head(15)[["time", "latitude", "longitude", "magnitude", "surpriz"]].round(3).to_string(index=False).splitlines()]
    (out / "ARTIK_RAPORU.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
