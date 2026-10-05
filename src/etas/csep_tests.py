"""CSEP katalog-tabanlı tutarlılık testleri (Savran vd. 2020; pyCSEP 'Theory of CSEP Tests' tanımları).

Ardışık 30 günlük pencerelerde (2022-01 .. 2026-06) model N_SIM sentetik katalog üretir; gözlenen katalog
(M >= M_TEST, çalışma alanı) şu testlerle karşılaştırılır:
  N-testi : δ1 = P(N_sim >= N_obs), δ2 = P(N_sim <= N_obs)
  M-testi : büyüklük histogramı (0.1 kutu) d_obs vs D_j, γ_m = P(D_j <= d_obs)
  S-testi : 0.1° hücrelerde normalize uzaysal yoğunluk, γ_s = P(S_j <= S_obs)
  PL-testi: sözde-olabilirlik, γ_L = P(L_j <= L_obs)
Hücre yoğunluğu: tüm simülasyonların birleşimi / N_SIM + 1e-5 (log(0) önlemek için taban, belgelenmiş seçim).
Simülasyon istatistikleri (S_j, L_j) kendisi hariç (leave-one-out) birleşik yoğunlukla hesaplanır; aksi halde
sınırlı simülasyon sayısında her katalog kendi olaylarını içeren yoğunlukla puanlanıp yapay olarak iyi görünür.
Başarısızlık: N için δ1<0.025 veya δ2<0.025; M, S, PL için γ<0.05 (tek taraflı, alt kuyruk).
Kullanım: python3 src/etas/csep_tests.py <config.json> [etiket]
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import simulate_forecast as SF  # noqa
import evaluate_etas as E  # noqa

H = 30; N_SIM = 400; M_TEST = 3.5; DM = 0.1; CELL = 0.1
BOX = (35.0, 44.0, 25.0, 46.0)
MBINS = np.round(np.arange(M_TEST - 0.05, 8.05 + 1e-9, DM), 2)
NLAT = int(round((BOX[1] - BOX[0]) / CELL)); NLON = int(round((BOX[3] - BOX[2]) / CELL))


def cell_idx(lat, lon):
    i = np.clip(((lat - BOX[0]) / CELL).astype(int), 0, NLAT - 1); j = np.clip(((lon - BOX[2]) / CELL).astype(int), 0, NLON - 1)
    return i * NLON + j


def window_tests(ev, obs, rng):
    ev = ev[(ev.m >= M_TEST - 0.05) & ev.lat.between(BOX[0], BOX[1]) & ev.lon.between(BOX[2], BOX[3])]
    Nj = ev.groupby("sim").size().reindex(range(N_SIM), fill_value=0).values
    No = len(obs)
    res = dict(N_obs=No, N_medyan=float(np.median(Nj)), N_q025=float(np.percentile(Nj, 2.5)), N_q975=float(np.percentile(Nj, 97.5)),
               delta1=float((Nj >= No).mean()), delta2=float((Nj <= No).mean()))
    # uzaysal yoğunluk (birleşim)
    cid = cell_idx(ev.lat.values, ev.lon.values)
    lam = np.bincount(cid, minlength=NLAT * NLON) / N_SIM + 1e-5
    lam_n = lam / lam.sum(); Nbar = Nj.mean()
    sims = ev.sim.values
    # kendisi hariç (LOO) birleşik yoğunluk: simülasyon j'nin olayları j'nin kendi katkısı çıkarılmış yoğunlukla puanlanır
    key = sims.astype(np.int64) * (NLAT * NLON) + cid
    uk, inv, cnt_jc = np.unique(key, return_inverse=True, return_counts=True)
    tot_c = np.bincount(cid, minlength=NLAT * NLON)
    lam_loo = (tot_c[cid] - cnt_jc[inv]) / (N_SIM - 1) + 1e-5
    lam_loo_sum = (tot_c.sum() - Nj[sims]) / (N_SIM - 1) + 1e-5 * NLAT * NLON
    logl = np.log(lam_loo / lam_loo_sum); loglam = np.log(lam_loo)
    S_j = np.bincount(sims, weights=logl, minlength=N_SIM) / np.maximum(Nj, 1)
    L_j = np.bincount(sims, weights=loglam, minlength=N_SIM) - Nbar
    if No > 0:
        co = cell_idx(obs.latitude.values, obs.longitude.values)
        S_o = np.log(lam_n[co]).mean(); L_o = np.log(lam[co]).sum() - Nbar
        ok = Nj > 0
        res.update(gamma_S=float((S_j[ok] <= S_o).mean()), gamma_PL=float((L_j <= L_o).mean()))
        # M-testi
        hU = np.histogram(ev.m.values, MBINS)[0].astype(float); NU = hU.sum()
        hO = np.histogram(obs.magnitude.values, MBINS)[0].astype(float)
        ref = np.log(No / NU * hU + 1)
        d_obs = ((ref - np.log(hO + 1)) ** 2).sum()
        D = np.empty(N_SIM)
        mbin = np.clip(np.digitize(ev.m.values, MBINS) - 1, 0, len(MBINS) - 2)
        H2 = np.zeros((N_SIM, len(MBINS) - 1)); np.add.at(H2, (sims, mbin), 1)
        sc = np.where(Nj > 0, No / np.maximum(Nj, 1), 0)[:, None]
        D = ((ref[None] - np.log(sc * H2 + 1)) ** 2).sum(1)
        res["gamma_M"] = float((D[Nj > 0] <= d_obs).mean())
    else:
        res.update(gamma_S=np.nan, gamma_PL=np.nan, gamma_M=np.nan)
    return res


def main(cfg, tag=None):
    tag = tag or Path(cfg).stem
    cat = pd.read_csv(ROOT / "data/processed/etas_girdi_2010_M25.csv", parse_dates=["time"])
    cat["magnitude"] = np.round(cat.magnitude, 1)
    cat = cat[(cat.magnitude >= M_TEST - 0.05) & cat.latitude.between(BOX[0], BOX[1]) & cat.longitude.between(BOX[2], BOX[3])]
    out = ROOT / f"data/processed/etas/csep/{tag}.txt"; out.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(out, sep="\t").to_dict("records") if out.exists() else []
    done = {r["baslangic"] for r in rows}
    starts = [s for s in pd.date_range("2022-01-01", "2026-06-30", freq=f"{H}D") if str(s.date()) not in done]
    t0 = time.time(); S = None; rng = np.random.default_rng(5)
    for T0 in starts:
        if time.time() - t0 > 120:
            print("SURE DOLDU"); break
        if S is None:
            import json as _j
            spec = _j.load(open(cfg))
            if "ensemble" in spec:  # ağırlıklı topluluk: her model N_SIM·w senaryo
                S = [(SF.Simulator(c_, seed=21 + q), w_) for q, (c_, w_) in enumerate(spec["ensemble"])]
            else:
                S = [(SF.Simulator(cfg, seed=21), 1.0)]
        parts_ = []; off = 0
        for sim_, w_ in S:
            n_ = int(round(N_SIM * w_)) if len(S) > 1 else N_SIM
            e_, _ = sim_.run(str(T0.date()), H, n_); e_["sim"] = e_["sim"] + off; off += n_; parts_.append(e_)
        ev = pd.concat(parts_, ignore_index=True)
        ev["m"] = np.round(ev.m, 1)
        obs = cat[(cat.time >= T0) & (cat.time < T0 + pd.Timedelta(days=H))]
        r = dict(baslangic=str(T0.date()), **window_tests(ev, obs, rng))
        rows.append(r); pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    df = pd.DataFrame(rows)
    if len(df) >= len(pd.date_range("2022-01-01", "2026-06-30", freq=f"{H}D")):
        fN = ((df.delta1 < 0.025) | (df.delta2 < 0.025)).mean()
        L = [f"CSEP katalog-tabanlı testler — {tag} ({len(df)} pencere x {H} gün, M>={M_TEST}, {N_SIM} simülasyon)",
             f"  N-testi başarısız oranı: %{100*fN:.0f}  (fazla tahmin: %{100*(df.delta2<0.025).mean():.0f}, eksik tahmin: %{100*(df.delta1<0.025).mean():.0f}); medyan N_sim/N_obs = {np.median(df.N_medyan/np.maximum(df.N_obs,1)):.2f}",
             f"  M-testi başarısız oranı: %{100*(df.gamma_M<0.05).mean():.0f}  (ortalama γ_M {df.gamma_M.mean():.2f})",
             f"  S-testi başarısız oranı: %{100*(df.gamma_S<0.05).mean():.0f}  (ortalama γ_S {df.gamma_S.mean():.2f})",
             f"  PL-testi başarısız oranı: %{100*(df.gamma_PL<0.05).mean():.0f}  (ortalama γ_PL {df.gamma_PL.mean():.2f})",
             "  (Rastgele bir doğru modelde beklenen: N ~%5, diğerleri ~%5)"]
        Path(str(out).replace(".txt", "_OZET.txt")).write_text("\n".join(L) + "\n", encoding="utf-8")
        print("\n".join(L))
    else:
        print(f"{len(df)} pencere tamam")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
