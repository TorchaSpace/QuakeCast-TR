"""Prospektif tahminlerin değerlendirilmesi ve defter (prospektif/DEFTER.csv).

Her tahmin penceresi iki aşamada puanlanır:
  on    : pencere bittikten >= 3 gün sonra, o anki canlı katalogla (son aylarda yalnız AFAD; Kandilli ~2 ay gecikmeli)
  kesin : Kandilli kataloğu pencere sonunu kapsadığında (birleşik katalog), bir kez
Testler (pyCSEP katalog-tabanlı tanımlar, csep_tests.window_tests ile aynı kod): N (δ1, δ2), M, S, PL.
Ayrıca 0.1° hücrelerde Poisson log-olabilirliğiyle, zamandan bağımsız uzun dönem referansına (1900–2013 yumuşatılmış harita ×
2014–pencere başı ortalama oranı) göre olay başına bilgi kazancı (IGPE).
Kullanım: python3 src/prospective/evaluate_forecasts.py
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import csep_tests as CT  # noqa: E402
import longterm_background as LB  # noqa: E402

PK = ROOT / "data/processed/prospektif"
DEF = ROOT / "prospektif/DEFTER.csv"
BOX = CT.BOX
FLOOR = 1e-5  # hücre başına beklenen sayıya taban (log 0'ı önler; csep_tests ile aynı)


def ref_cells(T0, H, cat):
    """Zamandan bağımsız referans: 0.1° hücre beklentisi = R · H · ∫_hücre f."""
    f, area, (glat, glon, dens) = LB.load_shape()
    G = LB.GRID
    ca = (111.2 * G) * (111.2 * G * np.cos(np.radians(glat)))[:, None] * np.ones((1, len(glon)))
    p = dens * ca; p = p / p.sum()
    LA, LO = np.meshgrid(glat, glon, indexing="ij")
    ci = CT.cell_idx(LA.ravel(), LO.ravel())
    pc = np.bincount(ci, weights=p.ravel(), minlength=CT.NLAT * CT.NLON)
    t = pd.to_datetime(cat.time)
    hist = cat[(t >= "2014-01-01") & (t < T0)]
    R = len(hist) / (T0 - pd.Timestamp("2014-01-01")).days
    return R * H * pc


def evaluate_one(d, stage, cat, rng):
    meta = json.load(open(d / "meta.json"))
    T0 = pd.Timestamp(meta["pencere_baslangic"]); T1 = pd.Timestamp(meta["pencere_bitis"]); H = meta["ufuk_gun"]
    z = np.load(d / "sentetik_kataloglar.npz")
    n_sim = int(z["n_sim"])
    ev = pd.DataFrame({"sim": z["sim"], "lat": z["enlem"], "lon": z["boylam"], "m": z["M"]})
    t = pd.to_datetime(cat.time)
    obs = cat[(t >= T0) & (t < T1)]
    CT.N_SIM = n_sim
    r = CT.window_tests(ev, obs, rng)
    # hücre bazlı Poisson olabilirliği ve referansa göre kazanç
    cid = CT.cell_idx(ev.lat.values, ev.lon.values)
    lam_m = np.bincount(cid, minlength=CT.NLAT * CT.NLON) / n_sim + FLOOR
    lam_r = ref_cells(T0, H, cat[(cat.magnitude >= 3.45) & cat.latitude.between(BOX[0], BOX[1]) & cat.longitude.between(BOX[2], BOX[3])]) + FLOOR
    co = CT.cell_idx(obs.latitude.values, obs.longitude.values)
    LLm = np.log(lam_m[co]).sum() - lam_m.sum(); LLr = np.log(lam_r[co]).sum() - lam_r.sum()
    N = len(obs)
    row = dict(baslangic=str(T0.date()), ufuk_gun=H, asama=stage, degerlendirme_utc=str(pd.Timestamp.utcnow().tz_localize(None))[:19],
               gec_yayin=meta["gec_yayin"], N_gozlenen=N, N_medyan=r["N_medyan"], N_q025=r["N_q025"], N_q975=r["N_q975"],
               delta1=round(r["delta1"], 4), delta2=round(r["delta2"], 4),
               N_gecti_95=bool(min(r["delta1"], r["delta2"]) >= 0.025), N_gecti_99=bool(min(r["delta1"], r["delta2"]) >= 0.005),
               gamma_M=r.get("gamma_M"), gamma_S=r.get("gamma_S"), gamma_PL=r.get("gamma_PL"),
               LL_model=round(LLm, 2), LL_referans=round(LLr, 2), IGPE_referansa_gore=round((LLm - LLr) / N, 3) if N else np.nan,
               en_buyuk_M=float(obs.magnitude.max()) if N else np.nan)
    return row


def main():
    info = json.load(open(PK / "etas_girdi_canli.json"))
    kand_end = pd.Timestamp(info["kandilli_son"])
    cat = pd.read_csv(PK / "etas_girdi_canli.csv")
    cat["magnitude"] = np.round(cat.magnitude, 1)
    cat = cat[(cat.magnitude >= 3.45) & cat.latitude.between(BOX[0], BOX[1]) & cat.longitude.between(BOX[2], BOX[3])]
    last = pd.to_datetime(cat.time).max()
    led = pd.read_csv(DEF) if DEF.exists() else pd.DataFrame(columns=["baslangic", "ufuk_gun", "asama"])
    done = set(zip(led.baslangic.astype(str), led.ufuk_gun.astype(int), led.asama))
    rows = led.to_dict("records"); rng = np.random.default_rng(7)
    for d in sorted((ROOT / "prospektif/tahminler").glob("*_*g")):
        meta = json.load(open(d / "meta.json"))
        T0 = pd.Timestamp(meta["pencere_baslangic"]); T1 = pd.Timestamp(meta["pencere_bitis"]); H = int(meta["ufuk_gun"])
        for stage, ready in [("on", last >= T1 + pd.Timedelta(days=3) or pd.Timestamp.utcnow().tz_localize(None) >= T1 + pd.Timedelta(days=3)),
                             ("kesin", kand_end >= T1)]:
            if ready and (str(T0.date()), H, stage) not in done:
                rows.append(evaluate_one(d, stage, cat, rng)); print(rows[-1], flush=True)
    if rows:
        pd.DataFrame(rows).sort_values(["baslangic", "ufuk_gun", "asama"]).to_csv(DEF, index=False)
    print(f"defter: {len(rows)} satır")


if __name__ == "__main__":
    main()
