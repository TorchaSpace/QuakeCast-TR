"""Prospektif tahmin yayını (dondurulmuş v6 topluluk modeli).

Kullanım: python3 src/prospective/issue_forecast.py <pencere_baslangici YYYY-MM-DD> [ufuklar=7,30] [N_SIM=5000]
Adımlar (her biri önbellekli; zaman sınırlı ortamlarda tekrar çağrılarak tamamlanır):
  1) Dondurulmuş model ve kod dosyalarının SHA256 doğrulaması (prospektif/MODEL_V6_MANIFEST.json); uyuşmazlıkta durur.
  2) Canlı katalog pencere başlangıcından ÖNCEKİ olaylarla kesilir (ileriye bakış yok).
  3) Dizi sonsalı hazırlıkları (iki üye model), sonra simülasyon (üye ağırlıklarıyla N_SIM senaryo).
  4) Çıktılar: prospektif/tahminler/<başlangıç>_<H>g/ (özet, bölge olasılıkları, 0.1° hücre beklentileri,
     sayı dağılımı, sentetik kataloglar) + meta.json (yayın zamanı, veri kesimi, hash'ler).
Hedef: Mw >= 3.5 (0.1 yuvarlama), 35–44K / 25–46D, olay türü deprem.
"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PK = ROOT / "data/processed/prospektif"
MAN = ROOT / "prospektif/MODEL_V6_MANIFEST.json"
ENS = "configs/v6_topluluk.json"
BOX = (35.0, 44.0, 25.0, 46.0)


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def verify():
    M = json.load(open(MAN))
    bad = [p for p, h in M["dosyalar"].items() if not (ROOT / p).exists() or sha(p) != h]
    if bad:
        raise SystemExit("DONDURULMUŞ MODEL/KOD DEĞİŞMİŞ — yayın durduruldu: " + ", ".join(bad))
    return M


def main(start, horizons=(7, 30), n_sim=5000):
    t_start = time.time(); budget = float(os.environ.get("BUDGET", 1e9))
    M = verify()
    T0 = pd.Timestamp(start)
    issued = pd.Timestamp.utcnow().tz_localize(None)
    live = pd.read_csv(PK / "etas_girdi_canli.csv")
    cut = live[pd.to_datetime(live.time) < T0]
    kat = PK / f"katalog_kesim_{start}.csv"
    if not kat.exists():
        cut.to_csv(kat, index=False)
    os.environ["QC_KATALOG"] = str(kat.relative_to(ROOT)); os.environ["QC_T_END"] = start
    os.environ["QC_PREP_TAG"] = "prospektif"
    sys.path.insert(0, str(ROOT / "src" / "etas"))
    import importlib
    import seq_update as SU; importlib.reload(SU)
    import seq_omori as SO; importlib.reload(SO)
    spec = json.load(open(ROOT / ENS))
    for cfg, _ in spec["ensemble"]:
        if time.time() - t_start > budget - 60:
            print("SURE DOLDU (hazırlık)"); return False
        SU.prepare_sparse(cfg)
        if time.time() - t_start > budget - 45:
            print("SURE DOLDU (hazırlık)"); return False
        SO.prepare_omori(cfg)
    import simulate_forecast as SF
    sims = [(SF.Simulator(cfg, seed=int(T0.strftime("%Y%m%d")) + q), w) for q, (cfg, w) in enumerate(spec["ensemble"])]
    for H in horizons:
        out = ROOT / f"prospektif/tahminler/{start}_{H}g"; out.mkdir(parents=True, exist_ok=True)
        parts = []; off = 0
        for sim, w in sims:
            n = int(round(n_sim * w))
            ev, info = sim.run(start, float(H), n)
            ev = ev[(ev.m >= 3.45) & ev.lat.between(BOX[0], BOX[1]) & ev.lon.between(BOX[2], BOX[3])].copy()
            ev["sim"] += off; off += n; parts.append(ev)
        ev = pd.concat(parts, ignore_index=True); ev["m"] = np.round(ev.m, 1)
        t0d = (T0 - SF.T_ORIGIN).total_seconds() / 86400
        np.savez_compressed(out / "sentetik_kataloglar.npz", sim=ev.sim.values.astype(np.int32),
                            dt_gun=(ev.t.values - t0d).astype(np.float32), enlem=ev.lat.values.astype(np.float32),
                            boylam=ev.lon.values.astype(np.float32), M=ev.m.values.astype(np.float32), n_sim=off)
        cnt = ev.groupby("sim").size().reindex(range(off), fill_value=0).values
        pd.DataFrame({"N": np.arange(cnt.max() + 1), "senaryo": np.bincount(cnt)}).to_csv(out / "sayi_dagilimi.csv", index=False)
        rows = []
        for name, (a0, a1, b0, b1) in SF.REGIONS.items():
            inr = ev.lat.between(a0, a1) & ev.lon.between(b0, b1)
            for thr in [3.5, 4.0, 5.0, 6.0]:
                c = ev[inr & (ev.m >= thr - 0.05)].groupby("sim").size().reindex(range(off), fill_value=0).values
                rows.append(dict(bolge=name, esik=f"M>={thr}", olasilik=round((c > 0).mean(), 4), beklenen=round(c.mean(), 3),
                                 q025=int(np.percentile(c, 2.5)), q975=int(np.percentile(c, 97.5))))
        pd.DataFrame(rows).to_csv(out / "bolge_olasiliklari.csv", index=False)
        ci = ((ev.lat - BOX[0]) // 0.1).astype(int) * 1000 + ((ev.lon - BOX[2]) // 0.1).astype(int)
        cell = pd.DataFrame({"hucre": ci, "m": ev.m})
        g = pd.DataFrame({"M35": cell.groupby("hucre").size() / off,
                          "M40": cell[cell.m >= 3.95].groupby("hucre").size() / off,
                          "M50": cell[cell.m >= 4.95].groupby("hucre").size() / off}).fillna(0).reset_index()
        g["enlem"] = BOX[0] + (g.hucre // 1000) * 0.1 + 0.05; g["boylam"] = BOX[2] + (g.hucre % 1000) * 0.1 + 0.05
        g.round(6).to_csv(out / "hucre_beklenen.csv.gz", index=False)
        try:
            commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            commit = ""
        meta = dict(model=M["model"], model_manifest_sha256=sha("prospektif/MODEL_V6_MANIFEST.json"), kod_commit=commit,
                    pencere_baslangic=str(T0), pencere_bitis=str(T0 + pd.Timedelta(days=H)), ufuk_gun=H,
                    yayin_zamani_utc=str(issued), gec_yayin=bool(issued >= T0),
                    veri_kesimi_son_olay=str(cut.time.iloc[-1]), katalog_sha256=hashlib.sha256(kat.read_bytes()).hexdigest(),
                    n_sim=off, uyeler=spec["ensemble"],
                    hedef="Mw>=3.5 (0.1 yuvarlama), 35-44K 25-46D, olay_turu=deprem",
                    medyan_N=float(np.median(cnt)), q025_N=float(np.percentile(cnt, 2.5)), q975_N=float(np.percentile(cnt, 97.5)))
        json.dump(meta, open(out / "meta.json", "w"), indent=1, ensure_ascii=False)
        print(f"{start} {H}g: medyan N={np.median(cnt):.0f} [{np.percentile(cnt, 2.5):.0f}, {np.percentile(cnt, 97.5):.0f}]", flush=True)
    return True


if __name__ == "__main__":
    a = sys.argv
    hz = tuple(int(x) for x in a[2].split(",")) if len(a) > 2 else (7, 30)
    ok = main(a[1], hz, int(a[3]) if len(a) > 3 else 5000)
    print("TAMAM" if ok else "DEVAM GEREKİYOR")
