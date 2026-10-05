"""Topluluk (ensemble) tahmini: birden çok ETAS modelinin (farklı eğitim pencereleri) simülasyonları birleştirilir.

Her model eşit ağırlıkla N_SIM/k senaryo üretir; bölge x eşik için P(en az 1 olay) ve beklenen sayı, ayrıca modeller
arası en düşük-en yüksek olasılık (parametre belirsizliği) raporlanır.
Kullanım: python3 src/etas/ensemble_forecast.py <T0 YYYY-MM-DD> <H_gun> <N_SIM_model_basina> cfg1.json cfg2.json ...
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "etas"))
import simulate_forecast as SF  # noqa


def main(T0, H, N, cfgs):
    per = []; allc = {}
    for i, cp in enumerate(cfgs):
        S = SF.Simulator(cp, seed=100 + i)
        ev, info = S.run(T0, H, N)
        for name, (a0, a1, b0, b1) in SF.REGIONS.items():
            inr = ev.lat.between(a0, a1) & ev.lon.between(b0, b1)
            for thr in SF.THRESH:
                cnt = ev[inr & (ev.m >= thr)].groupby("sim").size().reindex(range(N), fill_value=0).values
                allc.setdefault((name, thr), []).append(cnt)
                per.append(dict(model=Path(cp).stem, bolge=name, esik=thr, olasilik=(cnt > 0).mean(), beklenen=cnt.mean()))
        print(cp, info, flush=True)
    per = pd.DataFrame(per)
    rows = []
    for (name, thr), lst in allc.items():
        c = np.concatenate(lst); pm = per[(per.bolge == name) & (per.esik == thr)]
        rows.append(dict(bolge=name, esik=f"M>={thr:.0f}", olasilik=round((c > 0).mean(), 3),
                         olasilik_model_araligi=f"{pm.olasilik.min():.3f}-{pm.olasilik.max():.3f}",
                         beklenen=round(c.mean(), 2), q05=int(np.percentile(c, 5)), q95=int(np.percentile(c, 95))))
    tab = pd.DataFrame(rows)
    out = ROOT / "data/processed/tahmin" / f"topluluk_{T0}_{int(H)}g"; out.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out / "bolge_olasiliklari.txt", sep="\t", index=False); per.to_csv(out / "model_bazinda.txt", sep="\t", index=False)
    L = [f"QuakeCast-TR topluluk tahmini — başlangıç {T0}, ufuk {int(H)} gün, {len(cfgs)} model x {N} senaryo",
         "Modeller: " + ", ".join(Path(c).stem for c in cfgs), "",
         tab.to_string(index=False), "",
         "olasilik: en az bir olay olasılığı (tüm senaryolar); olasilik_model_araligi: modeller arası en düşük-en yüksek (parametre belirsizliği);",
         "q05-q95: olay sayısının %90 aralığı. Bölgeler yaklaşık enlem-boylam kutularıdır.",
         "Kalibrasyon notu: 2022-2026 testlerinde model M>=3.5-4 sayılarını sistematik olarak fazla tahmin etti (bkz. kalibrasyon_tau1y.txt);",
         "M>=5 için kalibrasyon kabul edilebilir düzeyde. Olasılıklar bu bilinen yanlılıkla birlikte yorumlanmalıdır."]
    (out / "ozet.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]), int(a[3]), a[4:])
