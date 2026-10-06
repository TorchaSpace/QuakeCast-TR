"""Yayın takvimi: 7 günlük pencereler her Çarşamba 00:00 UTC, 30 günlük pencereler her ayın 1'i 00:00 UTC başlar.
Günlük çalıştırmada (ör. 20:30 UTC) ertesi gün başlayan pencereler için tahmin yayınlanır (zaten varsa atlanır).
Elle: python3 src/prospective/schedule.py [YYYY-MM-DD]  (verilen gün başlayan pencereler)"""
import subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    day = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=1)).normalize()
    hz = []
    if day.weekday() == 2:
        hz.append(7)
    if day.day == 1:
        hz.append(30)
    hz = [h for h in hz if not (ROOT / f"prospektif/tahminler/{day.date()}_{h}g/meta.json").exists()]
    if not hz:
        print(f"{day.date()}: yayınlanacak pencere yok"); sys.exit(0)
    print(f"{day.date()}: yayın {hz}", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "src/prospective/issue_forecast.py"), str(day.date()), ",".join(map(str, hz))])
    sys.exit(r.returncode)
