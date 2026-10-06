"""Dondurulmuş v6 veri paketini depoya koyar / açar (GitHub Actions için; data/processed git dışında).
  python3 src/prospective/paket.py olustur   -> prospektif/model_v6_paket/<yol>.gz
  python3 src/prospective/paket.py ac        -> dosyaları yerine açar ve manifest SHA256'larını doğrular"""
import gzip, hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PAK = ROOT / "prospektif/model_v6_paket"
MAN = json.load(open(ROOT / "prospektif/MODEL_V6_MANIFEST.json"))
DATA = [p for p in MAN["dosyalar"] if p.startswith("data/")]

if __name__ == "__main__":
    if sys.argv[1] == "olustur":
        for p in DATA:
            q = PAK / (p + ".gz"); q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes(gzip.compress((ROOT / p).read_bytes(), mtime=0))
        print(f"{len(DATA)} dosya paketlendi")
    else:
        for p in DATA:
            b = gzip.decompress((PAK / (p + ".gz")).read_bytes())
            assert hashlib.sha256(b).hexdigest() == MAN["dosyalar"][p], p
            (ROOT / p).parent.mkdir(parents=True, exist_ok=True); (ROOT / p).write_bytes(b)
        print(f"{len(DATA)} dosya açıldı ve doğrulandı")
