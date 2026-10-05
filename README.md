# QuakeCast-TR

Türkiye için ETAS + fay-farkında yapay zekâ (PINN kısıtlı) hibrit, olasılıksal deprem tahmin sistemi.

**Soru:** "R bölgesinde önümüzdeki X gün içinde M ≥ Y deprem olma olasılığı nedir?"

## Veri
- **TURHEC** (1900–Ekim 2018, Mw*) — tarihsel omurga
- **AFAD Event Web Service** (2017 → bugün) — güncel katalog, 2017–2018 örtüşme ile Mw*'a kalibre
- Aktif fay veritabanı (TBD)

## Yapı
```
data/        # raw, interim, processed (git'e girmez)
src/         # kaynak kod
notebooks/   # keşif analizleri
configs/     # deney ayarları
```
