# QuakeCast-TR prospektif test protokolü (v6)

Amaç: dondurulmuş bir modelin tahminlerini **pencere başlamadan önce**, zaman damgalı ve herkese açık biçimde yayınlamak. Pencere bittikten sonra bu tahminleri önceden belirlenmiş testlerle puanlamak. Sözde-prospektif (geriye dönük) testlerin tersine burada model, test verisini hiçbir biçimde görmez.

## 1. Dondurulmuş model
- **Model:** QuakeCast-TR v6 topluluk = 0.8 × `configs/v6_dizi_omori.json` + 0.2 × `configs/v6_sabitMc_dizi_omori.json`.
- **Dondurma:** 6 Ekim 2026, 08:26 UTC. `prospektif/MODEL_V6_MANIFEST.json` tüm model parametre dosyalarının, eğitim kataloglarının, dondurulmuş geçmiş kataloğun, eşleştirme ve Mw dönüşüm katsayılarının ve tahmin kodunun SHA256 değerlerini içerir (33 dosya).
- **etas kütüphanesi:** lmizrahi/etas@097f08b.
- **Doğrulama:** Yayın betiği her çalıştığında manifesti doğrular. Tek bir bayt değişse bile yayın durur.
- **Yeni sürümler:** Modelde değişiklik yasaktır. Yeni bir model (v7 …) ayrı manifestle **paralel** kaydedilir. v6 kaydı kesilmez.

## 2. Takvim
- **7 günlük pencereler:** her Çarşamba 00:00 UTC başlar.
- **30 günlük pencereler:** her ayın 1'i 00:00 UTC başlar.
- **Yayın:** GitHub Actions her gün 20:30 UTC'de ertesi gün başlayan pencereleri yayınlar ve commit eder. Commit zamanı yayın kanıtıdır.
- **Geç yayın:** pencere başladıktan sonra yayınlanan tahmin `gec_yayin = true` ile işaretlenir ve ana istatistiklere girmez.
- **Kaçırılan pencereler:** geriye dönük doldurulmaz.
- **Pilot:** 2026-10-07 başlangıçlı 7 günlük pencere (takvime uygun) ve 30 günlük pencere (takvim dışı pilot). Her ikisi 6 Ekim 2026 08:28 UTC'de yayınlandı.

## 3. Veri
- **Canlı katalog** (`src/prospective/live_catalog.py`):
  - 2026-08-01 öncesi: dondurulmuş araştırma kataloğu.
  - Sonrası: AFAD (API ve web olay türleri) ile Kandilli yeniden indirilir. Eşleştirme dondurulmuş kalibrasyonla, Mw dondurulmuş GOR katsayılarıyla hesaplanır. Bu kurulum araştırma kataloğunu birebir yeniden üretiyor.
- **Kandilli gecikmesi:** Kandilli kataloğu ~2 ay gecikmeli yayınlanıyor. Bu yüzden tahminlerin geçmişi son ~2 ayda yalnız AFAD'dır.
- **İleriye bakış yok:** Tahmin yalnız pencere başlangıcından önceki olayları kullanır. Kesim zamanı ve katalog SHA256'sı her tahminin `meta.json` dosyasında.

## 4. Hedef olaylar
- Mw ≥ 3.5 (0.1'e yuvarlanmış).
- Alan: 35–44°K, 25–46°D.
- Olay türü: deprem. Derinlik sınırı yok.

## 5. Çıktılar (`prospektif/tahminler/<başlangıç>_<H>g/`)
- `meta.json`: yayın zamanı, veri kesimi, model ve katalog hash'leri, sayı dağılımı özeti.
- `bolge_olasiliklari.csv`: 9 bölge × M≥3.5/4/5/6 için en az bir olay olasılığı, beklenen sayı ve %95 aralığı.
- `hucre_beklenen.csv.gz`: 0.1° hücrelerde beklenen sayı (M≥3.5, 4.0, 5.0).
- `sayi_dagilimi.csv`: toplam sayının simülasyon dağılımı.
- `sentetik_kataloglar.npz`: 5000 sentetik katalog (katalog-tabanlı CSEP testleri için).

## 6. Değerlendirme (`src/prospective/evaluate_forecasts.py` → `prospektif/DEFTER.csv`)
**Aşamalar:**
- **on:** pencere bittikten 3 gün sonra, o anki canlı katalogla (son dönem yalnız AFAD).
- **kesin:** Kandilli pencere sonunu kapsadığında, birleşik katalogla.

**Testler (pyCSEP katalog-tabanlı tanımlar):**
- N-testi (δ1, δ2; %95 ve %99).
- M-, S- ve PL-testleri (γ < 0.05 başarısız).
- Zamandan bağımsız referansa göre olay başına bilgi kazancı. Referans: 1900–2013 ayıklanmış yumuşatılmış harita × 2014'ten pencere başına kadarki ortalama oran. 0.1° hücre Poisson olabilirliğiyle hesaplanır.

**Özet ölçütler:**
- N-testi başarısızlık oranı. Literatür karşılaştırması: OEF-İtalya 7 günlük pencerelerde %1.8 (%99 düzeyinde).
- Ortalama bilgi kazancı.
- Büyük olaylarda (M≥5) olasılık kalibrasyonu (güvenilirlik diyagramı; yeterli sayıda pencere birikince).

## 7. Bilinen sınırlamalar
- **Tamlık maskesi:** Değerlendirmede tamlık maskesi uygulanmaz. Büyük bir depremden sonraki ilk saatlerde katalogdaki eksik olaylar fazla tahmin gibi görünebilir. Bu durum, araştırma testlerindeki Mc(t,x) maskeli puanlamadan farklıdır.
- **Kandilli gecikmesi:** tahmin geçmişinde ve "on" aşamasında küçük bir tutarsızlık yaratır. "Kesin" aşama bunu giderir.
- **GitHub Actions erişimi:** Sunucularının AFAD/Kandilli'ye erişimi ilk çalıştırmada görülecek. Yedek: `bash prospektif/yerel_calistir.sh`.

## 8. Değişiklik günlüğü
- 2026-10-06: Protokol, v6 dondurma, pilot yayınlar.
