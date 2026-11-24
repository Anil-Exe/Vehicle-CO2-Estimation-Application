# Vehicle CO₂ Estimation

**Araç özelliklerinden CO₂ emisyonu tahmini · Streamlit & Machine Learning**

| Alan | Ayrıntı |
| --- | --- |
| Amaç | Araç özelliklerinden g/km cinsinden CO₂ emisyonu tahmini |
| Modeller | LightGBM, K-En Yakın Komşu (KNN), yapay sinir ağı |
| Arayüz | Streamlit |
| Veri | Canada vehicle CO₂ emissions CSV |
| Geliştirici | Anıl Polat |

## Proje bağlamı

Üniversite eğitimi sırasında geliştirilen bir makine öğrenmesi projesidir. Veri ön işleme, regresyon modellerinin karşılaştırılması ve tahminlerin etkileşimli bir arayüzde sunulması üzerine yapılan uygulamalı çalışmaları içerir.

## Genel bakış

Motor hacmi, silindir sayısı, yakıt tüketimi, marka, model ve diğer araç bilgilerini kullanarak üç regresyon modeli arasında karşılaştırmalı tahmin sunan bir eğitim projesidir.

- Model seçimi ve araç özellikleri için etkileşimli girişler
- Eğitimde hiperparametre araması ve erken durdurma
- R², MSE ve MAE hesaplamaları ile karşılaştırma grafikleri
- Eğitimden saklanan özellik sırası ve sayısal ölçekleyici

## Yerel kurulum

```sh
python -m venv .venv
# Oluşturduğunuz sanal ortamı işletim sisteminize uygun biçimde etkinleştirin.
python -m pip install -r requirements.txt
streamlit run app.py
```

Komutları depo kökünde çalıştırın. `app.py`, CSV dosyasını, üç model dosyasını, `scaler.pkl` ve `model_features.pkl` dosyalarını bekler. Eski modellerin oluşturulduğu paket sürümleri kaydedilmemiştir; requirements.txt sürüm kilidi değildir ve model uyumluluğu ayrıca doğrulanmalıdır.

## Dosya rehberi

| Dosya | İşlev |
| --- | --- |
| `app.py` | Streamlit tahmin arayüzü |
| `preprocessing.py` | Tek tahmin satırını eğitim özelliklerine hizalama |
| `makine proje 2.py` | Modellerin eğitimi, değerlendirmesi ve kaydı |
| `test_preprocessing.py` | Kategori seçiminin kaybolmasına karşı regresyon testleri |

Eğitim betiği `python "makine proje 2.py"` ile çalıştırılabilir; yeniden eğitim mevcut model dosyalarını değiştirebilir ve uzun sürebilir.

## Doğrulama ve sınırlamalar

```sh
python -m unittest test_preprocessing.py
```

Tek satırlık tahminde `drop_first=True` kullanılması seçilen kategori bilgisini siliyordu. Kodlama artık kategoriyi korur ve eğitimde saklanan sütunlara hizalar. Eğitim dosyasındaki bilgisayara özgü mutlak veri yolu da kaldırılmıştır.

Bu düzenleme kapsamında model dosyaları yüklenmedi ve modeller yeniden eğitilmedi. Mevcut eğitim betiği ölçekleyiciyi eğitim/test ayrımından önce öğrenir; veri sızıntısını önleyen bir Pipeline ve ayrı değerlendirme düzeni kurulmadan tarihsel sonuçlar genelleme başarısının kanıtı olarak sunulmamalıdır. Uygulama uçtan uca çalıştırılmamıştır.

## Veri ve model güvenliği

Yalnız kaynağını bildiğiniz kendi model dosyalarınızı kullanın. Pickle/joblib dosyaları yüklenirken kod çalıştırabilir; dışarıdan gelen model dosyalarını doğrulamadan açmayın. CSV'nin özgün indirme bağlantısı proje arşivinde kayıtlı değildir; veri kaynağına ilişkin atıf bilgisinin tamamlanması geliştirme notları arasındadır. Gerçek kullanıcı verileri, servis anahtarları ve ortam dosyaları depoya eklenmemelidir.
