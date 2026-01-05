#  BİR ALANIN OTONOM SÜRÜŞ İÇİN UYGUNLUK TESTİ ARACI GELİŞTİRMEK

> **Ders:** BOZ213 Nesne Yönelimli Programlama (OOP)
> **Proje Türü:** Final Projesi
> **Geliştirici:** Abdullah Şerefoğlu
> **Durum:** Tamamlandı (v1.0)

---

##  Proje Hakkında
Bu proje, otonom araçların seyir güvenliğini artırmak ve belirli bir güzergahın otonom sürüşe uygunluğunu (Suitability) test etmek amacıyla geliştirilmiş yapay zeka destekli bir simülasyon ve analiz yazılımıdır.

Yazılım, **YOLOv8** derin öğrenme modelini kullanarak trafik akışındaki dinamik nesneleri (araç, yaya, bisiklet vb.) gerçek zamanlı tespit eder. Tespit edilen nesnelerin vektörel hareketlerini analiz ederek potansiyel çarpışma risklerini, şerit ihlallerini ve takip mesafesi uyarılarını hesaplar. Analiz sonunda, güzergahın risk haritasını içeren detaylı bir **PDF Raporu** oluşturur.

---

##  Temel Özellikler

* **Gerçek Zamanlı Nesne Tespiti:** YOLOv8 Nano modeli ile yüksek performanslı araç ve yaya tespiti.
* **Dinamik Risk Analizi:**
    * **Çarpışma Riski (TTC):** Araca tehlikeli hızla yaklaşan nesnelerin tespiti.
    * **Yanal Hareket Analizi:** Önüne kırma (Cut-in) ve şerit ihlallerinin vektörel analizi.
* **Akıllı Bölge Kontrolü:** "Takip Mesafesi" ve "Çarpışma Bölgesi" ihlallerinin anlık izlenmesi.
* **Ortam Farkındalığı:** Gece/Gündüz modunu otomatik algılayarak risk katsayılarını dinamik olarak ayarlar.
* **Profesyonel Raporlama:**
    * Güzergah üzerine bindirilmiş (Overlay) **Isı Haritası (Heatmap)**.
    * Risk istatistiklerini içeren grafikler.
    * Otonom sürüş uygunluk skoru (%0 - %100).
    * QR Kod ile doğrulanabilir PDF çıktısı.

---

##  Kullanılan Teknolojiler ve Kütüphaneler

Proje, **Python 3.10+** kullanılarak geliştirilmiştir. Aşağıdaki temel kütüphanelerden yararlanılmıştır:

| Kütüphane | Kullanım Amacı |
| :--- | :--- |
| **Ultralytics (YOLOv8)** | Nesne tespiti ve takibi (Object Detection & Tracking). |
| **OpenCV (cv2)** | Görüntü işleme, çizim işlemleri ve ısı haritası oluşturma. |
| **PySide6 (Qt)** | Modern, responsive ve thread-safe kullanıcı arayüzü (GUI). |
| **ReportLab** | Vektörel tabanlı profesyonel PDF raporlarının oluşturulması. |
| **NumPy** | Vektörel hız ve mesafe hesaplamaları (Öklid, matris işlemleri). |
| **Matplotlib** | İstatistiksel verilerin grafiklere dökülmesi. |

---

# Test Verileri (Demo Videoları)
GitHub dosya boyutu sınırları (25MB) nedeniyle test videoları bu depoya eklenmemiştir. Projeyi test etmek için aşağıdaki telifsiz (royalty-free) örnek videoları bilgisayarınıza indirip kullanabilirsiniz:

1- Pexels. Dash cam footage in city driving. Video Dataset 2024. https://www.pexels.com/video/dash-cam-footage-in-city-driving-4644521/ 

2- Pexels. A video footage of moving cars in the city. Video Dataset 2024. https://www.pexels.com/video/a-video-footage-of-moving-cars-in-the-city-4644437/ 

3- Pexels. Person driving in a city street under a blue sky. Video Dataset 2024. https://www.pexels.com/video/person-driving-in-a-city-street-under-a-blue-sky-4483549//n

**Nasıl Test Edilir?**
1. Yukarıdaki linklerden bir videoyu indirin (önerim 1.videoyu indirmeniz.).
2. Uygulamayı çalıştırın (`py "BİR_ALANIN_OTONOM_SÜRÜŞ_İÇİN_UYGUNLUK_TESTİ_ARACI_GELİŞTİRMEK.py"`).
3. Arayüzden **"Dosya Aç"** butonuna basarak indirdiğiniz videoyu seçin.

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi bilgisayarınızda çalıştırmak için sırasıyla şu adımları izleyin:

### 1. Repoyu Klonlayın

Terminali açın ve projeyi indirin:

```bash
git clone https://github.com/abdullahserefoglu0-c/BOZ213-FINAL-Abdullah-Serefoglu-BIR-ALANIN-OTONOM-SURUS-ICIN-UYGUNLUK-TESTI-ARACI-GELISTIRMEK.git
cd BOZ213-FINAL-Abdullah-Serefoglu-BIR-ALANIN-OTONOM-SURUS-ICIN-UYGUNLUK-TESTI-ARACI-GELISTIRMEK
```
2. Sanal Ortam Oluşturun (Önerilen)

Kütüphanelerin çakışmaması için sanal ortam kurun:
```python -m venv venv

Windows için:
.\venv\Scripts\activate

Mac/Linux için:
source venv/bin/activate
```
   not: Eğer hata alırsanız "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process" bu yazıyı terminal'e yapıştırıp tekrar deneyin.
   
3. Gereksinimleri Yükleyin

Gerekli tüm kütüphaneleri (YOLO, OpenCV, PySide6 vb.) tek komutla yükleyin:
```
pip install ultralytics opencv-python PySide6 reportlab matplotlib qrcode[pil] Pillow numpy
```
4. Uygulamayı Başlatın

Kurulum bittikten sonra projeyi çalıştırın:
```
py "BİR_ALANIN_OTONOM_SÜRÜŞ_İÇİN_UYGUNLUK_TESTİ_ARACI_GELİŞTİRMEK.py"
```
Not: İlk çalıştırmada yolov8n.pt modeli otomatik olarak indirilecektir, internet bağlantısı gerektirir.

## 🏗️ Yazılım Mimarisi ve OOP Prensipleri

Bu proje, **"Temiz Kod" (Clean Code)** standartlarına ve **Nesne Yönelimli Programlama (OOP)** ilkelerine sadık kalınarak tasarlanmıştır.

### 1. Sınıflar ve Sorumluluklar
* **`MainWindow(QMainWindow)`:** Kullanıcı arayüzünü yönetir. Ayarların kapsüllenmesi (Encapsulation) ve kullanıcı etkileşimlerinin işlenmesinden sorumludur.
* **`VideoThread(QThread)`:** Görüntü işleme yükünü ana arayüzden ayırır (Multithreading). YOLO analizi, matematiksel hesaplamalar ve veri işleme bu sınıfta soyutlanmıştır (Abstraction).

### 2. Kullanılan OOP Prensipleri
* 🧬 **Kalıtım (Inheritance):** `VideoThread` sınıfı `QThread` sınıfından; `MainWindow` sınıfı `QMainWindow` sınıfından türetilmiştir.
* 🔄 **Çok Biçimlilik (Polymorphism):** `run()` ve `closeEvent()` gibi temel metotlar override edilerek projenin ihtiyaçlarına göre yeniden şekillendirilmiştir.
* 🔒 **Kapsülleme (Encapsulation):** Kritik veriler (`tracker_history`, `risk_weights`) sınıf içinde korunmuş, dışarıdan doğrudan müdahale engellenmiştir.

### 3. Veri Yapıları ve Algoritmalar
Performans optimizasyonu için aşağıdaki veri yapıları tercih edilmiştir:
* **Deque:** Nesne hareket geçmişi (Trajectory) için sabit boyutlu kuyruk yapısı kullanılarak bellek yönetimi sağlanmıştır.
* **Set (Küme):** Benzersiz nesne sayımı için `set` kullanılarak O(1) karmaşıklığında veri tekrarı önlenmiştir.
* **Algoritmalar:** Hız tahmini için *Öklid Mesafesi*, şerit ihlali tespiti için *Point-in-Polygon (Ray Casting)* kullanılmıştır.

---

## 📄 Lisans ve Telif Hakkı

Bu projenin tüm hakları saklıdır (All Rights Reserved).

Kaynak kodları sadece inceleme ve eğitim amaçlı erişime açıktır. İzin alınmadan ticari amaçla kullanılması, kopyalanması veya dağıtılması yasaktır.

**Copyright © 2026 Abdullah Şerefoğlu**

*Not: Bu proje, Ankara Üniversitesi BOZ213 dersi kapsamında geliştirilmiştir.*
