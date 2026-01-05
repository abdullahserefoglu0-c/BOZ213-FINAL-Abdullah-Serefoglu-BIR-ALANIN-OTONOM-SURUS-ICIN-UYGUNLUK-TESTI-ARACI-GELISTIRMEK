"""
PROJE: BİR ALANIN OTONOM SÜRÜŞ İÇİN UYGUNLUK TESTİ ARACI GELİŞTİRMEK
DERS: Nesne Yönelimli Programlama (OOP)
YAZAR: ABDULLAH ŞEREFOĞLU
TARİH: 01.01.2026

AÇIKLAMA:
Bu modül, YOLOv8 ve OpenCV kullanarak otonom araçlar için çevresel farkındalık
ve risk analizi sağlar. Şerit takibi, nesne tespiti, hız tahmini ve çarpışma
riski hesaplamaları yaparak sonuçları GUI üzerinde ve PDF raporu olarak sunar.
"""

import sys
import cv2
import numpy as np
import tempfile
import shutil
import os
import time
import threading
from datetime import datetime
from collections import deque 
from typing import Tuple, List, Dict, Optional, Union # Type Hinting için

# Harici Kütüphaneler
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt

# Ses Kütüphanesi (Opsiyonel)
try:
    import winsound
except ImportError:
    winsound = None

# GUI Kütüphaneleri (PySide6)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QFileDialog, QMessageBox, QHBoxLayout, QTextEdit, QStatusBar,
    QProgressBar, QLineEdit, QFormLayout, QGroupBox
)
from PySide6.QtGui import QPixmap, QImage, QDoubleValidator
from PySide6.QtCore import QThread, Signal, Qt, Slot 

# =============================================================================
# 1. AYARLAR VE SABİTLER (CONFIGURATION)
# =============================================================================
MODEL_NAME = 'yolov8n.pt' # YOLO Nano modeli (Hız optimizasyonu için)

# Bölge Ayarları (Ekran oranlarına göre dikey çizgiler)
WARNING_ZONE_PERCENTAGE = 0.75  # Sarı Çizgi: Takip Mesafesi (Ekranın %75'i)
DANGER_ZONE_PERCENTAGE = 0.92   # Kırmızı Çizgi: Tampon/Çarpışma Mesafesi (Ekranın %92'si)

# Performans ve Analiz Ayarları
FRAME_SKIP_RATE = 2            # Her 2 karede bir yapay zeka analizi yap (FPS artırmak için)
PRUNE_INTERVAL_SECONDS = 10    # 10 saniyeden eski nesne geçmişini sil (RAM optimizasyonu)
NIGHT_BRIGHTNESS_THRESHOLD = 75 # Parlaklık 75'in altındaysa Gece Modu'nu aç
AUDIO_COOLDOWN = 2.0           # Sesli uyarılar arasında en az 2 saniye bekle

# Yol Alanı Tanımları (Poligon Koordinatları - Normalize Edilmiş 0.0-1.0)
# Bu alan otonom aracın kendi şeridini temsil eder.
ROAD_POLYGON_POINTS = [              
    (0.46, 0.77), # Sol Üst
    (0.50, 0.77), # Sağ Üst
    (0.65, 1.00), # Sağ Alt
    (0.30, 1.00)  # Sol Alt
]

# Risk Haritası: Hangi nesne ne kadar tehlikeli?
RISK_MAP = {
    'person': 'YÜKSEK', 'bicycle': 'YÜKSEK', 'motorcycle': 'YÜKSEK',
    'dog': 'YÜKSEK', 'cat': 'YÜKSEK', 'horse': 'YÜKSEK',
    'stop sign': 'YÜKSEK', 
    'car': 'ORTA', 'truck': 'ORTA', 'bus': 'ORTA', 'train': 'ORTA',
    'traffic light': 'ORTA', 
    'bench': 'DÜŞÜK', 'fire hydrant': 'DÜŞÜK', 'backpack': 'DÜŞÜK'
}

# Görselleştirme Renkleri (BGR Formatı)
COLOR_MAP = {
    'ACİL DURUM': (0, 0, 255),    # Kırmızı
    'KRİTİK': (0, 100, 255),      # Turuncu
    'YÜKSEK': (0, 200, 255),      # Sarı
    'ORTA': (255, 255, 0),        # Mavi
    'DÜŞÜK': (0, 255, 0)          # Yeşil
}
DEFAULT_COLOR = (255, 0, 255)

# =============================================================================
# 2. YARDIMCI FONKSİYONLAR (HELPER FUNCTIONS)
# =============================================================================

def play_alert_sound():
    """
    Sistemde acil durum algılandığında sesli uyarı verir.
    Windows sistemler için winsound kullanır.
    """
    if winsound:
        winsound.Beep(1000, 400) # 1000 Hz frekans, 400 ms süre

# =============================================================================
# 3. VİDEO İŞLEME VE ANALİZ SINIFI (WORKER THREAD)
# =============================================================================

class VideoThread(QThread):
    """
    Video işleme işlemlerini arka planda yürüten Thread sınıfı.
    GUI'nin donmasını engeller ve görüntü işleme algoritmalarını kapsüller.
    """
    # GUI ile iletişim sinyalleri
    changePixmap = Signal(QImage)           # İşlenmiş kareyi GUI'ye gönderir
    updateLiveCounts = Signal(dict)         # Anlık risk sayılarını gönderir
    updateCumulativeSummary = Signal(dict)  # Toplam istatistikleri gönderir
    analysisComplete = Signal(str, str, float) # Analiz bittiğinde rapor verisini gönderir

    def __init__(self, video_path: str, location_text: str, weights: dict):
        """
        Thread başlatıcı.
        :param video_path: Analiz edilecek videonun dosya yolu
        :param location_text: Rapor için konum bilgisi
        :param weights: Risk hesaplama ağırlık katsayıları
        """
        super().__init__()
        self.video_path = video_path
        self._is_running = True
        self.location_text = location_text
        self.weights = weights 

        # İstatistik Tutucular
        self.total_counts = {'ACİL DURUM': 0, 'KRİTİK': 0, 'YÜKSEK': 0, 'ORTA': 0, 'DÜŞÜK': 0}
        self.unique_ids_seen = {'ACİL DURUM': set(), 'KRİTİK': set(), 'YÜKSEK': set(), 'ORTA': set(), 'DÜŞÜK': set()}

        # Grafik Verileri
        self.risk_over_time = []
        self.frame_timestamps = []

        # Nesne Takip Geçmişi (Trajectory Analysis için)
        self.tracker_history = {} 
        self.TRAJ_HISTORY_LEN = 15 # Son 15 kareyi hatırla
        self.TRAJ_MIN_LEN = 5      # Hız hesabı için en az 5 kare gerekli
        
        # Hareket Hassasiyet Ayarları
        self.TRAJ_YAW_THRESHOLD_PX = 25  # Yanal kayma (Önüne kırma) eşiği
        self.TRAJ_FWD_THRESHOLD_PX = 5   # Yaklaşma eşiği
        self.SPEED_THRESHOLD_PX = 15.0   # Hız eşiği
        
        # Çevresel Değişkenler
        self.is_night_time = False 
        self.heatmap_accumulator = None # Isı haritası verisi
        self.last_audio_time = 0
        
        # Raporlama Değişkenleri
        self.highest_risk_frame = None  
        self.max_risk_score_so_far = -1 
        self.last_frame_rgb = None      

    def run(self):
        """
        Thread'in ana döngüsü. Videoyu kare kare okur ve işler.
        Deterministik zamanlama (FPS bazlı) kullanır.
        """
        try:
            model = YOLO(MODEL_NAME) # Yapay zeka modelini yükle
        except Exception as e:
            print(f"HATA: Model yüklenemedi: {e}") 
            return

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("HATA: Video dosyası açılamadı.")
            return

        # Video özelliklerini al
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        self.heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

        # --- ORTAM ANALİZİ (GECE/GÜNDÜZ TESPİTİ) ---
        # İlk 5 kareye bakarak ortam parlaklığını ölçer.
        brightness_samples = []
        for _ in range(5): 
            ret_b, frame_b = cap.read()
            if not ret_b: break
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
            brightness_samples.append(gray_b.mean())
        if brightness_samples:
            avg_brightness = np.mean(brightness_samples)
            if avg_brightness < NIGHT_BRIGHTNESS_THRESHOLD:
                self.is_night_time = True # Gece modu aktif
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Videoyu başa sar
        
        self.tracker_history.clear()
        
        frame_count = 0
        last_counts = {} 
        frame_to_display = None 

        # --- ANA ANALİZ DÖNGÜSÜ ---
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                break # Video bitti
            
            frame_count += 1
            
            # Deterministik Zaman Hesabı (Donanımdan bağımsız tutarlı analiz için)
            current_video_time = frame_count / fps
            
            # Rapor görseli için arka plan karesini sakla
            if frame_count % (FRAME_SKIP_RATE * 5) == 0: 
                 self.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame_to_display is None:
                frame_to_display = frame.copy()

            # Performans optimizasyonu: Her kareyi değil, belirlenen aralıklarla işle
            if frame_count % FRAME_SKIP_RATE == 0:
                
                # Görüntü İşleme ve AI Analizi Fonksiyonunu Çağır
                annotated_frame, per_counts, frame_risk = self.analyze_draw_and_track(
                    frame, model, self.weights, FPS=fps, current_time=current_video_time
                )
                
                # En yüksek riskli anı rapor için yakala
                if frame_risk > self.max_risk_score_so_far:
                    self.max_risk_score_so_far = frame_risk
                    self.highest_risk_frame = annotated_frame.copy()

                frame_to_display = annotated_frame 
                last_counts = per_counts 
                
                # İstatistikleri güncelle
                for k in self.total_counts:
                    self.total_counts[k] += per_counts.get(k, 0)
                
                live_unique_counts = {k: len(v) for k, v in self.unique_ids_seen.items()}
                self.updateCumulativeSummary.emit(live_unique_counts)
                
                self.risk_over_time.append(frame_risk)
                self.frame_timestamps.append(current_video_time)
            
            # Görüntüyü GUI formatına (QImage) çevir ve gönder
            rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.changePixmap.emit(qimg.copy())

            self.updateLiveCounts.emit(last_counts)
            
            # Bellek Yönetimi: Eski nesne geçmişlerini temizle
            if frame_count % int(fps) == 0:
                ids_to_delete = [
                    tid for tid, data in self.tracker_history.items() 
                    if current_video_time - data['last_seen_frame'] > PRUNE_INTERVAL_SECONDS
                ]
                for tid in ids_to_delete:
                    if tid in self.tracker_history:
                        del self.tracker_history[tid]

        cap.release()
        
        if not self._is_running: 
            return

        # --- ANALİZ SONRASI İŞLEMLER (HEATMAP & PDF) ---
        heatmap_path = os.path.join(tempfile.gettempdir(), "final_heatmap.png")
        try:
            # Isı haritasını normalize et ve renklendir
            heatmap_norm = cv2.normalize(self.heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_norm = heatmap_norm.astype(np.uint8)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            
            # Isı haritasını gerçek görüntü üzerine bindir (Overlay)
            if self.last_frame_rgb is not None:
                background = cv2.cvtColor(self.last_frame_rgb, cv2.COLOR_RGB2BGR)
                # Boyut eşitleme
                if background.shape[:2] != heatmap_color.shape[:2]:
                    background = cv2.resize(background, (heatmap_color.shape[1], heatmap_color.shape[0]))
                
                overlay_img = cv2.addWeighted(background, 0.6, heatmap_color, 0.4, 0)
                cv2.imwrite(heatmap_path, overlay_img)
            else:
                cv2.imwrite(heatmap_path, heatmap_color)
        except Exception as e:
            print(f"Heatmap oluşturma hatası: {e}")
            heatmap_path = None

        try:
            # PDF Raporunu Oluştur
            pdf_path, report_text, score = self.generate_pdf_report(0, self.weights, heatmap_path)
            self.analysisComplete.emit(pdf_path, report_text, score)
        except Exception as e:
            print(f"Rapor oluşturulamadı: {e}", exc_info=True)

    def _update_tracker_history(self, track_id: int, box, current_time: float) -> list:
        """
        Nesnenin konum geçmişini günceller.
        :param track_id: Nesnenin YOLO ID'si
        :param box: Bounding box verisi
        :param current_time: O anki video zamanı
        :return: Nesnenin son koordinat listesi (coords)
        """
        cx = (box.xyxy[0][0] + box.xyxy[0][2]) / 2
        cy = box.xyxy[0][3] # Nesnenin yere bastığı nokta (Foot point)
        
        if track_id not in self.tracker_history:
            self.tracker_history[track_id] = {
                'coords': deque(maxlen=self.TRAJ_HISTORY_LEN), 
                'last_seen_frame': 0
            }
        self.tracker_history[track_id]['coords'].append((cx, cy))
        self.tracker_history[track_id]['last_seen_frame'] = current_time
        return self.tracker_history[track_id]['coords']

    def _analyze_trajectory(self, coords: list) -> Tuple[bool, bool, str, float]:
        """
        Nesnenin hareket vektörünü analiz eder (Hız ve Yön).
        :return: (Yanal Hareket Var mı, Yaklaşıyor mu, Etiket Ekleri, Hız Px)
        """
        is_moving_laterally = False 
        is_moving_forward = False   
        label_addition = ""
        speed_px = 0.0 
        
        if len(coords) < self.TRAJ_MIN_LEN: 
            return is_moving_laterally, is_moving_forward, label_addition, speed_px 
            
        # Hareket hesabı (Son nokta - Başlangıç noktası)
        start_point = coords[-5] if len(coords) >= 5 else coords[0]
        end_point = coords[-1]     
        
        # Öklid mesafesi ile hız tahmini
        dist = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        speed_px = dist / 5.0 
        
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        
        # Dikey hareket (Yaklaşma)
        if delta_y > self.TRAJ_FWD_THRESHOLD_PX:
            is_moving_forward = True
            label_addition += " (YAKLAŞIYOR)"
            
        # Yatay hareket (Şerit ihlali / Önüne kırma)
        if abs(delta_x) > self.TRAJ_YAW_THRESHOLD_PX:
            is_moving_laterally = True
            label_addition += " (KAYMA)"
            
        return is_moving_laterally, is_moving_forward, label_addition, speed_px

    def _determine_risk_level(self, base_risk, is_in_warning_zone, is_in_danger_zone, is_moving_laterally, is_moving_forward, speed_px):
        """
        Nesnenin konumuna ve hareketine göre nihai risk seviyesini belirler.
        Karar Ağacı Mantığı kullanılır.
        """
        risk = base_risk
        risk_label_suffix = ""
        is_fast = speed_px > self.SPEED_THRESHOLD_PX
        
        if risk == 'ORTA': # Genelde Araçlar
            if is_in_danger_zone:
                if is_fast: 
                    risk = 'ACİL DURUM'; risk_label_suffix = " (ÇARPIŞMA RİSKİ!)"
                else:
                    risk = 'KRİTİK'; risk_label_suffix = " (ÇOK YAKIN TAKİP)"
            elif is_in_warning_zone:
                if is_moving_laterally:
                    risk = 'KRİTİK'; risk_label_suffix = " (ÖNÜNE KIRMA)"
                else:
                    risk = 'ORTA'
                    if is_moving_forward: risk_label_suffix = " (YAKLAŞIYOR)" 
                    else: risk_label_suffix = " (TAKİP)"
                    
        elif risk == 'YÜKSEK': # Yayalar ve Savunmasızlar
            if is_in_warning_zone:
                risk = 'KRİTİK' 
                if is_fast: risk_label_suffix += " (HIZLI!)"
            if risk == 'KRİTİK':
                if is_moving_forward or is_in_danger_zone or (is_in_warning_zone and is_moving_laterally):
                    risk = 'ACİL DURUM'
                    if is_moving_forward: risk_label_suffix = " (ACİL: YAKLAŞIYOR!)"
                    else: risk_label_suffix = " (ACİL: YOLA GİRİYOR!)"
                elif is_fast and is_in_warning_zone:
                     risk = 'ACİL DURUM'; risk_label_suffix = " (ACİL: YÜKSEK HIZ!)"
                     
        return risk, risk_label_suffix

    def analyze_draw_and_track(self, frame, model, weights, FPS=30, current_time=0.0):
        """
        Görüntü üzerindeki ana analiz pipeline'ı.
        1. Bölgeleri çizer.
        2. YOLO ile nesneleri bulur.
        3. Risk hesaplar ve ekrana yazar.
        """
        counts = {'ACİL DURUM': 0, 'KRİTİK': 0, 'YÜKSEK': 0, 'ORTA': 0, 'DÜŞÜK': 0}
        frame_height, frame_width, _ = frame.shape
        
        # Bölge Çizgileri
        warning_line_y = int(frame_height * WARNING_ZONE_PERCENTAGE) 
        danger_line_y = int(frame_height * DANGER_ZONE_PERCENTAGE)   
        
        # Yol Poligonunu Çiz (Şerit Alanı)
        poly_points = []
        for pt in ROAD_POLYGON_POINTS:
            poly_points.append((int(pt[0] * frame_width), int(pt[1] * frame_height)))
        road_poly_np = np.array([poly_points], np.int32)
        
        cv2.polylines(frame, [road_poly_np], isClosed=True, color=(0, 255, 100), thickness=2)
        
        # Görselleştirme: Yarı saydam yeşil yol alanı
        overlay = frame.copy()
        cv2.fillPoly(overlay, [road_poly_np], (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, frame, 0.9, 0, frame) 
        
        # Uyarı Çizgileri ve Metinler
        cv2.line(frame, (0, warning_line_y), (frame_width, warning_line_y), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "TAKIP MESAFESI", (10, warning_line_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.line(frame, (0, danger_line_y), (frame_width, danger_line_y), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "CARPISMA BOLGESI", (10, danger_line_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # --- YOLO TESPİTİ ---
        # conf=0.50 ile güven eşiği filtresi uygulanıyor (Yanlış pozitifleri azaltır)
        results = model.track(frame, persist=True, conf=0.50, verbose=False)
        
        frame_risk_score = 0
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                try:
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    
                    # Koordinatları al
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2) # Merkez nokta
                    foot_point_y = y2  # Nesnenin yere bastığı nokta (Risk hesabı için kritik)
                    
                    track_id = -1
                    if box.id is not None: track_id = int(box.id[0])
                    
                    # Hareket Analizi
                    coords = self._update_tracker_history(track_id, box, current_time)
                    is_lat, is_fwd, traj_label, speed_px = self._analyze_trajectory(coords)
                    
                    # Risk Belirleme
                    base_risk = RISK_MAP.get(class_name, 'DÜŞÜK') 
                    
                    # Nesne bizim şeridimizde mi? (Point in Polygon Test)
                    is_on_road = cv2.pointPolygonTest(road_poly_np, (float(cx), float(foot_point_y)), False) >= 0
                    is_in_warning = foot_point_y > warning_line_y
                    is_in_danger = foot_point_y > danger_line_y
                    
                    final_risk = base_risk
                    risk_label_suffix = ""
                    
                    if is_on_road:
                        final_risk, suffix = self._determine_risk_level(base_risk, is_in_warning, is_in_danger, is_lat, is_fwd, speed_px)
                        risk_label_suffix = suffix
                    else:
                        # Yol dışındaysa riski düşür
                        if base_risk in ['YÜKSEK', 'KRİTİK', 'ACİL DURUM']:
                            if is_lat: final_risk = 'ORTA'; risk_label_suffix = " (YOLA YÖNELİM!)"
                            else: final_risk = 'DÜŞÜK'; risk_label_suffix = " (KALDIRIM)"
                        else: final_risk = base_risk 
                    
                    color = COLOR_MAP.get(final_risk, DEFAULT_COLOR)
                    
                    # Isı Haritası (Heatmap) Güncelleme
                    if final_risk in ['YÜKSEK', 'KRİTİK', 'ACİL DURUM']:
                        cv2.circle(self.heatmap_accumulator, (cx, cy), 20, (1.0), -1)

                    counts[final_risk] += 1 
                    if track_id != -1: self.unique_ids_seen[final_risk].add(track_id)
                    
                    # Ekrana Çizim
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"[ID:{track_id}] {class_name} ({final_risk}){risk_label_suffix}{traj_label}"
                    if speed_px > 5: label += f" S:{int(speed_px)}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    
                except Exception as e: print(f"Çizim hatası: {e}")
        
        # Sesli Uyarı Kontrolü
        if counts['ACİL DURUM'] > 0:
            now = time.time()
            if now - self.last_audio_time > AUDIO_COOLDOWN:
                threading.Thread(target=play_alert_sound, daemon=True).start()
                self.last_audio_time = now
        
        # Ağırlıklı Risk Puanı Hesabı
        frame_risk_score += (counts.get('ACİL DURUM', 0) * weights['acil'])
        frame_risk_score += (counts.get('KRİTİK', 0) * weights['kritik'])
        frame_risk_score += (counts.get('YÜKSEK', 0) * weights['yuksek'])
        
        return frame, counts, frame_risk_score

    # ----------------------------- RAPORLAMA MODÜLÜ -----------------------------
    def generate_pdf_report(self, start_time_unused, weights, heatmap_path=None):
        """
        Analiz sonuçlarını içeren profesyonel bir PDF raporu oluşturur.
        ReportLab kütüphanesini kullanır.
        """
        pdf_path = os.path.join(os.getcwd(), f"alan_uygunluk_raporu_{int(time.time())}.pdf")
        
        # Font Ayarları (Türkçe karakter desteği için Arial)
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        try:
            pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
            pdfmetrics.registerFont(TTFont('Arial-Bold', 'arialbd.ttf'))
            font_normal = 'Arial'
            font_bold = 'Arial-Bold'
        except:
            font_normal = 'Helvetica'
            font_bold = 'Helvetica-Bold'
            print("UYARI: Arial fontu bulunamadı, varsayılan font kullanılıyor.")
        
        # --- İstatistikler ve Grafikler ---
        labels = ['ACİL DURUM', 'KRİTİK', 'YÜKSEK', 'ORTA', 'DÜŞÜK']
        unique_sizes = [len(self.unique_ids_seen[l]) for l in labels]
        total_unique_objects = sum(unique_sizes)
        
        # Grafik 1: Bar Chart
        bar_path = os.path.join(tempfile.gettempdir(), "risk_bar.png")
        plt.figure(figsize=(6, 3))
        colors_plt = ['red', 'orange', 'gold', 'skyblue', 'lightgreen']
        bars = plt.bar(labels, unique_sizes, color=colors_plt)
        plt.title('Nesne Dağılımı ve Risk Sınıfları')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.bar_label(bars)
        plt.tight_layout()
        plt.savefig(bar_path)
        plt.close()
        
        # Grafik 2: Zaman Serisi (Deterministik)
        times = self.frame_timestamps if self.frame_timestamps else list(range(len(self.risk_over_time)))
        ts_path = os.path.join(tempfile.gettempdir(), "risk_time.png")
        plt.figure(figsize=(10, 2.5)) 
        plt.plot(times, self.risk_over_time, color='darkblue', linewidth=1)
        plt.fill_between(times, self.risk_over_time, color='skyblue', alpha=0.3)
        plt.title('Zamana Bağlı Kümülatif Risk İndeksi') 
        plt.xlabel('Saniye')
        plt.ylabel('Risk Puanı')
        plt.tight_layout()
        plt.savefig(ts_path)
        plt.close()
        
        # En Yüksek Riskli Anın Fotosu
        high_risk_img_path = os.path.join(tempfile.gettempdir(), "high_risk_frame.png")
        if self.highest_risk_frame is not None:
             cv2.imwrite(high_risk_img_path, self.highest_risk_frame)
        else:
             high_risk_img_path = None

        # --- Skor Hesaplama Mantığı ---
        total_seconds = max(1.0, self.frame_timestamps[-1] if self.frame_timestamps else 1.0)
        total_minutes = max(0.01, total_seconds / 60) 
        
        # Ağırlıklı Ceza Puanı Hesabı
        total_danger_points = (self.total_counts['ACİL DURUM'] * weights['acil']) + \
                              (self.total_counts['KRİTİK'] * weights['kritik']) + \
                              (self.total_counts['YÜKSEK'] * weights['yuksek'])
        
        danger_density_per_second = total_danger_points / total_seconds
        final_score_deduction = danger_density_per_second * weights['multiplier']
        
        if self.is_night_time: final_score_deduction *= 1.5 # Gece cezası artırılır
        suitability_score = max(0, min(100, 100 - final_score_deduction))
        
        # --- PDF ÇİZİMİ (CANVAS) ---
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        
        # 1. Başlık Alanı
        c.setFillColorRGB(0.1, 0.2, 0.5) 
        c.rect(0, height - 80, width, 80, fill=1, stroke=0)
        c.setFillColorRGB(1, 1, 1) 
        c.setFont(font_bold, 20)
        c.drawCentredString(width / 2, height - 50, "OTONOM SÜRÜŞ GÜZERGAH ANALİZ RAPORU")
        
        # 2. Genel Bilgiler
        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_bold, 10)
        c.drawString(40, height - 100, "ANALİZ DETAYLARI:")
        c.setFont(font_normal, 10)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.drawString(40, height - 115, f"Tarih: {now_str}")
        c.drawString(40, height - 130, f"Konum: {self.location_text}")
        c.drawString(40, height - 145, f"Süre: {total_seconds:.1f} sn ({total_minutes:.1f} dk)")
        if self.is_night_time:
            c.setFillColorRGB(1, 0, 0)
            c.drawString(40, height - 160, "KOŞUL: GECE SÜRÜŞÜ (Görüş Kısıtlı)")
            c.setFillColorRGB(0, 0, 0)

        # 3. Uygunluk Skoru Kutusu
        score_color = (0.2, 0.7, 0.2) if suitability_score > 80 else ((0.9, 0.5, 0.1) if suitability_score > 50 else (0.8, 0, 0))
        c.setStrokeColorRGB(*score_color)
        c.setLineWidth(3)
        c.roundRect(width - 200, height - 160, 160, 70, 10, stroke=1, fill=0)
        c.setFont(font_bold, 12)
        c.drawCentredString(width - 120, height - 110, "UYGUNLUK SKORU")
        c.setFont(font_bold, 28)
        c.setFillColorRGB(*score_color)
        c.drawCentredString(width - 120, height - 140, f"{suitability_score:.1f} / 100")
        c.setFillColorRGB(0, 0, 0)
        
        # 4. Tespit Tablosu (Renkli Kutular)
        y_stats = height - 200
        c.setFont(font_bold, 12)
        c.drawString(40, y_stats, "TESPİT ÖZETİ (Olay Bazlı)")
        
        def draw_stat_box(x, y, label, value, color_rgb):
            c.setFillColorRGB(*color_rgb)
            c.rect(x, y - 30, 90, 40, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont(font_bold, 14)
            c.drawCentredString(x + 45, y - 10, str(value))
            c.setFont(font_normal, 8)
            c.drawCentredString(x + 45, y - 25, label)

        draw_stat_box(40, y_stats - 10, "ACİL DURUM", self.total_counts['ACİL DURUM'], (0.8, 0, 0))
        draw_stat_box(140, y_stats - 10, "KRİTİK", self.total_counts['KRİTİK'], (1, 0.5, 0))
        draw_stat_box(240, y_stats - 10, "YÜKSEK", self.total_counts['YÜKSEK'], (1, 0.8, 0))
        draw_stat_box(340, y_stats - 10, "ORTA", self.total_counts['ORTA'], (0.2, 0.6, 0.8))
        draw_stat_box(440, y_stats - 10, "DÜŞÜK", self.total_counts['DÜŞÜK'], (0.2, 0.7, 0.2))
        
        # 5. Görsel Kanıtlar
        y_images = y_stats - 70
        c.setFillColorRGB(0, 0, 0)
        c.setFont(font_bold, 11)
        
        c.drawString(40, y_images, "1. Risk Yoğunluk Haritası (Kümülatif)")
        if heatmap_path and os.path.exists(heatmap_path):
            c.drawImage(heatmap_path, 40, y_images - 160, width=240, height=150)
            
        risk_img_title = "2. Tespit Edilen En Riskli An" if self.max_risk_score_so_far > 0 else "2. Örnek Trafik Akışı Görüntüsü"
        c.drawString(300, y_images, risk_img_title)
        if high_risk_img_path and os.path.exists(high_risk_img_path):
            c.drawImage(high_risk_img_path, 300, y_images - 160, width=240, height=150)
            
        # 6. Grafikler
        y_charts = y_images - 190
        c.setFont(font_bold, 11)
        c.drawString(40, y_charts, "3. Analiz Grafikleri")
        
        if os.path.exists(bar_path):
            c.drawImage(bar_path, 30, y_charts - 130, width=250, height=120)
        if os.path.exists(ts_path):
            c.drawImage(ts_path, 290, y_charts - 130, width=270, height=120)
            
        # 7. SONUÇ VE ÖNERİLER (Yapay Zeka Destekli Yorum)
        y_text = y_charts - 150
        c.setFont(font_bold, 12)
        c.drawString(40, y_text, "ANALİTİK SONUÇ VE ÖNERİLER:")
        
        # Dinamik metin oluşturma
        risk_density = "DÜŞÜK"
        if self.total_counts['ACİL DURUM'] > 0: risk_density = "ÇOK YÜKSEK"
        elif self.total_counts['KRİTİK'] > 5: risk_density = "YÜKSEK"
        elif self.total_counts['KRİTİK'] > 0: risk_density = "ORTA"
        
        result_eval = "UYGUN" if suitability_score > 80 else ("KISMİ RİSKLİ" if suitability_score > 50 else "UYGUN DEĞİL")
        
        analysis_summary = [
            f"• Analiz edilen güzergah boyunca toplam {total_unique_objects} adet dinamik nesne etkileşimi taranmıştır.",
            f"• Güzergahın ortalama risk yoğunluğu '{risk_density}' seviyesinde ölçülmüştür.",
            f"• Yapılan değerlendirme sonucunda yol geometrisi ve trafik akışı, otonom sürüş (Seviye 2+) için {result_eval} bulunmuştur."
        ]
        
        recommendations = []
        if self.total_counts['ACİL DURUM'] > 0: 
            recommendations.append("• KRİTİK: Güzergahta ani çarpışma riski tespit edildi. AEB (Acil Fren) ve Lidar hassasiyeti %20 artırılmalıdır.")
        elif self.total_counts['KRİTİK'] > 10:
            recommendations.append("• DİKKAT: Şerit ihlalleri yoğun. Yanal kontrol (Lateral Control) algoritmaları sıkılaştırılmalıdır.")
        
        if self.is_night_time: 
            recommendations.append("• KOŞUL: Düşük ışık koşulları mevcuttur. Termal kamera veya Radar verilerine öncelik verilmelidir.")
            
        if not recommendations:
            recommendations.append("• Standart otonom sürüş protokolleri ve mevcut sensör kalibrasyonu bu güzergah için yeterlidir.")

        # Metni sayfaya yazdır
        text_obj = c.beginText(40, y_text - 20)
        text_obj.setFont(font_normal, 9)
        text_obj.setLeading(14)
        
        for line in analysis_summary:
            text_obj.textLine(line)
        
        text_obj.textLine("") 
        text_obj.setFont(font_bold, 9)
        text_obj.textLine("ÖNERİLEN AKSİYONLAR:")
        text_obj.setFont(font_normal, 9)
        
        for line in recommendations:
            text_obj.textLine(line)
            
        c.drawText(text_obj)

        # 8. QR KOD ENTEGRASYONU
        try:
            import qrcode
            # GitHub repo linki (Proje tesliminde bu linki güncelleyebilirsiniz)
            qr = qrcode.make('https://github.com/abdullahserefoglu0-c/BOZ213-FINAL-Abdullah-Serefoglu-BIR-ALANIN-OTONOM-SURUS-ICIN-UYGUNLUK-TESTI-ARACI-GELISTIRMEK') 
            qr_path = os.path.join(tempfile.gettempdir(), "report_qr.png")
            qr.save(qr_path)
            
            c.drawImage(qr_path, width - 70, 20, width=50, height=50)
            c.setFont(font_normal, 6)
            c.drawCentredString(width - 45, 15, "Raporu Doğrula")
        except ImportError:
            print("UYARI: 'qrcode' kütüphanesi eksik, QR kod eklenemedi.")

        # Alt Bilgi (Footer)
        c.setFont(font_normal, 8) 
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawCentredString(width/2, 30, f"Bu teknik rapor {MODEL_NAME} yapay zeka modeli kullanılarak otomatik oluşturulmuştur.")
        
        c.showPage()
        c.save()
        
        # Geçici dosyaları temizle
        for p in [bar_path, ts_path, high_risk_img_path]:
            if p and os.path.exists(p): 
                try: os.remove(p)
                except: pass
        try: 
            if os.path.exists(qr_path): os.remove(qr_path)
        except: pass
                
        # GUI için kısa özet metni döndür
        gui_summary = f"ANALİZ TAMAMLANDI\nSkor: {suitability_score:.1f}\n"
        gui_summary += f"Acil Durumlar: {self.total_counts['ACİL DURUM']}\n"
        gui_summary += f"Toplam Nesne: {total_unique_objects}"

        return pdf_path, gui_summary, suitability_score

    def stop(self):
        """Thread'i güvenli bir şekilde durdurur."""
        self._is_running = False

# =============================================================================
# 4. KULLANICI ARAYÜZÜ SINIFI (GUI)
# =============================================================================

class MainWindow(QMainWindow):
    """
    Uygulamanın ana penceresi.
    PySide6 kullanarak oluşturulmuştur.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Alan Uygunluk Testi v11.4 — Şerit/Kaldırım Algılama") 
        self.setGeometry(120, 120, 1100, 750)

        # Durum Değişkenleri
        self.video_thread = None
        self.last_pdf_path = None
        self.video_path = None 
        
        # --- ARAYÜZ ELEMANLARI ---
        
        # Video Görüntü Alanı
        self.image_label = QLabel("Bir video seçin veya analiz başlatın.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; min-height: 520px;")
        
        # Özet Paneli
        self.summary_panel = QTextEdit()
        self.summary_panel.setReadOnly(True)
        self.summary_panel.setMaximumWidth(320)
        self.summary_panel.setMinimumHeight(200) 
        
        # Ayarlar Grubu (Validator ile sadece sayı girişi)
        float_validator = QDoubleValidator()
        float_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        
        self.weight_acil = QLineEdit("10.0")
        self.weight_acil.setValidator(float_validator)
        self.weight_kritik = QLineEdit("3.0")
        self.weight_kritik.setValidator(float_validator)
        self.weight_yuksek = QLineEdit("1.0")
        self.weight_yuksek.setValidator(float_validator)
        self.score_multiplier = QLineEdit("5.0")
        self.score_multiplier.setValidator(float_validator)
        
        settings_group = QGroupBox("Skorlama Ayarları (Parametreler)")
        form_layout = QFormLayout()
        form_layout.addRow("Acil Durum Ağırlığı:", self.weight_acil)
        form_layout.addRow("Kritik Risk Ağırlığı:", self.weight_kritik)
        form_layout.addRow("Yüksek Risk Ağırlığı:", self.weight_yuksek)
        form_layout.addRow("Final Skor Çarpanı:", self.score_multiplier)
        settings_group.setLayout(form_layout)
        
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("Konum bilgisi (örn: Ankara, Kampus Alanı)")
        
        # Canlı Risk Çubukları (Progress Bars)
        live_risk_group = QGroupBox("Canlı Risk Tespiti")
        live_layout = QVBoxLayout()
        self.bar_acil = QProgressBar()
        self.bar_kritik = QProgressBar()
        self.bar_yuksek = QProgressBar()
        self.bar_orta = QProgressBar()
        self.bar_dusuk = QProgressBar()
        self.bars = [self.bar_acil, self.bar_kritik, self.bar_yuksek, self.bar_orta, self.bar_dusuk]
        labels = ["ACİL DURUM", "KRİTİK", "YÜKSEK", "ORTA", "DÜŞÜK"]
        colors = [COLOR_MAP.get(l, (255,0,255)) for l in labels] 
        
        for bar, label, color_bgr in zip(self.bars, labels, colors):
            bar.setRange(0, 20) 
            bar.setValue(0)
            bar.setFormat(f"{label}: %v")
            color_hex = f"#{color_bgr[2]:02x}{color_bgr[1]:02x}{color_bgr[0]:02x}"
            # Yazı rengini arka plana göre ayarla
            text_color = "black" if label in ['YÜKSEK', 'ORTA'] else "white"
            bar.setStyleSheet(f"""
                QProgressBar {{ text-align: center; color: {text_color}; }}
                QProgressBar::chunk {{ background-color: {color_hex}; }}
            """)
            live_layout.addWidget(bar)
        live_risk_group.setLayout(live_layout)
        
        # Kontrol Butonları
        self.open_button = QPushButton("Dosya Aç (Video)")
        self.open_button.clicked.connect(self.open_file)
        self.start_button = QPushButton("Analiz Başlat")
        self.start_button.clicked.connect(self.start_analysis)
        self.start_button.setEnabled(False)
        self.stop_button = QPushButton("Durdur")
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        self.save_pdf_button = QPushButton("PDF Kaydet")
        self.save_pdf_button.clicked.connect(self.save_pdf)
        self.save_pdf_button.setEnabled(False)
        
        # Uygunluk Skoru Göstergesi
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setFormat("Uygunluk Skoru: %p%")
        self.score_bar.setStyleSheet("QProgressBar { min-height: 28px; font-size: 14px; }")
        
        # Layout Yerleşimi
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.save_pdf_button)
        
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Konum (PDF için):"))
        right_layout.addWidget(self.location_input)
        right_layout.addWidget(settings_group)
        right_layout.addWidget(live_risk_group) 
        right_layout.addStretch(1) 
        right_layout.addWidget(self.score_bar)
        right_layout.addWidget(self.summary_panel) 
        
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.image_label, 1) # Video solda, geniş yer kaplar
        main_layout.addLayout(right_layout)        # Ayarlar sağda
        
        layout = QVBoxLayout()
        layout.addLayout(main_layout)
        layout.addLayout(controls_layout)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.setStatusBar(QStatusBar(self))
        self.reset_live_bars() 

    def reset_live_bars(self):
        """Risk çubuklarını sıfırlar."""
        for bar in self.bars:
            bar.setValue(0)

    def open_file(self):
        """Dosya seçme diyaloğunu açar."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Video Seçin", "", "Video Dosyaları (*.mp4 *.avi *.mov)")
        if not file_path:
            return
        self.video_path = file_path
        self.image_label.setText(f"Seçilen: {os.path.basename(file_path)}")
        self.start_button.setEnabled(True)
        self.summary_panel.setText("Analiz için 'Analiz Başlat' butonuna basın.")
        self.score_bar.setValue(0)
        self.reset_live_bars() 
        self.save_pdf_button.setEnabled(False)
        self.last_pdf_path = None

    def start_analysis(self):
        """Analiz Thread'ini başlatır."""
        if not self.video_path:
            QMessageBox.warning(self, "Uyarı", "Önce bir video seçin.")
            return
        
        location_text = self.location_input.text().strip() or "(Konum girilmedi)"
        
        # Ağırlıkları al
        try:
            weights = {
                'acil': float(self.weight_acil.text()),
                'kritik': float(self.weight_kritik.text()),
                'yuksek': float(self.weight_yuksek.text()),
                'multiplier': float(self.score_multiplier.text())
            }
        except ValueError:
            QMessageBox.critical(self, "Hata", "Lütfen skorlama ayarları için geçerli sayılar (örn: 10.0, 3.0) girin.")
            return
            
        # Thread'i oluştur ve başlat
        self.video_thread = VideoThread(self.video_path, location_text, weights)
        
        # Sinyal bağlantıları
        self.video_thread.changePixmap.connect(self.set_image_from_thread)
        self.video_thread.updateLiveCounts.connect(self.update_live_bars)
        self.video_thread.analysisComplete.connect(self.handle_analysis_completion) 
        self.video_thread.updateCumulativeSummary.connect(self.update_cumulative_panel)
        
        self.video_thread.start()
        
        # Buton durumlarını güncelle
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.save_pdf_button.setEnabled(False)
        self.setStatusBarMessage("Analiz başlatıldı...")

    def stop_analysis(self):
        """Analizi manuel olarak durdurur."""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.setStatusBarMessage("Analiz durduruldu.")
        self.summary_panel.append("\n\n--- ANALİZ DURDURULDU ---")
        self.reset_live_bars() 

    @Slot(QImage)
    def set_image_from_thread(self, q_img):
        """Thread'den gelen görüntüyü ekrana basar."""
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio))

    @Slot(dict)
    def update_cumulative_panel(self, cumulative_unique_counts):
        """Sağ paneldeki toplam istatistikleri günceller."""
        summary_lines = ["--- ŞU ANA KADAR TOPLAM (Benzersiz) ---"] 
        summary_lines.append(f"ACIL DURUM: {cumulative_unique_counts.get('ACİL DURUM', 0)}")
        summary_lines.append(f"KRİTİK RISK: {cumulative_unique_counts.get('KRİTİK', 0)}")
        summary_lines.append(f"YUKSEK RISK: {cumulative_unique_counts.get('YÜKSEK', 0)}")
        summary_lines.append(f"ORTA RISK:  {cumulative_unique_counts.get('ORTA', 0)}")
        summary_lines.append(f"DUSUK RISK: {cumulative_unique_counts.get('DÜŞÜK', 0)}")
        self.summary_panel.setText("\n".join(summary_lines))

    @Slot(dict)
    def update_live_bars(self, counts):
        """Canlı risk çubuklarını günceller."""
        self.bar_acil.setValue(counts.get('ACİL DURUM', 0))
        self.bar_kritik.setValue(counts.get('KRİTİK', 0))
        self.bar_yuksek.setValue(counts.get('YÜKSEK', 0))
        self.bar_orta.setValue(counts.get('ORTA', 0))
        self.bar_dusuk.setValue(counts.get('DÜŞÜK', 0))

    @Slot(str, str, float)
    def handle_analysis_completion(self, pdf_path, report_text, score):
        """Analiz bittiğinde çalışır, sonucu gösterir."""
        self.last_pdf_path = pdf_path
        self.save_pdf_button.setEnabled(True)
        self.score_bar.setValue(int(score))
        
        self.summary_panel.setText(report_text) 
        
        QMessageBox.information(self, "Analiz Tamamlandı", 
                                f"Analiz tamamlandı. Otonom Sürüş Uygunluk Skoru: {score:.1f} / 100\n\n"
                                f"Detaylar ve önerilen önlemler yandaki panelde ve PDF raporundadır.")
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.setStatusBarMessage("Analiz tamamlandı. Rapor ve önlemler oluşturuldu.")

    def save_pdf(self):
        """Oluşturulan PDF'i kullanıcının seçtiği yere kaydeder."""
        if not self.last_pdf_path or not os.path.exists(self.last_pdf_path):
            QMessageBox.warning(self, "Hata", "Kaydedilecek bir PDF bulunamadı.")
            return
        default_name = f"rapor_{self.location_input.text().strip() or 'analiz'}_{datetime.now().strftime('%Y%m%d')}.pdf"
        file_path, _ = QFileDialog.getSaveFileName(self, "PDF Kaydet", default_name, "PDF Dosyası (*.pdf)")
        if not file_path:
            return
        try:
            shutil.copy(self.last_pdf_path, file_path)
            QMessageBox.information(self, "Kaydedildi", f"PDF kaydedildi:\n{file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Hata", f"PDF kaydedilemedi: {e}")

    def setStatusBarMessage(self, text, timeout=5000):
        self.statusBar().showMessage(text, timeout)

    def closeEvent(self, event):
        """Uygulama kapatılırken thread'leri temizler."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

#  .\venv\Scripts\activate
 #  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
 #  py "BİR_ALANIN_OTONOM_SÜRÜŞ_İÇİN_UYGUNLUK_TESTİ_ARACI_GELİŞTİRMEK.py"






"""
=============================================================================
KURULUM VE ÇALIŞTIRMA KILAVUZU (INSTALLATION GUIDE)
=============================================================================

Bu yazılımı çalıştırmak için aşağıdaki adımları takip ediniz.

1. GEREKSİNİMLER (PREREQUISITES)
-----------------------------------------------------------------------------
- Python 3.8 veya daha yeni bir sürümün yüklü olması gerekir.
- İnternet bağlantısı (YOLO yapay zeka modelini ilk açılışta indirmek için).

2. KURULUM (INSTALLATION)
-----------------------------------------------------------------------------
Gerekli kütüphaneleri yüklemek için Terminal veya Komut İstemi'ne (CMD) 
aşağıdaki komutu yapıştırıp Enter'a basın:

pip install ultralytics opencv-python PySide6 reportlab matplotlib qrcode[pil] Pillow numpy

Not: 'qrcode[pil]' ifadesi önemlidir, QR kod üretimi için gereklidir.

3. ÇALIŞTIRMA (EXECUTION)
-----------------------------------------------------------------------------
Kurulum tamamlandıktan sonra programı başlatmak için:

python "otonom araç projem.py"

4. ÖNEMLİ NOTLAR
-----------------------------------------------------------------------------
- İlk Çalıştırma: Program ilk açıldığında yaklaşık 6MB boyutundaki 'yolov8n.pt'
  yapay zeka modelini otomatik olarak indirecektir. Bu işlem bir kez yapılır.
- Türkçe Karakterler: PDF raporunda Türkçe karakterlerin düzgün görünmesi için
  Windows işletim sisteminde 'Arial' fontunun yüklü olması yeterlidir.
- Performans: Daha akıcı bir görüntü için NVIDIA ekran kartınız varsa
  PyTorch'un CUDA sürümünü yüklemeniz önerilir (Zorunlu değildir).

=============================================================================
"""










# Yol Alanı (Şerit) Tanımları
#ROAD_POLYGON_POINTS = [              
#    (0.46, 0.77), 
#    (0.50, 0.77), 
#    (0.65, 1.00), 
#    (0.30, 1.00) 
#]    
#--- [KARARLI BÖLGE AYARLARI] ---
#WARNING_ZONE_PERCENTAGE = 0.75  # Sarı Çizgi (Takip Mesafesi - Ekranın %60'ı)
#DANGER_ZONE_PERCENTAGE = 0.92   # Kırmızı Çizgi (Tampon Mesafesi - Ekranın %85'i)
#        # gerçek hayat videoları için










#   --- [KARARLI BÖLGE AYARLARI] ---
#WARNING_ZONE_PERCENTAGE = 0.60  # Sarı Çizgi (Takip Mesafesi - Ekranın %60'ı)
#DANGER_ZONE_PERCENTAGE = 0.90   # Kırmızı Çizgi (Tampon Mesafesi - Ekranın %85'i)
#
#Yol Alanı (Şerit) Tanımları
#ROAD_POLYGON_POINTS = [              
#    (0.40, 0.65), 
#    (0.60, 0.65), 
#    (0.85, 1.00), 
#    (0.16, 1.00) 


#   carla videoları için

