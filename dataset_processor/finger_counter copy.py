# Import library yang dibutuhkan
import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime

class FeatureExtractor:
    def __init__(self):
        # Set base path untuk dataset
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")
        
        # Set path untuk menyimpan hasil
        self.output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "processed_data")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            
        # Buat folder untuk gambar preprocessing
        self.preprocess_path = os.path.join(self.output_path, "preprocessed_images")
        if not os.path.exists(self.preprocess_path):
            os.makedirs(self.preprocess_path)
        
        # Parameter untuk pemrosesan gambar
        self.image_size = (224, 224)  # Ukuran standar untuk normalisasi
        self.blur_value = 15
        self.threshold_value = 200  # 127 Nilai threshold yang lebih tinggi untuk mengatasi bayangan
        
        # Validasi folder dataset
        if not os.path.exists(self.base_path):
            raise IOError("Folder dataset tidak ditemukan")
            
        # List untuk menyimpan hasil ekstraksi fitur
        self.features_data = []

    # fungsi untuk mengolah gambar ketika ada putih yang muncul tapi di sekitarnya tidak ada putih lainnya maka di buat hitam
    # tujuannya untuk mengurangi noise pada gambar,  sehingga hanya bagian tangan yang terlihat
    # nanti akan di panggil setelah adaptive threshold    
    def remove_noise(self, thresh):
        """Hapus noise kecil pada gambar threshold"""
        # Temukan kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Buat mask untuk menghapus noise
        mask = np.zeros_like(thresh)
        
        # Tentukan area minimum untuk kontur yang valid
        min_area = 75
        
        # Filter kontur berdasarkan area dan gambar hanya kontur besar pada mask
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
        # Kembalikan hasil dengan hanya kontur yang valid
        return mask
    
    # Fungsi untuk preprocessing gambar resize, grayscale, blur, dan adaptive threshold
    def preprocess_image(self, frame):
        """Preprocessing gambar: resize, grayscale, threshold"""
        # Resize gambar
        frame = cv2.resize(frame, self.image_size)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplikasikan blur untuk mengurangi noise
        blur = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0)
        
        # Aplikasikan adaptive threshold untuk mengatasi masalah pencahayaan
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Hapus noise kecil
        cleaned = self.remove_noise(thresh)
        
        return cleaned

    def get_features(self, thresh):
        """Ekstrak fitur dari gambar threshold"""
        try:
            # Cari kontur
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, None, [0, 0, 0]  # Return default features jika tidak ada kontur
                
            # Ambil kontur terbesar (tangan)
            contour = max(contours, key=cv2.contourArea)
            
            # Sederhanakan kontur untuk mengurangi noise
            epsilon = 0.01 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Hitung convex hull
            # Convex hull adalah batas luar dari kontur tangan
            # Convex hull adalah titik-titik dari batas luar kontur tangan
            # Jika tidak ada kontur, kita set ke 0
            try:
                hull = cv2.convexHull(contour, returnPoints=False)
                # Hitung defects
                defects = cv2.convexityDefects(contour, hull)
            except:
                # Jika gagal menghitung defects, gunakan hull points
                hull_points = cv2.convexHull(contour)
                return contour, hull_points, [0, 0, 0]
            
            # Ekstrak fitur
            features = []
            
            # Fitur 1: Jumlah defects yang signifikan (celah antar jari)
            # Hitung jumlah defects
            # Defects adalah array yang berisi informasi tentang celah pada kontur (kontur adalah sisi dari jari)
            # Setiap defect memiliki 4 elemen: start, end, far, depth
            # Start dan end adalah indeks dari kontur, far adalah titik terjauh dari garis antara start dan end
            # Depth adalah jarak dari titik terjauh ke garis antara start dan end
            # Kita akan menghitung sudut antara garis start-end dan garis start-far
            # Jika tidak ada defects, kita set ke 0
            # Jika ada defects, kita hitung sudut dan kedalaman
            n_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    
                    # Hitung sudut
                    a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    
                    # Hindari pembagian dengan nol
                    if b * c == 0:
                        continue
                        
                    angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                    
                    # Hitung kedalaman defect
                    depth = d / 256.0
                    
                    # Filter defects berdasarkan sudut dan kedalaman
                    if angle <= 90 and depth > 30:
                        n_defects += 1
                        
            features.append(n_defects)
            
            # Fitur 2: Rasio area kontur terhadap area convex hull
            # Hitung area kontur dan area convex hull
            # Kontur adalah kontur tangan, hull_points adalah convex hull
            # Area kontur adalah luas dari kontur tangan
            # Area convex hull adalah luas dari hull_points
            # Convex hull adalah batas luar dari kontur tangan
            # hull_points adalah titik-titik dari convex hull
            # Jika tidak ada hull_points, kita set ke 0
            hull_points = cv2.convexHull(contour)
            contour_area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(hull_points)
            if hull_area > 0:
                area_ratio = contour_area / hull_area
            else:
                area_ratio = 0
            features.append(area_ratio)
            
            # Fitur 3: Aspect ratio dari bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h if h > 0 else 0
            features.append(aspect_ratio)
            
            return contour, hull_points, features
            
        except Exception as e:
            print(f"Error dalam ekstraksi fitur: {str(e)}")
            return None, None, [0, 0, 0]

    def save_debug_image(self, original, thresh, contour, hull, features, label, filename):
        """Simpan gambar debug dengan visualisasi"""
        try:
            # Resize dan konversi semua gambar ke BGR (3 channel) dengan ukuran yang sama
            original_resized = cv2.resize(original, self.image_size)
            
            # Konversi grayscale ke BGR untuk visualisasi
            original_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
            original_gray_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
            
            # Proses blur
            original_blur = cv2.GaussianBlur(original_gray, (self.blur_value, self.blur_value), 0)
            original_blur_bgr = cv2.cvtColor(original_blur, cv2.COLOR_GRAY2BGR)
              # Proses threshold
            original_thresh = cv2.adaptiveThreshold(
                original_blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            original_thresh_bgr = cv2.cvtColor(original_thresh, cv2.COLOR_GRAY2BGR)
            
            # Aplikasikan remove noise
            cleaned_thresh = self.remove_noise(original_thresh)
            cleaned_thresh_bgr = cv2.cvtColor(cleaned_thresh, cv2.COLOR_GRAY2BGR)
            
            # Buat debug image dengan kontur
            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            if contour is not None and hull is not None:
                cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(debug, [hull], -1, (0, 0, 255), 2)
              # Gabungkan semua gambar
            top_row = np.hstack((
                original_resized,      # Gambar asli
                original_gray_bgr,     # Hasil grayscale
                original_blur_bgr,     # Hasil blur
                original_thresh_bgr,   # Hasil threshold
                cleaned_thresh_bgr,    # Hasil penghapusan noise
                debug                  # Hasil final dengan kontur
            ))
            
            # Tambahkan label dan informasi
            # Posisikan teks di gambar asli
            cv2.putText(top_row, f'Original', (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar grayscale
            x_offset = self.image_size[0]
            cv2.putText(top_row, f'Grayscale', (x_offset + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar blur
            x_offset = self.image_size[0] * 2
            cv2.putText(top_row, f'Blur', (x_offset + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
              # Posisikan teks di gambar threshold
            x_offset = self.image_size[0] * 3
            cv2.putText(top_row, f'Threshold', (x_offset + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar cleaned threshold
            x_offset = self.image_size[0] * 4
            cv2.putText(top_row, f'Noise Removed', (x_offset + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar debug
            x_offset = self.image_size[0] * 5
            cv2.putText(top_row, f'Label: {label}', (x_offset + 10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(top_row, f'Defects: {features[0]}', (x_offset + 10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(top_row, f'Area Ratio: {features[1]:.3f}', (x_offset + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(top_row, f'Aspect Ratio: {features[2]:.3f}', (x_offset + 10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Simpan gambar hasil
            output_filename = os.path.join(self.preprocess_path, f'debug_{filename}')
            cv2.imwrite(output_filename, top_row)
            
        except Exception as e:
            print(f"Error dalam menyimpan gambar debug: {str(e)}")
            import traceback
            traceback.print_exc()  # Tambahkan ini untuk debug yang lebih detail

    def process_dataset(self):
        """Proses seluruh dataset dan ekstrak fitur"""
        print("\n=== Ekstraksi Fitur Dataset ===")
        print("Processing...")
        
        # Iterasi setiap folder (label)
        for label in range(6):
            folder_path = os.path.join(self.base_path, str(label))
            print(f"\nMemproses folder {label}...")
            
            # Proses setiap gambar
            image_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in image_files:
                img_path = os.path.join(folder_path, img_file)
                
                # Baca dan proses gambar
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Tidak dapat membaca: {img_file}")
                    continue
                    
                # Preprocessing
                thresh = self.preprocess_image(frame)
                
                # Ekstrak fitur
                contour, hull, features = self.get_features(thresh)
                
                # Simpan gambar debug
                self.save_debug_image(frame, thresh, contour, hull, features, 
                                    label, img_file)
                
                # Simpan hasil ekstraksi
                self.features_data.append({
                    'filename': img_file,
                    'label': label,
                    'n_defects': features[0],
                    'area_ratio': features[1],
                    'aspect_ratio': features[2]
                })
                
        # Simpan hasil ke CSV
        df = pd.DataFrame(self.features_data)
        csv_path = os.path.join(self.output_path, 'features.csv')
        df.to_csv(csv_path, index=False)
        
        print("\nProses selesai!")
        print(f"Hasil ekstraksi fitur disimpan di: {csv_path}")
        print(f"Gambar debug disimpan di: {self.preprocess_path}")

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process_dataset()