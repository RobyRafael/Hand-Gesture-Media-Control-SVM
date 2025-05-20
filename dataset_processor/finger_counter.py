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
        
        # List untuk menyimpan hasil ekstraksi fitur
        self.features_data = []
        
        # Validasi folder dataset dan buat struktur output
        self.validate_and_setup_folders()
        
    def validate_and_setup_folders(self):
        """Validasi keberadaan folder dataset dan setup folder output"""
        print("Memeriksa dan membuat struktur folder...")
        
        # Cek folder dataset
        if not os.path.exists(self.base_path):
            raise IOError(f"Folder dataset tidak ditemukan di: {self.base_path}")
            
        # Cek subfolder dataset (0-5)
        for label in range(6):
            folder_path = os.path.join(self.base_path, str(label))
            if not os.path.exists(folder_path):
                raise IOError(f"Folder dataset {label} tidak ditemukan di: {folder_path}")
            
            # Hitung jumlah gambar
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"Found {len(image_files)} images in folder {label}")
        
        print("\nMembuat struktur folder output...")
        
        # Buat folder output utama
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created: {self.output_path}")
        
        # Buat subfolder untuk setiap label
        for label in range(6):
            label_path = os.path.join(self.output_path, str(label))
            if not os.path.exists(label_path):
                os.makedirs(label_path)
                print(f"Created: {label_path}")
            
            # Buat subfolder untuk setiap tahap preprocessing
            stages = ['1_resized', '2_grayscale', '3_blur', '4_threshold', '5_final']
            for stage in stages:
                stage_path = os.path.join(label_path, stage)
                if not os.path.exists(stage_path):
                    os.makedirs(stage_path)
                    print(f"Created: {stage_path}")
        
        print("\nStruktur folder siap digunakan!")

    def save_preprocessing_step(self, image, label, stage, filename):
        """Menyimpan hasil setiap tahap preprocessing"""
        # Tentukan path berdasarkan label dan tahap
        save_path = os.path.join(self.output_path, str(label), stage, f"proc_{filename}")
        
        # Konversi ke BGR jika grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        # Simpan gambar
        cv2.imwrite(save_path, image)
        
    def preprocess_image(self, frame, label, filename):
        """Preprocessing gambar dengan menyimpan setiap tahap"""
        # 1. Resize gambar
        resized = cv2.resize(frame, self.image_size)
        self.save_preprocessing_step(resized, label, '1_resized', filename)
        
        # 2. Konversi ke grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray_colored = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Untuk visualisasi
        self.save_preprocessing_step(gray_colored, label, '2_grayscale', filename)
        
        # 3. Aplikasikan blur
        blur = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0)
        blur_colored = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)  # Untuk visualisasi
        self.save_preprocessing_step(blur_colored, label, '3_blur', filename)
        
        # 4. Aplikasikan threshold
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        thresh_colored = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Untuk visualisasi
        self.save_preprocessing_step(thresh_colored, label, '4_threshold', filename)
        
        return thresh

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
            # Buat salinan gambar threshold untuk debug
            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            if contour is not None and hull is not None:
                # Gambar kontur
                cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
                # Gambar hull
                cv2.drawContours(debug, [hull], -1, (0, 0, 255), 2)
            
            # Buat gambar komposit (original dan debug side by side)
            original_resized = cv2.resize(original, (self.image_size[0], self.image_size[1]))
            composite = np.hstack((original_resized, debug))
            
            # Tambah informasi
            cv2.putText(composite, f'Label: {label}', (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(composite, f'Defects: {features[0]}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(composite, f'Area Ratio: {features[1]:.3f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(composite, f'Aspect Ratio: {features[2]:.3f}', (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Simpan gambar
            output_filename = os.path.join(self.preprocess_path, f'debug_{filename}')
            cv2.imwrite(output_filename, composite)
            
        except Exception as e:
            print(f"Error dalam menyimpan gambar debug: {str(e)}")

    def save_final_result(self, original, thresh, contour, hull, features, label, filename):
        """Menyimpan hasil akhir dengan visualisasi lengkap"""
        try:
            # Buat visualisasi debug
            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            if contour is not None and hull is not None:
                # Gambar kontur dan hull
                cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
                cv2.drawContours(debug, [hull], -1, (0, 0, 255), 2)
            
            # Resize original untuk visualisasi
            original_resized = cv2.resize(original, (self.image_size[0], self.image_size[1]))
            
            # Buat komposit 4 gambar (2x2 grid)
            top_row = np.hstack((original_resized, debug))
            
            # Tambahkan informasi
            info_image = np.zeros_like(original_resized)
            cv2.putText(info_image, f'Label: {label}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, f'Defects: {features[0]}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, f'Area Ratio: {features[1]:.3f}', (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(info_image, f'Aspect Ratio: {features[2]:.3f}', (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Simpan hasil akhir
            final_path = os.path.join(self.output_path, str(label), '5_final', f'final_{filename}')
            cv2.imwrite(final_path, top_row)
            
        except Exception as e:
            print(f"Error dalam menyimpan hasil akhir: {str(e)}")

    def process_dataset(self):
        """Proses seluruh dataset dan ekstrak fitur"""
        print("\n=== Ekstraksi Fitur Dataset ===")
        total_processed = 0
        total_images = 0
        
        # Hitung total gambar
        for label in range(6):
            folder_path = os.path.join(self.base_path, str(label))
            if os.path.exists(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                             if f.endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(image_files)
        
        print(f"Total gambar yang akan diproses: {total_images}")
        print("Processing...")
        
        # Iterasi setiap folder (label)
        for label in range(6):
            folder_path = os.path.join(self.base_path, str(label))
            
            # Proses setiap gambar
            image_files = [f for f in os.listdir(folder_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            print(f"\nMemproses folder {label} ({len(image_files)} gambar)...")
            
            for i, img_file in enumerate(image_files, 1):
                img_path = os.path.join(folder_path, img_file)
                print(f"[{total_processed + 1}/{total_images}] Processing: {img_file}")
                
                try:
                    # Baca gambar
                    frame = cv2.imread(img_path)
                    if frame is None:
                        print(f"ERROR: Tidak dapat membaca: {img_file}")
                        continue
                    
                    # Preprocessing dengan menyimpan setiap tahap
                    thresh = self.preprocess_image(frame, label, img_file)
                    
                    # Ekstrak fitur
                    contour, hull, features = self.get_features(thresh)
                    
                    # Simpan hasil akhir
                    self.save_final_result(frame, thresh, contour, hull, features, 
                                         label, img_file)
                    
                    # Simpan hasil ekstraksi ke list
                    self.features_data.append({
                        'filename': img_file,
                        'label': label,
                        'n_defects': features[0],
                        'area_ratio': features[1],
                        'aspect_ratio': features[2]
                    })
                    
                    total_processed += 1
                    if total_processed % 10 == 0:
                        print(f"Progress: {total_processed}/{total_images} ({(total_processed/total_images)*100:.1f}%)")
                        
                except Exception as e:
                    print(f"ERROR processing {img_file}: {str(e)}")
                    continue
        
        # Simpan hasil ke CSV
        if self.features_data:
            df = pd.DataFrame(self.features_data)
            csv_path = os.path.join(self.output_path, 'features.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nBerhasil menyimpan {len(self.features_data)} hasil ekstraksi ke: {csv_path}")
        else:
            print("\nWARNING: Tidak ada data yang berhasil diekstrak!")
        
        print(f"\nProses selesai! {total_processed}/{total_images} gambar berhasil diproses.")