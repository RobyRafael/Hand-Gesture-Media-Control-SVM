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
    def detect_wrist_and_crop(self, cleaned):
        """Deteksi pergelangan tangan dan crop gambar dari pergelangan ke ujung jari
           Menganggap semua piksel putih sebagai bagian dari tangan"""
        # Cari semua piksel putih dalam gambar
        white_pixels = np.where(cleaned == 255)
        
        # Jika tidak ada piksel putih, kembalikan gambar asli
        if len(white_pixels[0]) == 0:
            return cleaned
        
        # Temukan batas atas (ujung jari) dan batas bawah (pergelangan tangan)
        highest_y = np.min(white_pixels[0])  # Nilai y terkecil (ujung jari)
        lowest_y = np.max(white_pixels[0])   # Nilai y terbesar (pergelangan tangan)
        
        # Tambahkan sedikit padding
        padding = 10
        highest_y = max(0, highest_y - padding)
        lowest_y = min(cleaned.shape[0] - 1, lowest_y + padding)
        
        # Buat gambar hasil crop dari pergelangan ke ujung jari
        # Crop berdasarkan koordinat y saja, pertahankan lebar penuh
        cropped = cleaned[highest_y:lowest_y + 1, 0:cleaned.shape[1]]
        
        return cropped
      
    def detect_wrist_horizontal(self, image):
        """Deteksi pergelangan tangan berdasarkan lebar terkecil dan crop dari pergelangan ke atas
           Pergelangan tangan diasumsikan sebagai titik dengan lebar paling sempit"""
        
        # Pastikan gambar tidak kosong
        if image is None or image.size == 0:
            return image
            
        # Ambil dimensi gambar
        height, width = image.shape
        
        # Variabel untuk menyimpan informasi pergelangan tangan
        min_distance = width  # Nilai terbesar untuk mencari jarak terkecil
        wrist_row = -1  # Baris yang akan dianggap pergelangan tangan
        
        # Loop untuk memeriksa setiap baris dalam gambar untuk mencari lebar terkecil
        for i in range(height):
            # Temukan piksel putih di baris ini
            white_pixels_in_row = np.where(image[i, :] == 255)[0]
            
            # Jika ada piksel putih di baris ini
            if len(white_pixels_in_row) > 0:
                left = white_pixels_in_row[0]  # Piksel putih paling kiri
                right = white_pixels_in_row[-1]  # Piksel putih paling kanan
                
                # Hitung lebar area putih
                distance = right - left
                
                # Jika lebar ini lebih kecil dari sebelumnya dan bukan 0
                if distance < min_distance and distance > 0:
                    min_distance = distance
                    wrist_row = i
        
        # Jika pergelangan tangan terdeteksi
        if wrist_row != -1:
            # Tambahkan padding untuk memastikan bagian pergelangan tangan terlihat
            padding = 10
            wrist_row = min(height - 1, wrist_row + padding)
            
            # Crop gambar dari atas hingga ke pergelangan tangan
            cropped = image[:wrist_row + 1, :]
            return cropped
        else:
            # Jika tidak terdeteksi, kembalikan gambar asli
            return image
    
    def preprocess_image(self, frame):
        """Preprocessing gambar: resize, grayscale, threshold"""
        # Resize gambar
        frame = cv2.resize(frame, self.image_size)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Aplikasikan blur untuk mengurangi noise
        blur = cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), 0) # 0 untuk 
        
        # Aplikasikan adaptive threshold untuk mengatasi masalah pencahayaan
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            6    # Constant subtracted from mean
        )
        
        # Hapus noise kecil
        cleaned = self.remove_noise(thresh)
        
        # Deteksi pergelangan tangan dan crop gambar secara vertikal (tinggi)
        cropped_upperlimb = self.detect_wrist_and_crop(cleaned)
        
        # Deteksi pergelangan tangan secara horizontal (lebar) dan crop lagi
        cropped_hand = self.detect_wrist_horizontal(cropped_upperlimb)
        
        return cropped_hand
    
    # Fungsi untuk ekstrak fitur dari gambar cropped hand    
    def get_features(self, thresh):
        """Ekstrak fitur dari gambar cropped hand secara manual tanpa cv2 atau np"""
        try:
            # Memastikan gambar tidak kosong
            if thresh is None or len(thresh) == 0:
                return None, None, [0, 0, 0, 0, 0, 0]
            
            # Fitur-fitur yang akan diekstrak
            features = []
            
            # Dapatkan dimensi gambar
            height, width = thresh.shape if hasattr(thresh, 'shape') else (len(thresh), len(thresh[0]))
            print(f"Processing cropped hand of size: {height}x{width}")

            # Mencari semua piksel putih
            white_pixels = []
            for y in range(height):
                for x in range(width):
                    if thresh[y][x] == 255:  # Piksel putih
                        white_pixels.append((x, y))
            
            # Jika tidak ada piksel putih, kembalikan nilai default
            if not white_pixels:
                return None, None, [0, 0, 0, 0, 0, 0]
                
            # 1. Estimasi jumlah jari berdasarkan clustering piksel
            n_fingers = self.estimate_fingers_count(white_pixels, height, width)
            features.append(n_fingers)
                
            # 2. Hitung rasio area: tinggi vs lebar dari bounding box
            min_x, max_x, min_y, max_y = self.find_bounding_box(white_pixels)
            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            
            # Aspect ratio: lebar/tinggi
            aspect_ratio = float(box_width) / box_height if box_height > 0 else 0
            features.append(aspect_ratio)
            
            # 3. Hitung rasio area
            convex_hull = self.create_simple_convex_hull(white_pixels)
            hull_area = self.calculate_polygon_area(convex_hull)
            hand_area = len(white_pixels)
            
            # Rasio area: piksel tangan / area convex hull
            area_ratio = hand_area / hull_area if hull_area > 0 else 0
            features.append(area_ratio)
            
            # 4 & 5. Mean distance dan standar deviasi
            mean_distance, std_dev, density = self.calculate_pixel_distribution(white_pixels, height, width)
            features.append(mean_distance)
            features.append(std_dev)
            
            # 6. Kepadatan piksel
            features.append(density)
            
            # Normalisasi fitur
            normalized_features = self.normalize_features(features)
            
            # Buat representasi kontur dan hull untuk visualisasi
            visual_contour = self.create_visual_contour(white_pixels)
            visual_hull = self.create_visual_contour(convex_hull)
            
            return visual_contour, visual_hull, normalized_features
            
        except Exception as e:
            print(f"Error dalam ekstraksi fitur manual: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None, [0, 0, 0, 0, 0, 0]
            
    # Fungsi untuk normalisasi fitur
    def normalize_features(self, features):
        """Normalisasi fitur ke range [0,1]"""
        # Nilai maksimum tipikal untuk setiap fitur berdasarkan eksperimen
        max_values = [4, 1, 5, 1, 1, 1]  # sesuaikan nilai ini berdasarkan data
        
        normalized = []
        
        for i, feature in enumerate(features):
            # Batasi nilai ke range [0,1]
            if max_values[i] > 0:
                norm_val = min(1.0, max(0.0, feature / max_values[i]))
            else:
                norm_val = 0
            normalized.append(norm_val)
            
        return normalized
        
    # Fungsi untuk mendapatkan mean dan standar deviasi dari cropped hand
    def get_mean_area(self, cropped_hand):
        """Menghitung rata-rata area, std deviasi, dan densitas dari piksel putih"""
        height, width = cropped_hand.shape
        print(f"Processing cropped hand of size: {height}x{width}")
        pixel_positions = []
        white_pixels = 0
        
        # Temukan semua piksel putih dan catat posisinya
        for y in range(height):
            for x in range(width):
                if cropped_hand[y, x] == 255:
                    pixel_positions.append((x, y))
                    white_pixels += 1

        # Jika tidak ada piksel putih, kembalikan nilai default
        if white_pixels == 0:
            return 0, 0, 0
        
        # Hitung centroid (titik pusat)
        sum_x = sum(pos[0] for pos in pixel_positions)
        sum_y = sum(pos[1] for pos in pixel_positions)
        center_x = sum_x / white_pixels
        center_y = sum_y / white_pixels
        
        # Hitung jarak dari setiap piksel ke centroid
        distances = []
        for pos in pixel_positions:
            dist = ((pos[0] - center_x)**2 + (pos[1] - center_y)**2)**0.5
            distances.append(dist)
        
        # Hitung mean (rata-rata jarak)
        mean_distance = sum(distances) / len(distances) if distances else 0
        
        # Hitung standard deviation (standar deviasi jarak)
        sum_squared_diff = sum((d - mean_distance)**2 for d in distances) if distances else 0
        std_dev = (sum_squared_diff / len(distances))**0.5 if distances else 0
        
        # Hitung densitas (proporsi piksel putih terhadap total area)
        total_area = height * width
        density = white_pixels / total_area if total_area > 0 else 0
        
        return mean_distance, std_dev, density
    
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
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                11,  # Block size
                6    # Constant subtracted from mean
            )
            original_thresh_bgr = cv2.cvtColor(original_thresh, cv2.COLOR_GRAY2BGR)
            
            # Aplikasikan remove noise
            cleaned_thresh = self.remove_noise(original_thresh)
            cleaned_thresh_bgr = cv2.cvtColor(cleaned_thresh, cv2.COLOR_GRAY2BGR)
            
            # Aplikasikan crop tangan vertikal (tinggi)
            cropped_upperlimb = self.detect_wrist_and_crop(cleaned_thresh)
            # Resize kembali agar sesuai dengan ukuran gambar lain untuk visualisasi
            cropped_upperlimb_resized = cv2.resize(cropped_upperlimb, self.image_size)
            cropped_upperlimb_bgr = cv2.cvtColor(cropped_upperlimb_resized, cv2.COLOR_GRAY2BGR)
            
            # Aplikasikan crop tangan horizontal (lebar)
            cropped_hand = self.detect_wrist_horizontal(cropped_upperlimb)
            # Resize kembali agar sesuai dengan ukuran gambar lain untuk visualisasi
            cropped_hand_resized = cv2.resize(cropped_hand, self.image_size)
            cropped_hand_bgr = cv2.cvtColor(cropped_hand_resized, cv2.COLOR_GRAY2BGR)
            
            # Buat debug image dengan kontur
            # Pastikan semua gambar memiliki ukuran yang sama
            thresh_resized = cv2.resize(thresh, self.image_size)
            debug = cv2.cvtColor(thresh_resized, cv2.COLOR_GRAY2BGR)
            
            if contour is not None and hull is not None:
                # Skalakan kontur ke ukuran gambar baru
                scale_x = self.image_size[0] / float(thresh.shape[1])
                scale_y = self.image_size[1] / float(thresh.shape[0])
                
                # Skalakan kontur
                scaled_contour = np.array([[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in contour])
                scaled_hull = np.array([[[int(pt[0][0] * scale_x), int(pt[0][1] * scale_y)]] for pt in hull])
                
                cv2.drawContours(debug, [scaled_contour], -1, (0, 255, 0), 2)
                cv2.drawContours(debug, [scaled_hull], -1, (0, 0, 255), 2)
            
            # Gabungkan semua gambar
            top_row = np.hstack((
                original_resized,      # Gambar resized asli
                original_gray_bgr,     # Hasil grayscale
                original_blur_bgr,     # Hasil blur
                original_thresh_bgr,   # Hasil threshold
                cleaned_thresh_bgr,    # Hasil penghapusan noise
                cropped_upperlimb_bgr, # Hasil crop vertikal (tinggi)
                cropped_hand_bgr,      # Hasil crop horizontal (lebar)
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
            
            # Posisikan teks di gambar upper limb cropped
            x_offset = self.image_size[0] * 5
            cv2.putText(top_row, f'Upper Limb Crop', (x_offset + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar hand cropped
            x_offset = self.image_size[0] * 6
            cv2.putText(top_row, f'Hand Cropped', (x_offset + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Posisikan teks di gambar debug
            x_offset = self.image_size[0] * 7
            cv2.putText(top_row, f'Contours', (x_offset + 10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Tambahkan informasi label dan fitur dalam format yang lebih kompak
            feature_names = ['n_fingers', 'aspect_ratio', 'area_ratio', 'mean_dist', 'std_dev', 'density']
            
            # Buat panel informasi dengan tinggi tetap 150px (tidak terlalu tinggi)
            feature_info = np.zeros((150, top_row.shape[1], 3), dtype=np.uint8)
            
            # Tambahkan judul di tengah atas
            title_text = f'Label: {label}'
            title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            title_x = (feature_info.shape[1] - title_size[0]) // 2  # Posisi tengah
            cv2.putText(feature_info, title_text, (title_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            
            # Bagi informasi fitur menjadi 3 kolom
            col_width = feature_info.shape[1] // 3
            
            # Kolom 1: fitur 0-1
            for i in range(2):
                if i < len(features):
                    name, feat = feature_names[i], features[i]
                    cv2.putText(feature_info, f'{name}: {feat:.3f}', (50, 70 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Kolom 2: fitur 2-3
            for i in range(2):
                idx = i + 2
                if idx < len(features):
                    name, feat = feature_names[idx], features[idx]
                    cv2.putText(feature_info, f'{name}: {feat:.3f}', (col_width + 50, 70 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Kolom 3: fitur 4-5
            for i in range(2):
                idx = i + 4
                if idx < len(features):
                    name, feat = feature_names[idx], features[idx]
                    cv2.putText(feature_info, f'{name}: {feat:.3f}', (2*col_width + 50, 70 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Gabungkan gambar dengan informasi fitur (tanpa resize yang bisa menyebabkan stretch)
            final_image = np.vstack((top_row, feature_info))
            
            # Simpan gambar hasil
            output_filename = os.path.join(self.preprocess_path, f'debug_{filename}')
            cv2.imwrite(output_filename, final_image)
            
        except Exception as e:
            print(f"Error dalam menyimpan gambar debug: {str(e)}")
            import traceback
            traceback.print_exc()  # Debug lebih detail

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
                    'aspect_ratio': features[2],
                    'mean_distance': features[3],
                    'std_dev': features[4],
                    'density': features[5]
                })
                
        # Simpan hasil ke CSV
        df = pd.DataFrame(self.features_data)
        csv_path = os.path.join(self.output_path, 'features.csv')
        df.to_csv(csv_path, index=False)
        
        print("\nProses selesai!")
        print(f"Hasil ekstraksi fitur disimpan di: {csv_path}")
        print(f"Gambar debug disimpan di: {self.preprocess_path}")    
        
    def detect_wrist_horizontal(self, cropped):
        """Deteksi pergelangan tangan secara horizontal (lebar pergelangan tangan)
        dan crop gambar dari pergelangan tangan sampai ujung jari"""
        height, width = cropped.shape
        
        # Inisialisasi variabel untuk menyimpan informasi lebar pergelangan
        min_distance = width  # Nilai terbesar untuk mencari jarak terkecil
        wrist_y = -1  # Baris yang akan dianggap pergelangan tangan
        
        # Scan setiap baris dari atas ke bawah
        for y in range(height):
            # Mencari piksel putih terkiri dan terkanan di baris ini
            white_pixels = np.where(cropped[y, :] == 255)[0]
            
            # Jika terdapat piksel putih di baris ini
            if len(white_pixels) > 0:
                left_most = np.min(white_pixels)
                right_most = np.max(white_pixels)
                
                # Menghitung jarak antara piksel putih terkiri dan terkanan
                distance = right_most - left_most
                
                # Menyesuaikan kriteria jarak, lebih fleksibel terhadap pergelangan tangan
                if distance < min_distance and distance > 0 and y < height // 2:  # Membatasi pada area lebih atas
                    min_distance = distance
                    wrist_y = y
                
        # Crop gambar dari pergelangan tangan (wrist_y) ke atas
        hand_crop = cropped[:wrist_y + 1, :]
        
        # Tambahkan padding di bagian bawah jika perlu
        padding = 80  # Padding lebih banyak untuk memastikan area jari terpotong
        if wrist_y + padding < height:
            padding_rows = cropped[wrist_y + 1:min(wrist_y + padding + 1, height), :]
            hand_crop = np.vstack((hand_crop, padding_rows))
        
        return hand_crop
    
    def find_bounding_box(self, white_pixels):
        """Temukan bounding box dari sekumpulan piksel putih"""
        if not white_pixels:
            return 0, 0, 0, 0
            
        # Inisialisasi dengan nilai ekstrem
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')
        
        # Temukan batas-batas
        for x, y in white_pixels:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            
        return min_x, max_x, min_y, max_y
        
    def create_simple_convex_hull(self, points):
        """Buat convex hull sederhana menggunakan algoritma Gift Wrapping / Jarvis March"""
        if len(points) <= 3:
            return points  # Jika poin <= 3, semua poin adalah hull
            
        # Fungsi untuk menentukan orientasi tiga poin
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # kolinear
            return 1 if val > 0 else 2  # Searah jarum jam atau berlawanan jarum jam
        
        # Temukan titik dengan koordinat y terkecil (dan jika ada beberapa, ambil yang paling kiri)
        start_idx = 0
        for i in range(1, len(points)):
            if points[i][1] < points[start_idx][1] or (points[i][1] == points[start_idx][1] and points[i][0] < points[start_idx][0]):
                start_idx = i
        
        hull = []
        p = start_idx
        q = 0
        while True:
            # Tambahkan titik saat ini ke hasil
            hull.append(points[p])
            
            # Temukan titik yang paling berlawanan jarum jam dari titik saat ini
            q = (p + 1) % len(points)
            for i in range(len(points)):
                # Jika i lebih berlawanan jarum jam dari current q
                if orientation(points[p], points[i], points[q]) == 2:
                    q = i
            
            # q sekarang menjadi p untuk iterasi berikutnya
            p = q
            
            # Keluar dari loop jika kembali ke titik awal
            if p == start_idx:
                break
                
        return hull
    
    def calculate_polygon_area(self, vertices):
        """Hitung area polygon menggunakan formula Shoelace"""
        # Harus ada minimal 3 verteks untuk membuat polygon
        if len(vertices) < 3:
            return 0
            
        area = 0
        for i in range(len(vertices)):
            j = (i + 1) % len(vertices)
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
            
        area = abs(area) / 2.0
        return area
        
    def calculate_pixel_distribution(self, white_pixels, height, width):
        """Hitung distribusi piksel: mean distance, std dev, dan densitas"""
        if not white_pixels:
            return 0, 0, 0
            
        # Hitung centroid (titik pusat)
        sum_x = sum(pos[0] for pos in white_pixels)
        sum_y = sum(pos[1] for pos in white_pixels)
        n_pixels = len(white_pixels)
        center_x = sum_x / n_pixels
        center_y = sum_y / n_pixels
        
        # Hitung jarak dari setiap piksel ke centroid
        distances = []
        for x, y in white_pixels:
            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
            distances.append(dist)
            
        # Hitung mean distance
        mean_distance = sum(distances) / len(distances)
        print(f"Perhitungan dari {n_pixels} piksel putih: mean distance = {mean_distance}")

        
        # Hitung standard deviation
        sum_squared_diff = sum((d - mean_distance)**2 for d in distances)
        std_dev = (sum_squared_diff / len(distances))**0.5
        
        # Hitung densitas (proporsi piksel putih terhadap total area)
        total_area = height * width
        density = n_pixels / total_area if total_area > 0 else 0

        print(f"Total area: {total_area}, Densitas: {density}, Standar deviasi: {std_dev}")
        
        return mean_distance, std_dev, density
    
    def create_visual_contour(self, points):
        """Buat representasi kontur untuk visualisasi"""
        if not points:
            return None
            
        # Konversi format poin untuk visualisasi dalam format yang cocok untuk cv2.drawContours
        contour = []
        for x, y in points:
            contour.append([[x, y]])
            
        return np.array(contour, dtype=np.int32)
        
    def estimate_fingers_count(self, white_pixels, height, width):
        """Estimasi jumlah jari dari gambar tangan"""
        if not white_pixels:
            return 0
            
        # Tentukan centroid tangan
        sum_x = sum(p[0] for p in white_pixels)
        sum_y = sum(p[1] for p in white_pixels)
        center_x = sum_x / len(white_pixels)
        center_y = sum_y / len(white_pixels)
        
        # Bagi gambar menjadi region angular untuk mendeteksi jari
        angular_bins = [0] * 36  # 36 bin untuk 10 derajat masing-masing
        max_dist = 0
        
        # Hitung jarak terjauh dari centroid
        for x, y in white_pixels:
            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
            max_dist = max(max_dist, dist)
        
        # Threshold jarak untuk mendeteksi ujung jari (ambil 70% dari jarak maks)
        dist_threshold = 0.7 * max_dist
        
        # Temukan piksel yang jauh dari centroid
        for x, y in white_pixels:
            dist = ((x - center_x)**2 + (y - center_y)**2)**0.5
            if dist > dist_threshold:
                # Hitung sudut
                angle = 0
                if x - center_x != 0:  # Hindari pembagian dengan nol
                    # Hitung sudut dalam derajat
                    dy = y - center_y
                    dx = x - center_x
                    angle_rad = 0
                    
                    # Hitung arctan secara manual
                    if dx > 0:  # Kuadran I dan IV
                        if dy >= 0:  # Kuadran I
                            angle_rad = 0  # Akan ditambah dengan arctan(dy/dx)
                        else:  # Kuadran IV
                            angle_rad = 6.28318  # 2π, akan dikurangi arctan(abs(dy)/dx)
                    else:  # Kuadran II dan III
                        angle_rad = 3.14159  # π
                    
                    # Hitung arctan(abs(dy)/abs(dx)) secara manual dengan aproksimasi
                    ratio = abs(dy) / abs(dx) if abs(dx) > 0 else float('inf')
                    arctan_approx = 0
                    
                    if ratio != float('inf'):
                        if ratio < 1.0:
                            arctan_approx = ratio * (1 - ratio * (1/3 - ratio * ratio * (1/5 - ratio * ratio / 7)))
                        else:
                            ratio = 1.0 / ratio
                            arctan_approx = 1.5708 - ratio * (1 - ratio * (1/3 - ratio * ratio * (1/5 - ratio * ratio / 7)))
                    else:
                        arctan_approx = 1.5708  # π/2
                    
                    # Sesuaikan nilai sudut berdasarkan kuadran
                    if dx > 0:  # Kuadran I dan IV
                        if dy >= 0:  # Kuadran I
                            angle_rad += arctan_approx
                        else:  # Kuadran IV
                            angle_rad -= arctan_approx
                    else:  # Kuadran II dan III
                        if dy >= 0:  # Kuadran II
                            angle_rad += arctan_approx
                        else:  # Kuadran III
                            angle_rad -= arctan_approx
                    
                    # Konversi ke derajat
                    angle = (angle_rad * 180) / 3.14159
                
                # Pastikan sudut berada di range [0, 360)
                angle = (angle + 360) % 360
                
                # Masukkan ke bin angular yang sesuai
                bin_idx = min(35, int(angle / 10))
                angular_bins[bin_idx] += 1
        
        # Deteksi puncak dalam distribusi angular (jari)
        peaks = 0
        threshold = max(angular_bins) * 0.2 if angular_bins else 0  # Minimal 20% dari nilai bin tertinggi
        
        for i in range(36):
            # Check if bin is a local maximum
            prev_bin = angular_bins[(i - 1) % 36]
            next_bin = angular_bins[(i + 1) % 36]
            
            if angular_bins[i] > threshold and angular_bins[i] > prev_bin and angular_bins[i] > next_bin:
                peaks += 1
        
        # Jumlah jari adalah jumlah puncak, dengan adjustment untuk estimasi lebih baik
        # Minimal 0 jari, maksimal 5 jari
        n_fingers = min(5, max(0, peaks))  # Sesuaikan berdasarkan pengamatan dalam dataset
        
        return n_fingers


if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process_dataset()