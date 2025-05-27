"""
Modul untuk mengganti fungsi save_debug_image yang bermasalah
"""

import cv2
import numpy as np
import os

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
        
        # Tambahkan informasi label dan fitur
        feature_names = ['n_fingers', 'aspect_ratio', 'area_ratio', 'mean_dist', 'std_dev', 'density']
        feature_info = np.zeros((200, self.image_size[1]*2, 3), dtype=np.uint8)
        
        # Tambahkan judul
        cv2.putText(feature_info, f'Label: {label}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Tambahkan fitur dengan deskripsi
        for i, (name, feat) in enumerate(zip(feature_names, features)):
            cv2.putText(feature_info, f'{name}: {feat:.3f}', (10, 70 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Gabungkan gambar dengan informasi fitur
        feature_info_resized = cv2.resize(feature_info, (top_row.shape[1], 200))
        final_image = np.vstack((top_row, feature_info_resized))
        
        # Simpan gambar hasil
        output_filename = os.path.join(self.preprocess_path, f'debug_{filename}')
        cv2.imwrite(output_filename, final_image)
        
    except Exception as e:
        print(f"Error dalam menyimpan gambar debug: {str(e)}")
        import traceback
        traceback.print_exc()  # Debug lebih detail
