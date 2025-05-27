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
        
        # Hitung standard deviation
        sum_squared_diff = sum((d - mean_distance)**2 for d in distances)
        std_dev = (sum_squared_diff / len(distances))**0.5
        
        # Hitung densitas (proporsi piksel putih terhadap total area)
        total_area = height * width
        density = n_pixels / total_area if total_area > 0 else 0
        
        return mean_distance, std_dev, density
    
    def create_visual_contour(self, points):
        """Buat representasi kontur untuk visualisasi"""
        if not points:
            return None
            
        # Konversi format poin untuk visualisasi
        contour = []
        for x, y in points:
            contour.append([[x, y]])
            
        return contour
        
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
                    angle = (180 / 3.14159) * (3.14159 + 3.14159/2 - 
                             abs(3.14159/2 - abs(3.14159/2 - 
                                 (3.14159 + (0 if x - center_x > 0 else 3.14159) + 
                                  (0 if y - center_y > 0 else 0) - 
                                  (3.14159 if y - center_y > 0 else 0) + 
                                  abs(0 if y - center_y == 0 else ((y - center_y) / abs(y - center_y))) * 
                                  (3.14159/2 - abs(3.14159/2 - 
                                     3.14159/2 * abs((x - center_x) / 
                                                     (abs(y - center_y) if abs(y - center_y) > 0 else 1))))))))
                
                # Masukkan ke bin angular yang sesuai
                bin_idx = min(35, int(angle / 10))
                angular_bins[bin_idx] += 1
        
        # Deteksi puncak dalam distribusi angular (jari)
        peaks = 0
        threshold = max(angular_bins) * 0.2  # Minimal 20% dari nilai bin tertinggi
        
        for i in range(36):
            # Check if bin is a local maximum
            prev_bin = angular_bins[(i - 1) % 36]
            next_bin = angular_bins[(i + 1) % 36]
            
            if angular_bins[i] > threshold and angular_bins[i] > prev_bin and angular_bins[i] > next_bin:
                peaks += 1
        
        # Jumlah jari adalah jumlah puncak, dengan adjustment untuk estimasi lebih baik
        # Minimal 0 jari, maksimal 5 jari
        n_fingers = min(5, max(0, peaks - 1))  # -1 karena palm juga bisa dihitung sebagai puncak
        
        return n_fingers
