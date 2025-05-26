# Import library yang dibutuhkan
import numpy as np
import pandas as pd
from datetime import datetime
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, kernel='rbf', C=1.0, max_iter=1000, tol=1e-3, gamma='scale', degree=3):
        """
        Inisialisasi SVM dengan kernel
        
        Parameters:
        - kernel: tipe kernel ('linear', 'poly', 'rbf')
        - C: parameter regularisasi
        - max_iter: jumlah iterasi maksimum
        - tol: toleransi untuk kriteria penghentian
        - gamma: koefisien kernel untuk 'rbf' dan 'poly'
        - degree: derajat untuk kernel polynomial
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.degree = degree
        self.alphas = None
        self.b = None
        self.X = None
        self.y = None
        self.K = None
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        self.training_history = {
            'epoch': [],
            'num_support_vectors': [],
            'objective_value': []
        }
        print(f"\nInisialisasi SVM dengan kernel: {self.kernel}, C: {self.C}, gamma: {self.gamma}, degree: {self.degree}")
        print("=========================================")
        
    def _kernel_function(self, x1, x2):
        """Menghitung nilai kernel antara dua titik"""
        if self.gamma == 'scale' and not isinstance(self.gamma, float):
            self.gamma = 1.0 / (x1.shape[0] * np.var(x1))
            
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        else:
            raise ValueError(f"Tipe kernel tidak dikenal: {self.kernel}")

    def _compute_kernel_matrix(self, X):
        """Menghitung matriks kernel untuk data training"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        return K

    def _select_working_set(self, i):
        """Memilih variabel kedua untuk optimisasi"""
        valid_indices = list(range(len(self.y)))
        valid_indices.remove(i)
        return np.random.choice(valid_indices)

    def _compute_objective_value(self):
        """Menghitung nilai objective function"""
        n_samples = len(self.y)
        objective = 0.0
        
        # Hitung dual objective function
        for i in range(n_samples):
            for j in range(n_samples):
                objective += self.alphas[i] * self.alphas[j] * self.y[i] * self.y[j] * self.K[i, j]
        objective *= 0.5
        
        # Tambahkan linear term
        objective -= np.sum(self.alphas)
        
        return objective

    def fit(self, X, y):
        """
        Melatih SVM classifier menggunakan algoritma SMO
        
        Parameters:
        - X: array fitur [n_samples, n_features]
        - y: array label [-1, 1]
        """
        self.X = X
        self.y = y
        n_samples = X.shape[0]
        
        # Inisialisasi parameter
        self.alphas = np.zeros(n_samples)
        self.b = 0
        self.K = self._compute_kernel_matrix(X)
        
        # Reset training history
        self.training_history = {
            'epoch': [],
            'num_support_vectors': [],
            'objective_value': []
        }
        
        # Algoritma SMO
        iter_counter = 0
        while iter_counter < self.max_iter:
            num_changed_alphas = 0
            
            for i in range(n_samples):
                Ei = self._decision_function(X[i]) - y[i]
                
                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or
                    (y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    j = self._select_working_set(i)
                    Ej = self._decision_function(X[j]) - y[j]
                    
                    # Simpan alpha lama
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Hitung L dan H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alphas[j] = alpha_j_old - (y[j] * (Ei - Ej)) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < self.tol:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update threshold
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * self.K[i, j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * self.K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                        
                    num_changed_alphas += 1
            
            # Catat metrics untuk epoch ini
            sv_indices = np.where(self.alphas > self.tol)[0]
            self.training_history['epoch'].append(iter_counter)
            self.training_history['num_support_vectors'].append(len(sv_indices))
            self.training_history['objective_value'].append(self._compute_objective_value())
            
            if num_changed_alphas == 0:
                iter_counter += 1
            else:
                iter_counter = 0
                
        # Simpan support vectors
        sv_indices = np.where(self.alphas > self.tol)[0]
        self.support_vectors_ = X[sv_indices]
        self.support_vector_indices_ = sv_indices
        
        return self

    def _decision_function(self, x):
        """Menghitung fungsi keputusan untuk satu titik"""
        result = 0
        for i in range(len(self.y)):
            result += self.alphas[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return result + self.b

    def predict(self, X):
        """
        Melakukan prediksi untuk sampel dalam X
        
        Parameters:
        - X: array fitur [n_samples, n_features]
        
        Returns:
        - array prediksi label [-1, 1]
        """
        return np.array([np.sign(self._decision_function(x)) for x in X])

class SVMClassifier:
    def __init__(self):
        """Inisialisasi classifier dengan menyiapkan path dan parameter"""
        # Set base path
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_path = os.path.join(self.base_path, "processed_data")
        self.features_path = os.path.join(self.processed_path, "features.csv")
        
        # Buat folder untuk model
        self.models_path = os.path.join(self.processed_path, "models")
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
            
        # Parameter SVM
        self.kernel = 'rbf'
        self.C = 1.0
        self.gamma = 'scale'
        self.max_iter = 1000
        self.tol = 1e-3
        
        # Untuk menyimpan model dan scaler
        self.models = {}  # Untuk menyimpan model per kelas
        self.scaler = StandardScaler()
        self.training_histories = {}  # Untuk menyimpan history training per kelas
        
    def prepare_data(self):
        """Menyiapkan data untuk pelatihan"""
        print("\nMembaca dan menyiapkan data...")
        
        # Baca data
        df = pd.read_csv(self.features_path)
        
        # Pisahkan fitur dan label
        X = df[['n_defects', 'area_ratio', 'aspect_ratio']].values
        y = df['label'].values
        
        # Normalisasi fitur
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train(self):
        """Melatih model SVM untuk setiap kelas"""
        print("\nMemulai pelatihan model...")
        
        # Siapkan data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Untuk setiap kelas (0-5), latih model one-vs-all
        for label in range(6):
            print(f"\nMelatih model untuk kelas {label}...")
            
            # Buat label biner (-1 untuk bukan kelas ini, 1 untuk kelas ini)
            y_binary = np.where(y_train == label, 1, -1)
            
            # Buat dan latih model
            model = SVM(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                max_iter=self.max_iter,
                tol=self.tol
            )
            
            # Latih model
            model.fit(X_train, y_binary)
            
            # Simpan model dan history training
            self.models[label] = model
            self.training_histories[label] = model.training_history
            
            # Tampilkan jumlah support vectors
            n_sv = len(model.support_vectors_)
            print(f"Jumlah support vectors untuk kelas {label}: {n_sv}")
            
        # Evaluasi model
        self.evaluate(X_test, y_test)
        
        # Export training history
        self.export_training_history()
        
        # Simpan model dan scaler
        self.save_models()
        
    def export_training_history(self):
        """Export training history untuk visualisasi"""
        history_path = os.path.join(self.processed_path, 'visualisasi')
        if not os.path.exists(history_path):
            os.makedirs(history_path)
            
        # Plot training history untuk setiap kelas
        plt.figure(figsize=(15, 5))
        
        # Plot jumlah support vectors
        plt.subplot(1, 2, 1)
        for label, history in self.training_histories.items():
            plt.plot(history['epoch'], history['num_support_vectors'], 
                    label=f'Kelas {label}')
        plt.title('Jumlah Support Vectors per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Jumlah Support Vectors')
        plt.legend()
        
        # Plot nilai objective function
        plt.subplot(1, 2, 2)
        for label, history in self.training_histories.items():
            plt.plot(history['epoch'], history['objective_value'], 
                    label=f'Kelas {label}')
        plt.title('Nilai Objective Function per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(history_path, 'training_history.png'))
        plt.close()
        
        # Export data ke CSV
        history_data = []
        for label, history in self.training_histories.items():
            for i in range(len(history['epoch'])):
                history_data.append({
                    'kelas': label,
                    'epoch': history['epoch'][i],
                    'num_support_vectors': history['num_support_vectors'][i],
                    'objective_value': history['objective_value'][i]
                })
        
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(os.path.join(history_path, 'training_history.csv'), index=False)
        print(f"\nTraining history disimpan di: {history_path}")
        
    def predict(self, X):
        """Melakukan prediksi menggunakan semua model"""
        # Normalisasi input
        X_scaled = self.scaler.transform(X)
        
        # Tampung hasil dari setiap model
        scores = np.zeros((X.shape[0], len(self.models)))
        
        # Prediksi dengan setiap model
        for label, model in self.models.items():
            scores[:, label] = np.array([model._decision_function(x) for x in X_scaled])
            
        # Pilih kelas dengan skor tertinggi
        return np.argmax(scores, axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluasi performa model"""
        print("\nEvaluasi model...")
        
        # Lakukan prediksi
        y_pred = self.predict(X_test)
        
        # Hitung akurasi
        accuracy = np.mean(y_pred == y_test)
        print(f"Akurasi: {accuracy:.4f}")
        
        # Hitung confusion matrix
        confusion_matrix = np.zeros((6, 6), dtype=int)
        for i in range(len(y_test)):
            confusion_matrix[y_test[i]][y_pred[i]] += 1
            
        # Tampilkan confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Tambahkan label
        plt.xticks(np.arange(6))
        plt.yticks(np.arange(6))
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        
        # Tambahkan nilai di setiap sel
        for i in range(6):
            for j in range(6):
                plt.text(j, i, str(confusion_matrix[i, j]),
                        ha="center", va="center")
        
        # Simpan plot
        plt.savefig(os.path.join(self.processed_path, 'confusion_matrix.png'))
        plt.close()
        
        # Tambahan: Plot Support Vectors untuk visualisasi 2D
        if X_test.shape[1] == 2:  # Hanya jika data 2D
            plt.figure(figsize=(12, 8))
            colors = ['r', 'g', 'b', 'y', 'c', 'm']
            
            for label, model in self.models.items():
                # Plot support vectors untuk setiap kelas
                if model.support_vectors_ is not None:
                    plt.scatter(
                        model.support_vectors_[:, 0],
                        model.support_vectors_[:, 1],
                        c=colors[label],
                        marker='o',
                        label=f'SV Class {label}',
                        alpha=0.5
                    )
                    
            plt.title('Support Vectors Visualization')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.savefig(os.path.join(self.processed_path, 'support_vectors.png'))
            plt.close()
        
    def save_models(self):
        """Menyimpan model dan scaler"""
        print("\nMenyimpan model...")
        
        # Simpan setiap model
        for label, model in self.models.items():
            model_path = os.path.join(self.models_path, f'svm_model_{label}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Simpan scaler
        scaler_path = os.path.join(self.models_path, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
        print(f"Model disimpan di: {self.models_path}")
    
    def load_models(self):
        """Memuat model yang sudah disimpan"""
        print("\nMemuat model...")
        
        # Muat setiap model
        for label in range(6):
            model_path = os.path.join(self.models_path, f'svm_model_{label}.pkl')
            with open(model_path, 'rb') as f:
                self.models[label] = pickle.load(f)
        
        # Muat scaler
        scaler_path = os.path.join(self.models_path, 'scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        print("Model berhasil dimuat!")

if __name__ == "__main__":
    # Buat dan latih classifier
    classifier = SVMClassifier()
    classifier.train()
