# Import library yang dibutuhkan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import numpy as np
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D  # Ini penting untuk plot 3D

class DatasetVisualizer:
    def __init__(self):
        # Set path dasar
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = os.path.join(self.base_path, "dataset")
        self.processed_path = os.path.join(self.base_path, "processed_data")
        self.features_path = os.path.join(self.processed_path, "features.csv")
        
        # Buat folder untuk menyimpan grafik
        self.plots_path = os.path.join(self.processed_path, "visualisasi")
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)
        
        # Set style untuk plot
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def hitung_statistik_dataset(self):
        """Menghitung statistik dasar dataset"""
        print("\nMenghitung statistik dataset...")
        statistik = {}
        
        # Hitung jumlah gambar per label
        for i in range(6):
            folder = os.path.join(self.dataset_path, str(i))
            if os.path.exists(folder):
                jumlah = len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
                statistik[f'Label {i}'] = jumlah
        
        return statistik
    
    def plot_distribusi_dataset(self, statistik):
        """Membuat plot distribusi dataset"""
        print("Membuat plot distribusi dataset...")
        plt.figure(figsize=(10, 6))
        bars = plt.bar(statistik.keys(), statistik.values())
        plt.title('Distribusi Jumlah Gambar per Label')
        plt.xlabel('Label Gesture')
        plt.ylabel('Jumlah Gambar')
        
        # Tambah nilai di atas bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.savefig(os.path.join(self.plots_path, 'distribusi_dataset.png'))
        plt.close()
    
    def plot_fitur_histogram(self):
        """Membuat histogram untuk setiap fitur"""
        print("Membuat histogram fitur...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        features = ['n_defects', 'area_ratio', 'aspect_ratio']
        
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=feature, hue='label', multiple="stack")
            plt.title(f'Distribusi {feature} per Label')
            plt.xlabel(feature)
            plt.ylabel('Jumlah')
            plt.savefig(os.path.join(self.plots_path, f'histogram_{feature}.png'))
            plt.close()
    
    def plot_korelasi_fitur(self):
        """Membuat plot korelasi antar fitur"""
        print("Membuat plot korelasi fitur...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        features = ['n_defects', 'area_ratio', 'aspect_ratio']
        
        # Matrix korelasi
        corr = df[features].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title('Korelasi antar Fitur')
        plt.savefig(os.path.join(self.plots_path, 'korelasi_fitur.png'))
        plt.close()
    
    def plot_scatter_fitur(self):
        """Membuat scatter plot untuk kombinasi fitur"""
        print("Membuat scatter plot fitur...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        features = ['n_defects', 'area_ratio', 'aspect_ratio']
        
        # Plot semua kombinasi fitur
        fig = plt.figure(figsize=(15, 5))
        
        # n_defects vs area_ratio
        plt.subplot(131)
        sns.scatterplot(data=df, x='n_defects', y='area_ratio', hue='label')
        plt.title('n_defects vs area_ratio')
        
        # n_defects vs aspect_ratio
        plt.subplot(132)
        sns.scatterplot(data=df, x='n_defects', y='aspect_ratio', hue='label')
        plt.title('n_defects vs aspect_ratio')
        
        # area_ratio vs aspect_ratio
        plt.subplot(133)
        sns.scatterplot(data=df, x='area_ratio', y='aspect_ratio', hue='label')
        plt.title('area_ratio vs aspect_ratio')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'scatter_fitur.png'))
        plt.close()
    
    def plot_boxplot_fitur(self):
        """Membuat box plot untuk setiap fitur berdasarkan label"""
        print("Membuat box plot fitur...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        features = ['n_defects', 'area_ratio', 'aspect_ratio']
        
        plt.figure(figsize=(15, 5))
        for i, feature in enumerate(features, 1):
            plt.subplot(1, 3, i)
            sns.boxplot(data=df, x='label', y=feature)
            plt.title(f'Box Plot {feature} per Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'boxplot_fitur.png'))
        plt.close()
    
    def plot_preprocessing_steps(self):
        """Membuat visualisasi tahapan preprocessing untuk satu sampel dari setiap label"""
        print("Membuat visualisasi tahapan preprocessing...")
        for label in range(6):
            folder = os.path.join(self.dataset_path, str(label))
            if not os.path.exists(folder):
                continue
                
            # Ambil satu gambar sampel
            files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
            if not files:
                continue
                
            # Proses gambar sampel
            sample_path = os.path.join(folder, files[0])
            img = cv2.imread(sample_path)
            
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, (224, 224))
            
            # Preprocessing steps
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Gabungkan semua tahap
            steps = [img, gray, blur, thresh]
            titles = ['Original', 'Grayscale', 'Blur', 'Threshold']
            
            # Plot
            fig, axes = plt.subplots(1, len(steps), figsize=(15, 3))
            fig.suptitle(f'Tahapan Preprocessing - Label {label}')
            
            for ax, step, title in zip(axes, steps, titles):
                if len(step.shape) == 2:
                    ax.imshow(step, cmap='gray')
                else:
                    ax.imshow(cv2.cvtColor(step, cv2.COLOR_BGR2RGB))
                ax.set_title(title)
                ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_path, f'preprocessing_label_{label}.png'))
            plt.close()
    
    def plot_training_history(self):
        """Membuat visualisasi history training dari model SVM"""
        print("Membuat visualisasi history training...")
        history_path = os.path.join(self.processed_path, 'visualisasi', 'training_history.csv')
        
        if not os.path.exists(history_path):
            print("File training_history.csv tidak ditemukan!")
            return
            
        # Baca data history
        history_df = pd.read_csv(history_path)
        
        # Plot jumlah support vectors per epoch untuk setiap kelas
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        for kelas in history_df['kelas'].unique():
            kelas_data = history_df[history_df['kelas'] == kelas]
            plt.plot(kelas_data['epoch'], kelas_data['num_support_vectors'], 
                    label=f'Kelas {kelas}')
        
        plt.title('Jumlah Support Vectors per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Jumlah Support Vectors')
        plt.legend()
        
        # Plot nilai objective function per epoch untuk setiap kelas
        plt.subplot(1, 2, 2)
        for kelas in history_df['kelas'].unique():
            kelas_data = history_df[history_df['kelas'] == kelas]
            plt.plot(kelas_data['epoch'], kelas_data['objective_value'], 
                    label=f'Kelas {kelas}')
        
        plt.title('Nilai Objective Function per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Objective Value')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'training_history_plot.png'))
        plt.close()
        
    def plot_feature_distribution(self):
        """Membuat visualisasi distribusi fitur berdasarkan label"""
        print("Membuat visualisasi distribusi fitur...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        features = ['n_defects', 'area_ratio', 'aspect_ratio']
        
        # Violin plot untuk melihat distribusi
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(len(features), 1, i)
            sns.violinplot(data=df, x='label', y=feature)
            plt.title(f'Distribusi {feature} per Label')
            plt.xlabel('Label Gesture')
            plt.ylabel(feature)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'feature_distribution_violin.png'))
        plt.close()
        
        # KDE plot untuk distribusi secara umum
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(features, 1):
            plt.subplot(len(features), 1, i)
            for label in sorted(df['label'].unique()):
                subset = df[df['label'] == label]
                sns.kdeplot(subset[feature], label=f'Label {label}')
            plt.title(f'Distribusi Densitas {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'feature_distribution_kde.png'))
        plt.close()
        
    def plot_feature_space(self):
        """Membuat visualisasi ruang fitur dalam 3D"""
        print("Membuat visualisasi ruang fitur 3D...")
        if not os.path.exists(self.features_path):
            print("File features.csv tidak ditemukan!")
            return
            
        df = pd.read_csv(self.features_path)
        
        # Plot 3D scatter untuk semua fitur
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['r', 'g', 'b', 'y', 'c', 'm']
        labels = sorted(df['label'].unique())
        
        for label, color in zip(labels, colors):
            subset = df[df['label'] == label]
            ax.scatter(
                subset['n_defects'], 
                subset['area_ratio'], 
                subset['aspect_ratio'],
                c=color,
                label=f'Label {label}',
                alpha=0.7
            )
        
        ax.set_xlabel('n_defects')
        ax.set_ylabel('area_ratio')
        ax.set_zlabel('aspect_ratio')
        ax.set_title('Visualisasi Ruang Fitur 3D')
        ax.legend()
        
        plt.savefig(os.path.join(self.plots_path, 'feature_space_3d.png'))
        plt.close()
        
        # Membuat pairplot untuk melihat semua fitur dalam 2D
        plt.figure(figsize=(12, 10))
        sns.pairplot(df, hue='label', vars=df[['n_defects', 'area_ratio', 'aspect_ratio']])
        plt.savefig(os.path.join(self.plots_path, 'feature_space_pairplot.png'))
        plt.close()
    
    def run(self):
        """Jalankan semua visualisasi"""
        print("\n=== Memulai Visualisasi Dataset ===")
        
        # Buat semua visualisasi
        statistik = self.hitung_statistik_dataset()
        self.plot_distribusi_dataset(statistik)
        self.plot_fitur_histogram()
        self.plot_korelasi_fitur()
        self.plot_scatter_fitur()
        self.plot_boxplot_fitur()
        self.plot_preprocessing_steps()
        
        # Tambahan visualisasi baru
        self.plot_feature_distribution()
        self.plot_feature_space()
        self.plot_training_history()
        
        print("\nVisualisasi selesai!")
        print(f"Hasil visualisasi disimpan di: {self.plots_path}")

if __name__ == "__main__":
    visualizer = DatasetVisualizer()
    visualizer.run()
