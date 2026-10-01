# Fraud Clustering & Klasifikasi Transaksi

Project machine learning untuk **segmentasi nasabah berdasarkan pola transaksi** (clustering dengan K-Means), kemudian hasil cluster dipakai sebagai label untuk **model klasifikasi** (Decision Tree). Dibuat sebagai Submission Akhir kelas Belajar Machine Learning untuk Pemula (BMLP).

Penulis: **Muhammad Rayhan**

## Dataset

Dataset berisi sekitar 2.500 data transaksi keuangan dengan atribut transaksi, demografi nasabah, dan pola penggunaan. Dataset dimuat otomatis dari Google Drive di dalam notebook (tidak perlu diunduh manual).

| Kolom | Keterangan |
|---|---|
| `TransactionID`, `AccountID` | ID transaksi dan akun |
| `TransactionAmount` | Nilai transaksi |
| `TransactionDate`, `PreviousTransactionDate` | Waktu transaksi dan transaksi sebelumnya |
| `TransactionType` | `Credit` atau `Debit` |
| `Location` | Kota lokasi transaksi |
| `DeviceID`, `IP Address`, `MerchantID` | Identitas perangkat, IP, dan merchant |
| `AccountBalance` | Saldo akun setelah transaksi |
| `Channel` | `Online`, `ATM`, atau `Branch` |
| `CustomerAge`, `CustomerOccupation` | Usia dan profesi nasabah |
| `TransactionDuration` | Durasi transaksi (detik) |
| `LoginAttempts` | Jumlah percobaan login sebelum transaksi |

## Struktur Project

```
fraud_clustering/
├── [Clustering]_Submission_Akhir_BMLP_muhammad_rayhan.ipynb    # Notebook clustering
├── [Klasifikasi]_Submission_Akhir_bmlp_muhammadrayhan.ipynb    # Notebook klasifikasi
├── data_clustering.csv        # Output clustering (dipakai notebook klasifikasi)
├── model_clustering.h5        # Model K-Means (joblib)
├── decision_tree_model.h5     # Model Decision Tree (joblib)
├── tuning_classification.h5   # Decision Tree hasil GridSearchCV (joblib)
└── README.md
```

> Sesuaikan nama file dengan isi folder Anda yang sebenarnya.

## Alur Kerja

### 1. Clustering (notebook Clustering)

1. **Load data & EDA dasar**: `head()`, `info()`, `describe()`.
2. **Pra-pemrosesan**
   - Cek `isnull()` dan `duplicated()`.
   - Imputasi nilai kosong (median untuk numerik, modus untuk kategorikal).
   - Scaling fitur numerik dengan `MinMaxScaler`.
   - Encoding fitur kategorikal dengan `LabelEncoder`.
   - Drop kolom ID: `TransactionID`, `AccountID`, `DeviceID`, `IP Address`, `MerchantID`.
   - Visualisasi boxplot untuk melihat outlier, serta binning usia (`AgeGroup`) dan nominal transaksi (`TransactionCategory`).
3. **Model K-Means**
   - Menentukan jumlah cluster dengan Elbow Method (`KElbowVisualizer`).
   - Melatih K-Means dengan **k = 3** (`random_state=42`).
   - Menyimpan model ke `model_clustering.h5`.
4. **Interpretasi cluster** dengan `groupby("cluster").agg(["min","max","mean"])`:
   - **Cluster 0**: pelanggan umum dengan nominal transaksi bervariasi dan saldo rata-rata paling rendah.
   - **Cluster 1**: pelanggan lama dan kurang aktif, jeda antar transaksi paling lama.
   - **Cluster 2**: pelanggan loyal dan bernilai tinggi, saldo dan nominal transaksi rata-rata tertinggi.
5. **Ekspor**: kolom `cluster` diganti nama menjadi `Target`, lalu disimpan ke `data_clustering.csv`.

### 2. Klasifikasi (notebook Klasifikasi)

1. Memuat `data_clustering.csv` hasil clustering.
2. Split data 80/20 dengan `stratify` (`random_state=42`): 508 data uji.
3. Melatih **Decision Tree** dan menyimpannya ke `decision_tree_model.h5`.
4. **Hyperparameter tuning** dengan `GridSearchCV` (cv=5, 108 kombinasi) pada `criterion`, `max_depth`, `min_samples_split`, dan `min_samples_leaf`.
5. Menyimpan model terbaik ke `tuning_classification.h5`.

## Hasil

| Model | Akurasi | Presisi | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (default) | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree (tuning) | 1.00 | 1.00 | 1.00 | 1.00 |

Parameter terbaik hasil tuning: `criterion='gini'`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`.

**Catatan interpretasi:** skor 100% wajar terjadi di sini karena label (`Target`) berasal dari K-Means yang dilatih pada fitur yang sama dengan yang dipakai klasifikasi. Decision Tree pada dasarnya hanya "mempelajari ulang" batas cluster, bukan memprediksi label yang independen. Jadi angka ini menunjukkan cluster mudah dipisahkan, bukan bukti model akan akurat pada data baru dengan label nyata (misalnya label fraud sebenarnya).

## Cara Menjalankan

1. Clone repository:
   ```bash
   git clone https://github.com/USERNAME/fraud-clustering.git
   cd fraud-clustering
   ```
2. Install dependensi:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn yellowbrick joblib jupyter
   ```
3. Jalankan notebook **Clustering** terlebih dahulu sampai selesai (menghasilkan `data_clustering.csv`).
4. Jalankan notebook **Klasifikasi**.
   - Notebook ini awalnya ditulis untuk Google Colab (`files.upload()`). Jika dijalankan lokal, ganti bagian upload dengan `pd.read_csv("data_clustering.csv")`.

Alternatif: buka kedua notebook langsung di Google Colab.

## Teknologi

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Yellowbrick, joblib.

## Lisensi

Project ini dibuat untuk keperluan pembelajaran.
