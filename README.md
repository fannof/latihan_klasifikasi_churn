# Latihan Studi Kasus: Klasifikasi Pelanggan untuk Churn pada Perusahaan XYZ (Dicoding Platform)

## Langkah 1: Mengimpor Library

Pada langkah pertama, berbagai library yang diperlukan untuk menganalisis data dan membangun model machine learning akan diimpor. Berikut adalah penjelasan singkat dari masing-masing library yang diimpor.

- Pandas

  Library ini digunakan untuk memanipulasi dan menganalisis data dalam bentuk DataFrame. Library ini sangat berguna untuk membaca, menulis, dan mengelola data dalam berbagai format.
  
- Numpy

  Library ini menyediakan dukungan untuk array dan operasi matematika tingkat tinggi. numpy sangat penting untuk perhitungan numerik yang efisien dalam machine learning.
  
- Seaborn

  Seaborn adalah library untuk visualisasi data yang dibangun di atas matplotlib. Digunakan untuk membuat plot statistik yang lebih kompleks dan informatif.
  
- matplotlib.pyplot

  Library dasar untuk membuat visualisasi data, seperti grafik dan plot. Sering digunakan bersama seaborn untuk memperindah visualisasi.
  
- sklearn.model_selection.train_test_split

  Fungsi ini digunakan untuk membagi dataset menjadi data latih dan data uji, penting agar model yang dibangun dapat dievaluasi dengan data yang belum pernah dilihat sebelumnya.
  
- sklearn.preprocessing.LabelEncoder

  Fungsi ini digunakan untuk mengubah data kategori menjadi bentuk numerik yang dapat diproses oleh algoritma machine learning.
  
- sklearn.preprocessing.StandardScaler

  Fungsi ini digunakan untuk menstandardisasi fitur dengan menghilangkan rata-rata dan menyesuaikan skala ke unit varians. Ini sering diterapkan pada data sebelum dimasukkan ke model.
  
- sklearn.preprocessing.MinMaxScaler

  Fungsi ini digunakan untuk mengubah fitur dengan membuat skala dari setiap fitur setiap fitur ke rentang yang ditentukan (biasanya antara 0 dan 1).
  
- sklearn.neighbors.KNeighborsClassifier

  Fungsi ini merupakan algoritma K-Nearest Neighbors (KNN) yang digunakan untuk klasifikasi data berdasarkan kemiripan dengan tetangga terdekat.
  
- sklearn.tree.DecisionTreeClassifier

  Algoritma Decision Tree digunakan untuk membuat model klasifikasi atau regresi dalam bentuk struktur pohon keputusan.
  
- sklearn.ensemble.RandomForestClassifier

  Algoritma ini merupakan ensemble learning method yang menggabungkan beberapa Decision Tree untuk meningkatkan performa prediksi.
  
- sklearn.svm.SVC

  Support Vector Machine digunakan untuk klasifikasi dengan mencari hyperplane yang memisahkan kelas-kelas data dengan margin terbesar.
  
- sklearn.naive_bayes.GaussianNB

  Algoritma Naive Bayes digunakan untuk klasifikasi berdasarkan prinsip Teorema Bayes dengan asumsi sederhana bahwa fitur bersifat independen.
  
- sklearn.metrics

  Library ini menyediakan berbagai fungsi untuk mengevaluasi performa model, seperti confusion_matrix, accuracy_score, precision_score, recall_score, dan f1_score.

## Langkah 2: Memuat Data

Pada langkah ini, data yang akan dianalisis diimpor dari Google Drive. Dimulai dengan menentukan ID unik file yang diunggah ke Google Drive, ID ini digunakan untuk membuat URL unduhan langsung yang memungkinkan akses file CSV melalui kode Python. Setelah URL terbentuk, file CSV dibaca dalam DataFrame menggunakan pustaka pandas, yang memungkinkan data disimpan dalam bentuk tabel dua dimensi. 

Langkah awal dalam pengolahan data melibatkan pemeriksaan dan pembersihan dataset. Pertama, dilakukan peninjauan terhadap informasi umum dataset menggunakan fungsi data.info() untuk memahami jumlah entri, tipe data, serta memastikan tidak ada fitur yang memiliki data yang hilang. Selanjutnya, pengecekan nilai yang hilang dilakukan pada setiap fitur menggunakan perintah data.isnull().sum() untuk mengidentifikasi bahwa ada fitur yang memerlukan penanganan khusus akibat kekurangan data. 

![latihan1](https://github.com/user-attachments/assets/2106e13c-4e57-4b6b-a75d-959612ce1f58)

Berikut adalah penjelasan singkat setiap fitur dalam dataset.

- RowNumber

  Nomor baris dalam dataset yang digunakan untuk identifikasi unik setiap entri. Fitur ini tidak memiliki makna analitis.
  
- CustomerId

  ID unik yang mengidentifikasi setiap pelanggan dalam sistem. Ini berguna untuk referensi dan penggabungan data.

- Surname

  Nama belakang pelanggan. Fitur ini tidak digunakan dalam analisis model karena tidak relevan.

- CreditScore 

  Skor kredit yang menunjukkan kelayakan kredit pelanggan. Skor ini dapat memengaruhi keputusan mereka untuk tetap atau berhenti menggunakan layanan.

- Geography

  Lokasi geografis tempat tinggal pelanggan. Informasi ini dapat memengaruhi perilaku dan kebutuhan layanan pelanggan.

- Gender

  Jenis kelamin pelanggan. Meskipun tidak selalu memengaruhi churn secara langsung, informasi ini berguna untuk analisis demografis.

- Age

  Usia pelanggan. Usia dapat memengaruhi kebiasaan dan preferensi dalam menggunakan layanan.

- Tenure

  Lama berlangganan pelanggan. Durasi berlangganan ini sering kali berhubungan dengan kemungkinan pelanggan untuk churn.

- Balance

  Saldo rekening pelanggan. Saldo ini dapat memengaruhi kepuasan pelanggan dan kecenderungan mereka untuk tetap menggunakan layanan.

- NumOfProducts

  Jumlah produk yang dimiliki pelanggan. Fitur ini membantu memahami keterlibatan pelanggan dengan berbagai produk.

- HasCrCard

  Ini menunjukkan pelanggan memiliki kartu kredit atau tidak. Fitur ini dapat memengaruhi pengalaman pelanggan dengan layanan.

- IsActiveMember

  Status keanggotaan aktif pelanggan. Ini menunjukkan pelanggan masih aktif atau tidak dalam menggunakan layanan.

- EstimatedSalary

  Gaji yang diperkirakan dari pelanggan. Gaji dapat memengaruhi keputusan pelanggan untuk berlangganan atau berhenti dari layanan.

- Exited

  Label target yang menunjukkan pelanggan telah keluar dari layanan (1) atau tidak (0). Fitur ini merupakan variabel yang ingin diprediksi dalam model klasifikasi.

## Langkah 3: Exploratory Data Analysis (EDA) 
  
Exploratory Data Analysis (EDA) adalah tahap krusial dalam proses analisis data untuk memahami karakteristik, pola, dan hubungan antar fitur dalam dataset. Dalam tahap ini, distribusi fitur numerik pertama-tama dianalisis. Setiap fitur numerik divisualisasikan menggunakan histogram yang menunjukkan distribusi nilai-nilai dalam fitur tersebut. Histogram ini dilengkapi dengan kurva densitas untuk memberikan gambaran lebih jelas tentang pola distribusi data: apakah data terdistribusi normal atau mengalami skewness?. Berikut adalah distribusi fitur numerik pada dataset ini.

![latihan2](https://github.com/user-attachments/assets/47ddc80d-f2dc-4519-9b73-fb5e523c9e0e)

Setiap fitur numerik dianalisis melalui histogram yang menunjukkan sebaran nilai-nilai dalam fitur tersebut. Grafik-grafik ini membantu dalam memahami rentang nilai, kecenderungan pusat, serta potensi outliers pada setiap fitur numerik. Dengan memeriksa distribusi ini, wawasan penting mengenai karakteristik data yang akan memengaruhi pemodelan dan analisis selanjutnya dapat diperoleh.

Selanjutnya, distribusi fitur kategorikal diperiksa dengan menggunakan grafik batang horizontal. Grafik ini memperlihatkan frekuensi setiap kategori dalam fitur kategorikal, membantu untuk memahami seberapa sering masing-masing kategori muncul dalam dataset. Visualisasi ini berguna untuk mengidentifikasi kategori yang dominan atau langka, serta memberikan wawasan tentang sebaran data dalam kategori. Melalui EDA yang menyeluruh, pemahaman tentang dataset menjadi lebih mendalam, memfasilitasi pengambilan keputusan yang lebih baik dalam tahap pemodelan dan analisis lebih lanjut.

![latihan3](https://github.com/user-attachments/assets/e725421a-a928-4e8d-ab4f-ef45cb76c00a)

