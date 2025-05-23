# Latihan Studi Kasus: Klasifikasi Pelanggan untuk Churn pada Perusahaan XYZ

### Platform : Dicoding

### Kelas : Belajar Machine Learning untuk Pemula

### Modul : Supervised Learning - Klasifikasi

### Dataset : [Churn Dataset](https://drive.google.com/uc?id=19IfOP0QmCHccMu8A6B2fCUpFqZwCxuzO)

## Mengimpor Library

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

## Memuat Data

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

## Exploratory Data Analysis (EDA) 
  
Exploratory Data Analysis (EDA) adalah tahap krusial dalam proses analisis data untuk memahami karakteristik, pola, dan hubungan antar fitur dalam dataset. Dalam tahap ini, distribusi fitur numerik pertama-tama dianalisis. Setiap fitur numerik divisualisasikan menggunakan histogram yang menunjukkan distribusi nilai-nilai dalam fitur tersebut. Histogram ini dilengkapi dengan kurva densitas untuk memberikan gambaran lebih jelas tentang pola distribusi data: apakah data terdistribusi normal atau mengalami skewness?. Berikut adalah distribusi fitur numerik pada dataset ini.

![latihan2](https://github.com/user-attachments/assets/47ddc80d-f2dc-4519-9b73-fb5e523c9e0e)

Setiap fitur numerik dianalisis melalui histogram yang menunjukkan sebaran nilai-nilai dalam fitur tersebut. Grafik-grafik ini membantu dalam memahami rentang nilai, kecenderungan pusat, serta potensi outliers pada setiap fitur numerik. Dengan memeriksa distribusi ini, wawasan penting mengenai karakteristik data yang akan memengaruhi pemodelan dan analisis selanjutnya dapat diperoleh.

Selanjutnya, distribusi fitur kategorikal diperiksa dengan menggunakan grafik batang horizontal. Grafik ini memperlihatkan frekuensi setiap kategori dalam fitur kategorikal, membantu untuk memahami seberapa sering masing-masing kategori muncul dalam dataset. Visualisasi ini berguna untuk mengidentifikasi kategori yang dominan atau langka, serta memberikan wawasan tentang sebaran data dalam kategori. Melalui EDA yang menyeluruh, pemahaman tentang dataset menjadi lebih mendalam, memfasilitasi pengambilan keputusan yang lebih baik dalam tahap pemodelan dan analisis lebih lanjut.

![latihan3](https://github.com/user-attachments/assets/e725421a-a928-4e8d-ab4f-ef45cb76c00a)

Langkah berikutnya menghasilkan heatmap korelasi yang memvisualisasikan hubungan antar fitur numerik dalam dataset. Dengan ukuran gambar 12 × 10 inci, heatmap ini menggunakan skema warna 'coolwarm' untuk menampilkan kekuatan korelasi antara fitur-fitur, dari korelasi negatif (biru) hingga positif (merah). Nilai korelasi ditampilkan pada setiap sel dengan format dua desimal, sementara garis pembatas yang tipis antara sel memudahkan pembacaan. 

![latihan4](https://github.com/user-attachments/assets/e924f606-06b4-4346-bde2-2e19cc614d6f)

Pairplot ini menyajikan grafik scatter plot untuk setiap pasangan fitur numerik yang memungkinkan visualisasi hubungan antara fitur-fitur tersebut. Selain itu, diagonal pairplot menampilkan histogram dari distribusi masing-masing fitur. Pairplot membantu dalam mengidentifikasi pola, korelasi, dan distribusi di antara fitur-fitur numerik, serta mendeteksi potensi outlier atau hubungan non-linier yang mungkin ada. Berikut hasilnya.

![latihan5](https://github.com/user-attachments/assets/f518a735-b1a2-4576-9ed5-7cdcbecf57e1)

Warna pada grafik ditetapkan menggunakan palet 'viridis' untuk memperjelas perbedaan antar kelas. Visualisasi ini membantu memahami seberapa seimbang atau tidak seimbang distribusi kelas target dalam dataset.

![latihan6](https://github.com/user-attachments/assets/b354c8be-2c4e-4cf2-886d-e9c2293c0e2d)

Pada bagian ini, terlihat adanya ketidakseimbangan antara jumlah pelanggan yang keluar (churn) dan yang tidak keluar dalam dataset. Ketidakseimbangan ini sering kali dapat memengaruhi performa model klasifikasi. 

## Label Encoder

Pada langkah ini, encoding diterapkan pada fitur kategorikal dalam dataset untuk mempersiapkan data bagi algoritma pembelajaran mesin. LabelEncoder digunakan untuk mengonversi nilai kategorikal menjadi format numerik yang dapat diproses oleh model. Kolom-kolom kategorikal, seperti 'Geography' dan 'Gender' diubah menjadi angka dengan menerapkan LabelEncoder. Setelah proses encoding, DataFrame ditampilkan kembali untuk memastikan bahwa perubahan telah diterapkan dengan benar. Berikut adalah hasil yang didapatkan.

![latihan7](https://github.com/user-attachments/assets/060407e5-0f66-47d1-95d0-5ba026eb6f95)

## Data Splitting

Pada langkah ini, data numerik dinormalisasi menggunakan MinMaxScaler untuk memastikan bahwa semua fitur numerik berada dalam rentang yang sama, yang dapat meningkatkan performa model. Setelah normalisasi, data dibagi menjadi fitur (X) dan target (y). Data kemudian dipisahkan menjadi set pelatihan dan set uji menggunakan train_test_split dengan 20% data digunakan untuk uji dan 80% untuk pelatihan. Bentuk dari set pelatihan dan set uji ditampilkan untuk memastikan bahwa pemisahan telah dilakukan dengan benar.

## Pelatihan Model 

Pada langkah ini, setiap algoritma klasifikasi dilatih secara terpisah dengan menggunakan data pelatihan. Model KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, SVC, dan GaussianNB dipersiapkan serta dilatih. Setelah proses pelatihan selesai, model-model ini siap untuk diuji dengan data uji. Pesan "Model training selesai." menandakan bahwa semua model sudah berhasil dilatih.

## Evaluasi Model

Pada langkah ini, setiap model dievaluasi untuk mengukur kinerjanya. Fungsi evaluate_model digunakan untuk menghitung berbagai metrik performa, seperti matriks kebingungannya (confusion matrix), serta skor akurasi, presisi, recall, dan F1-Score. Hasil evaluasi dari setiap model—yaitu K-Nearest Neighbors (KNN), Decision Tree (DT), Random Forest (RF), Support Vector Machine (SVM), dan Naive Bayes (NB)—dikumpulkan dalam sebuah DataFrame yang merangkum semua metrik penting tersebut. DataFrame ini kemudian ditampilkan untuk memberikan gambaran jelas mengenai kinerja masing-masing model.

## Rangkuman Hasil dan Analisis 

Pada tahap ini, hasil dari evaluasi berbagai model klasifikasi yang telah diterapkan pada dataset pelanggan dirangkum untuk memahami kinerja masing-masing model. Evaluasi ini mencakup metrik-metrik penting, seperti akurasi, presisi, recall, dan F1-Score, yang memberikan gambaran menyeluruh tentang kemampuan model dalam memprediksi pelanggan berpotensi churn. Dengan menganalisis metrik-metrik tersebut, dapat diketahui model yang paling efektif dalam mengidentifikasi pelanggan yang akan berhenti berlangganan, serta kekuatan dan kelemahan masing-masing model.

- Hasil Confusion Matrix Algoritma KNN

  Model K-Nearest Neighbors (KNN) menunjukkan hasil evaluasi sebagai berikut. Matriks kebingungan mengungkapkan bahwa model berhasil mengidentifikasi 128 pelanggan yang sebenarnya churn (true positive) dengan benar, sementara 87 pelanggan yang sebenarnya tidak churn teridentifikasi sebagai churn (false positive).

  ![latihan8](https://github.com/user-attachments/assets/7a6b03eb-1781-4320-ba3f-6969593bb331)

  Sebaliknya, ada 265 pelanggan yang sebenarnya churn, tetapi tidak terdeteksi oleh model (false negative) dan 1520 pelanggan yang benar-benar tidak churn dan diprediksi dengan benar (true negative). Hasil ini memberikan gambaran tentang kemampuan model KNN dalam mengklasifikasikan pelanggan dengan tepat.

- Hasil Confusion Matrix Algoritma Decision Tree

  Untuk Decision Tree Classifier, hasil evaluasi menunjukkan distribusi prediksi sebagai berikut: ada 1360 true negative (TN) yang berarti pelanggan tidak churn terdeteksi dengan benar. Sebanyak 247 false positive (FP) menunjukkan bahwa pelanggan yang tidak churn salah diklasifikasikan sebagai churn.
  
  ![latihan9](https://github.com/user-attachments/assets/8eed64c2-0b1c-4ca9-a624-61e89bf12808)

  Selain itu, 193 false negative (FN) menggambarkan pelanggan yang sebenarnya churn, tetapi tidak teridentifikasi oleh model. Akhirnya, model berhasil mendeteksi 200 true positive (TP), yaitu pelanggan yang benar-benar churn. Analisis ini memberikan wawasan tentang kinerja model dalam memprediksi churn dan area yang perlu diperbaiki.
  
- Hasil Confusion Matrix Algoritma Random Forest

  Untuk Random Forest Classifier, hasil evaluasi menunjukkan distribusi prediksi sebagai berikut: ada 1557 true negative (TN) yang berarti pelanggan tidak churn terdeteksi dengan benar. Sebanyak 50 false positive (FP) menunjukkan pelanggan yang tidak churn salah diklasifikasikan sebagai churn.

  ![latihan10](https://github.com/user-attachments/assets/55b1ab6a-e35e-4193-8a21-afed7987c92b)

  Model ini juga menghasilkan 211 false negative (FN) yang menggambarkan pelanggan yang sebenarnya churn, tetapi tidak teridentifikasi oleh model. Terakhir, ada 182 true positive (TP), yaitu pelanggan benar-benar churn yang berhasil terdeteksi oleh model. Hasil ini memberikan gambaran tentang seberapa baik Random Forest dalam memprediksi churn dan menunjukkan bahwa model menangani masing-masing kelas.

- Hasil Confusion Matrix Algoritma SVM

  Untuk Support Vector Machine (SVM) Classifier, hasil evaluasi menunjukkan distribusi prediksi sebagai berikut: ada 1581 true negative (TN), yang berarti pelanggan tidak churn terdeteksi dengan benar. Sebanyak 26 false positive (FP) menunjukkan pelanggan yang tidak churn salah diklasifikasikan sebagai churn.

  ![latihan11](https://github.com/user-attachments/assets/7bf861e8-105b-4142-9e7e-9eb2c4dfd092)

  Model ini juga menghasilkan 268 false negative (FN), menggambarkan pelanggan yang sebenarnya churn, tetapi tidak teridentifikasi oleh model. Terakhir, ada 125 true positive (TP), yaitu pelanggan benar-benar churn yang berhasil terdeteksi oleh model. Hasil ini mencerminkan bahwa SVM mengelola prediksi churn dan performanya dalam klasifikasi.

- Hasil Confusion Matrix Algoritma Naive Bayes

  Untuk Naive Bayes Classifier, hasil evaluasi memberikan gambaran sebagai berikut: ada 1563 true negative (TN), menunjukkan jumlah pelanggan tidak churn yang terdeteksi dengan benar. Sebanyak 44 false positive (FP) menunjukkan pelanggan yang tidak churn salah diklasifikasikan sebagai churn.

  ![latihan12](https://github.com/user-attachments/assets/1d202e67-b3dd-40d2-9604-4b1033eeb96f)

  Model ini juga menghasilkan 299 false negative (FN), menunjukkan pelanggan yang sebenarnya churn, tetapi tidak teridentifikasi sebagai churn. Terakhir, ada 94 true positive (TP), yaitu pelanggan benar-benar churn yang berhasil terdeteksi oleh model. Hasil ini menggambarkan performa Naive Bayes dalam mengidentifikasi pelanggan yang akan churn.

## Rangkuman Hasil

Berikut adalah ringkasan hasil evaluasi untuk masing-masing model klasifikasi.

- K-Nearest Neighbors (KNN) menunjukkan akurasi sebesar 82.40%. Model ini memiliki precision 59.53%, recall 32.57%, dan F1-Score 42.11%. Angka precision yang relatif tinggi menunjukkan bahwa ketika model mengklasifikasikan seseorang sebagai churn, kemungkinan besar prediksi tersebut benar. Namun, recall yang rendah menunjukkan bahwa model ini sering gagal dalam mengidentifikasi pelanggan yang benar-benar churn.

- Decision Tree memperoleh akurasi sebesar 78.00%. Precision-nya adalah 44.74%, recall 50.89%, dan F1-Score 47.62%. Precision yang lebih rendah dibandingkan dengan KNN menunjukkan bahwa model ini kurang efektif dalam menghindari false positives. Meskipun recall-nya lebih baik, model ini masih kurang dalam hal ketepatan keseluruhan.

- Random Forest tampil dengan akurasi tertinggi sebesar 86.95%. Model ini memiliki precision 78.45%, recall 46.31%, dan F1-Score 58.24%. Tingginya precision menunjukkan model ini sangat baik dalam mengidentifikasi pelanggan yang churn dengan benar dan F1-Score yang baik menunjukkan keseimbangan yang baik antara precision dan recall.

- Support Vector Machine (SVM) memiliki akurasi 85.30%. Precision-nya mencapai 82.78%, recall 31.81%, dan F1-Score 45.96%. Precision yang sangat tinggi menandakan bahwa SVM efektif dalam mengklasifikasikan pelanggan yang churn. Namun, rendahnya recall menunjukkan bahwa model ini mungkin melewatkan banyak pelanggan churn yang sebenarnya.

- Naive Bayes menunjukkan akurasi 82.85%. Precision-nya adalah 68.12%, recall 23.92%, dan F1-Score 35.40%. Meskipun precision-nya relatif tinggi, recall yang sangat rendah menunjukkan bahwa model ini tidak efektif dalam mendeteksi banyak pelanggan churn.

Dalam rangkuman hasil evaluasi model, terlihat bahwa Random Forest adalah model dengan performa terbaik, mengungguli model lainnya dalam hal akurasi, precision, recall, dan F1-Score. Keunggulan ini menunjukkan kemampuannya dalam mengidentifikasi pelanggan yang churn dengan lebih baik dan akurat. Support Vector Machine (SVM) juga menunjukkan performa yang sangat baik dalam precision, tetapi memiliki recall lebih rendah, yang berarti model ini sering melewatkan beberapa pelanggan churn. 

Sementara itu, K-Nearest Neighbors (KNN) dan Naive Bayes memiliki akurasi dan precision yang baik, tetapi kurang optimal dalam recall sehingga sering kali gagal dalam mendeteksi pelanggan churn yang sebenarnya. Decision Tree, meskipun memberikan hasil yang baik dalam recall, memiliki akurasi dan precision lebih rendah dibandingkan model-model lainnya. 
