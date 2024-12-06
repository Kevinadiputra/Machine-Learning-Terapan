# **Predictive Analytics Project Report-Kevin Adiputra Mahesa**

## 1. Domain Project

### **Latar Belakang**

Dengan semakin meningkatnya penggunaan perangkat seluler dalam kehidupan sehari-hari, pemahaman mengenai perilaku pengguna perangkat menjadi sangat penting. Pengguna perangkat seluler kini tidak hanya menggunakan perangkat untuk komunikasi, tetapi juga untuk berbagai aktivitas digital lainnya seperti hiburan, belanja online, pendidikan, dan pekerjaan. Seiring dengan perubahan perilaku ini, perangkat seluler kini mengumpulkan sejumlah besar data terkait penggunaan, yang bisa memberikan wawasan berharga tentang preferensi dan kebiasaan pengguna.

Pengetahuan tentang pola perilaku pengguna tidak hanya bermanfaat bagi pengembangan produk, tetapi juga dapat berperan penting dalam berbagai aspek lain, seperti optimasi penggunaan energi, peningkatan pengalaman pengguna, serta strategi pemasaran berbasis data yang lebih efektif. Misalnya, dengan memahami durasi layar menyala dan konsumsi baterai, produsen perangkat dapat merancang produk dengan daya tahan baterai yang lebih baik atau fitur hemat energi yang lebih efisien. Di sisi lain, bagi pengembang aplikasi, wawasan tentang berapa lama aplikasi digunakan dan jumlah aplikasi yang diinstal dapat digunakan untuk menyesuaikan fungsionalitas dan antarmuka pengguna agar lebih menarik dan sesuai dengan kebutuhan pengguna.

Selain itu, data perilaku pengguna ini juga dapat diterapkan dalam pengambilan keputusan yang lebih cerdas untuk pemasaran digital. Dengan menganalisis pola penggunaan perangkat, pengiklan dapat menargetkan audiens dengan iklan yang lebih relevan berdasarkan kebiasaan dan minat mereka, meningkatkan efektivitas kampanye pemasaran.

Pada proyek ini, model machine learning dikembangkan untuk memprediksi **kategori perilaku pengguna perangkat** berdasarkan berbagai metrik, seperti penggunaan aplikasi, durasi layar menyala, konsumsi baterai, jumlah aplikasi yang terinstal, dan data penggunaan lainnya. Melalui analisis data ini, diharapkan dapat ditemukan pola yang mencerminkan kebiasaan pengguna yang berbeda, yang pada akhirnya dapat membantu dalam merancang produk dan layanan yang lebih sesuai dengan kebutuhan pengguna.

Pemahaman yang lebih baik tentang perilaku pengguna tidak hanya menguntungkan bagi perusahaan teknologi, tetapi juga membuka peluang untuk inovasi dalam desain perangkat dan aplikasi yang lebih ramah pengguna, berkelanjutan, dan relevan dengan perkembangan teknologi saat ini.
## 2. Business Understanding

## 2. Pemahaman Bisnis

### **Pernyataan Masalah**
1. **Fitur apa yang paling mempengaruhi kelas perilaku pengguna (user behavior class)?**
2. **Model mana yang paling efektif dan baik dalam memprediksi kelas perilaku pengguna?**

### **Tujuan**
1. **Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap kelas perilaku pengguna.**
2. **Mengidentifikasi model terbaik untuk memprediksi kelas perilaku pengguna.**

### **Solusi**
Untuk mencapai tujuan ini, langkah-langkah berikut dilakukan:

1. **Membandingkan Performa 9 Model:**
Sebagai langkah awal, 9 model berbeda akan dievaluasi untuk menentukan model mana yang memberikan akurasi terbaik dalam memprediksi kelas perilaku pengguna. Model yang diuji meliputi:
Berikut adalah penjelasan yang lebih rapi dan terstruktur mengenai 9 model yang digunakan dalam perbandingan performa untuk memprediksi kelas perilaku pengguna:
- **Decision Tree (DT)**
Decision Tree adalah algoritma yang membagi data menjadi beberapa cabang berdasarkan fitur tertentu, dengan tujuan meminimalkan ketidakpastian dalam prediksi. Setiap node dalam pohon mewakili sebuah fitur, dan setiap cabang mewakili keputusan berdasarkan nilai fitur tersebut. Model ini mudah dipahami dan diinterpretasikan.[[1]](https://www.ibm.com/id-id/topics/decision-trees)
- **Random Forest (RF)**
Random Forest adalah metode ensemble yang menggunakan banyak pohon keputusan (decision trees) untuk meningkatkan akurasi prediksi. Setiap pohon dibangun menggunakan subset acak dari data dan fitur, dengan hasil akhirnya ditentukan oleh voting mayoritas dari semua pohon. Ini membantu mengurangi overfitting dan meningkatkan generalisasi.[[2]](https://www.ibm.com/topics/random-forest)
- **Logistic Regression (LG)**
Logistic Regression adalah model statistik yang digunakan untuk klasifikasi biner atau multi-kelas. Model ini menghitung probabilitas bahwa suatu input termasuk dalam suatu kelas tertentu menggunakan fungsi logit. Logistic Regression sangat populer karena kesederhanaannya dalam interpretasi.[[3]](https://www.ibm.com/topics/logistic-regression)
- **K-Nearest Neighbors (KNN)**
KNN adalah algoritma non-parametrik yang mengklasifikasikan data berdasarkan kedekatannya dengan data lain. Setiap data diberi label berdasarkan mayoritas kelas dari \(k\) tetangga terdekatnya. Algoritma ini sederhana namun seringkali efektif untuk dataset kecil dan sederhana.[[4]](https://esairina.medium.com/algoritma-k-nearest-neighbor-knn-penjelasan-dan-implementasi-untuk-klasifikasi-kanker-ff9b7fbe0a4)
- **Support Vector Machine (SVM)**
SVM adalah algoritma klasifikasi yang berusaha menemukan hyperplane terbaik yang memisahkan kelas-kelas data. Tujuannya adalah untuk memaksimalkan margin (jarak) antara kelas-kelas tersebut. SVM sangat efektif untuk data dengan dimensi tinggi dan mampu menangani masalah klasifikasi non-linear dengan kernel trick.[[5]](https://www.ibm.com/id-id/topics/support-vector-machine)
- **AdaBoost**
AdaBoost (Adaptive Boosting) adalah teknik ensemble yang menggabungkan beberapa model lemah (weak learners) untuk membentuk model yang lebih kuat. Algoritma ini memberi bobot lebih pada data yang salah klasifikasi pada iterasi sebelumnya, dengan tujuan memperbaiki kesalahan yang terjadi.[[6]](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/)
- **XGBoost**
XGBoost adalah implementasi optimasi dari Gradient Boosting yang menggunakan banyak pohon keputusan. Teknik ini menggabungkan pembelajaran berbasis boosting dan regularisasi untuk meningkatkan kecepatan dan mengurangi overfitting, menjadikannya salah satu algoritma yang paling populer untuk masalah klasifikasi.[[7]](https://xgboost.readthedocs.io/)
- **Naive Bayes**
Naive Bayes adalah model probabilistik yang menggunakan teorema Bayes untuk klasifikasi. Model ini mengasumsikan bahwa fitur-fitur dalam data bersifat independen, yang menyederhanakan perhitungan probabilitas kelas. Naive Bayes sering digunakan untuk masalah klasifikasi teks, seperti analisis sentimen.[[8]](https://www.ibm.com/topics/naive-bayes)
- **Gradient Boosting**
Gradient Boosting adalah metode ensemble yang membangun model secara bertahap. Setiap model baru berfokus pada kesalahan yang dilakukan oleh model sebelumnya, sehingga model secara iteratif memperbaiki kesalahan prediksi. Teknik ini sangat efektif untuk berbagai macam tugas klasifikasi dan regresi.[[9]](https://www.geeksforgeeks.org/ml-gradient-boosting/)
### Evaluasi Model:
Setiap model ini akan dievaluasi dengan menggunakan **akurasi** untuk menilai seberapa baik prediksi yang dihasilkan dalam memprediksi kelas perilaku pengguna. Selain itu, **confusion matrix** akan digunakan untuk memberikan gambaran lebih mendalam mengenai kesalahan prediksi, termasuk jumlah **false positives** dan **false negatives** untuk masing-masing kelas.

3. **Feature Importance untuk Mengidentifikasi Fitur Utama:**
   - Setelah menentukan model terbaik, **feature importance** akan dilakukan untuk memahami fitur mana yang memiliki pengaruh paling besar terhadap prediksi kelas perilaku pengguna. Proses ini akan mengidentifikasi fitur-fitur seperti **waktu penggunaan aplikasi**, **waktu layar menyala**, dan **konsumsi baterai**, serta faktor-faktor lain yang dapat menjelaskan pola perilaku pengguna dengan lebih baik.
   
   Dengan mengidentifikasi fitur yang paling berpengaruh, kita dapat memberikan wawasan lebih dalam tentang perilaku pengguna dan memberikan rekomendasi untuk pengembangan produk atau strategi pemasaran yang lebih efektif berdasarkan data tersebut.

## 3. Data Understanding

### Deskripsi Dataset
Dataset yang digunakan berisi 700 sampel dengan 11 kolom fitur:
- **Device Model**: Model perangkat.
- **Operating System**: Sistem operasi perangkat (Android/iOS).
- **App Usage Time (min/day)**: Waktu penggunaan aplikasi per hari dalam menit.
- **Screen On Time (hours/day)**: Waktu layar menyala per hari dalam jam.
- **Battery Drain (mAh/day)**: Konsumsi baterai per hari.
- **Number of Apps Installed**: Jumlah aplikasi terinstal.
- **Data Usage (MB/day)**: Penggunaan data per hari dalam MB.
- **Age**: Usia pengguna.
- **Gender**: Jenis kelamin pengguna.
- **User Behavior Class**: Kelas perilaku pengguna (target).

Dataset ini tidak memiliki data yang hilang dan sudah dibersihkan dari kolom **User ID** yang tidak relevan.

### Eksplorasi Data
Kami melakukan **one-hot encoding** pada kolom **Device Model** dan **Operating System** untuk mengubah fitur kategorikal menjadi fitur numerik. Fitur numerikal seperti waktu penggunaan aplikasi, konsumsi baterai, dan lainnya di-normalisasi untuk memastikan skala yang serupa.

## 4. Data Preparation

### Teknik Data Preparation
Langkah-langkah persiapan data meliputi:
1. **One-Hot Encoding**: Kolom **Device Model** dan **Operating System** diubah menjadi vektor numerik.
2. **Normalisasi**: Fitur numerik seperti **App Usage Time**, **Screen On Time**, dan lainnya di-scale menggunakan **StandardScaler**.
3. **Pembagian Data**: Data dibagi menjadi training (80%) dan testing (20%).

## 5. Modeling

### Algoritma yang Digunakan
Algoritma **Random Forest Classifier** dipilih karena mampu menangani dataset dengan jumlah fitur yang cukup banyak dan bekerja dengan baik pada data klasifikasi.

### Hyperparameter Tuning
Hyperparameter tuning dilakukan untuk mencari kombinasi parameter terbaik dengan menggunakan metode **Grid Search**. Kombinasi parameter yang dicari termasuk:
- **n_estimators**: Jumlah pohon dalam hutan.
- **max_depth**: Kedalaman maksimum dari pohon.
- **min_samples_split**: Jumlah minimum sampel untuk membagi node internal.

### Evaluasi Model
Model diuji menggunakan data testing dan hasil evaluasi awal menunjukkan bahwa akurasi model mencapai **100%**, dengan nilai precision, recall, dan F1-score untuk semua kelas adalah **1.00**. Hasil ini mengindikasikan potensi overfitting yang bisa diperbaiki dengan tuning lebih lanjut atau pemilihan algoritma alternatif.

## 6. Evaluation

### Metrik Evaluasi
Metrik evaluasi yang digunakan adalah:
- **Accuracy**: Mengukur seberapa sering model memberikan prediksi yang benar.
- **Precision**: Mengukur seberapa baik model memprediksi kelas positif dengan benar.
- **Recall**: Mengukur seberapa baik model menangkap semua sampel yang benar-benar positif.
- **F1-Score**: Rata-rata harmonik dari precision dan recall.

### Hasil Proyek
Setelah pelatihan awal, model berhasil mencapai hasil evaluasi yang sangat baik dengan semua metrik bernilai sempurna. Namun, tuning lebih lanjut atau penggunaan algoritma lain seperti **XGBoost** dapat dilakukan untuk memastikan model lebih robust dan tidak overfitting.
