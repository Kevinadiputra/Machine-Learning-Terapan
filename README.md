# **Predictive Analytics Project Report-Kevin Adiputra Mahesa**

## 1. Domain Project

### **Latar Belakang**

Dengan semakin meningkatnya penggunaan perangkat seluler dalam kehidupan sehari-hari, pemahaman mengenai perilaku pengguna perangkat menjadi sangat penting. Pengguna perangkat seluler kini tidak hanya menggunakan perangkat untuk komunikasi, tetapi juga untuk berbagai aktivitas digital lainnya seperti hiburan, belanja online, pendidikan, dan pekerjaan. Seiring dengan perubahan perilaku ini, perangkat seluler kini mengumpulkan sejumlah besar data terkait penggunaan, yang bisa memberikan wawasan berharga tentang preferensi dan kebiasaan pengguna.

Pengetahuan tentang pola perilaku pengguna tidak hanya bermanfaat bagi pengembangan produk, tetapi juga dapat berperan penting dalam berbagai aspek lain, seperti optimasi penggunaan energi, peningkatan pengalaman pengguna, serta strategi pemasaran berbasis data yang lebih efektif. Misalnya, dengan memahami durasi layar menyala dan konsumsi baterai, produsen perangkat dapat merancang produk dengan daya tahan baterai yang lebih baik atau fitur hemat energi yang lebih efisien. Di sisi lain, bagi pengembang aplikasi, wawasan tentang berapa lama aplikasi digunakan dan jumlah aplikasi yang diinstal dapat digunakan untuk menyesuaikan fungsionalitas dan antarmuka pengguna agar lebih menarik dan sesuai dengan kebutuhan pengguna.

Selain itu, data perilaku pengguna ini juga dapat diterapkan dalam pengambilan keputusan yang lebih cerdas untuk pemasaran digital. Dengan menganalisis pola penggunaan perangkat, pengiklan dapat menargetkan audiens dengan iklan yang lebih relevan berdasarkan kebiasaan dan minat mereka, meningkatkan efektivitas kampanye pemasaran.

Pada proyek ini, model machine learning dikembangkan untuk memprediksi **kategori perilaku pengguna perangkat** berdasarkan berbagai metrik, seperti penggunaan aplikasi, durasi layar menyala, konsumsi baterai, jumlah aplikasi yang terinstal, dan data penggunaan lainnya. Melalui analisis data ini, diharapkan dapat ditemukan pola yang mencerminkan kebiasaan pengguna yang berbeda, yang pada akhirnya dapat membantu dalam merancang produk dan layanan yang lebih sesuai dengan kebutuhan pengguna.

Pemahaman yang lebih baik tentang perilaku pengguna tidak hanya menguntungkan bagi perusahaan teknologi, tetapi juga membuka peluang untuk inovasi dalam desain perangkat dan aplikasi yang lebih ramah pengguna, berkelanjutan, dan relevan dengan perkembangan teknologi saat ini.
## 2. Business Understanding

### **Problem Statements**
1. **Fitur apa yang paling mempengaruhi kelas perilaku pengguna (user behavior class)?**
2. **Model mana yang paling efektif dan baik dalam memprediksi kelas perilaku pengguna?**

### **Goals**
1. **Mengidentifikasi fitur-fitur yang paling berpengaruh terhadap kelas perilaku pengguna.**
2. **Mengidentifikasi model terbaik untuk memprediksi kelas perilaku pengguna.**

### **Solutions**
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
      - **Feature Importance**
Feature importance adalah teknik yang menghitung skor kontribusi masing-masing fitur terhadap kinerja model prediktif, dengan skor yang lebih tinggi menunjukkan fitur yang memiliki dampak lebih besar pada hasil prediksi. Teknik ini membantu dalam memahami data, mengoptimalkan model, dan mengurangi dimensionalitas. [[10]](https://builtin.com/data-science/feature-importance)
   
   Dengan mengidentifikasi fitur yang paling berpengaruh, kita dapat memberikan wawasan lebih dalam tentang perilaku pengguna dan memberikan rekomendasi untuk pengembangan produk atau strategi pemasaran yang lebih efektif berdasarkan data tersebut.

## 3. Data Understanding

### Dataset yang Digunakan 
Dataset yang digunakan dalam proyek ini adalah **Mobile Device Usage and User Behavior Dataset**, yang tersedia di [Kaggle](https://www.kaggle.com/datasets/valakhorasani/mobile-device-usage-and-user-behavior-dataset). Dataset ini menyediakan informasi mendetail tentang pola penggunaan perangkat seluler dan perilaku pengguna.

Berikut adalah gambaran awal dari dataset yang digunakan:

| **User ID** | **Device Model**     | **Operating System** | **App Usage Time (min/day)** | **Screen On Time (hours/day)** | **Battery Drain (mAh/day)** | **Number of Apps Installed** | **Data Usage (MB/day)** | **Age** | **Gender** | **User Behavior Class** |
|-------------|----------------------|----------------------|------------------------------|--------------------------------|-----------------------------|------------------------------|-------------------------|---------|------------|--------------------------|
| 1           | Google Pixel 5       | Android              | 393                          | 6.4                            | 1872                       | 67                           | 1122                   | 40      | Male       | 4                        |
| 2           | OnePlus 9            | Android              | 268                          | 4.7                            | 1331                       | 42                           | 944                    | 47      | Female     | 3                        |
| 3           | Xiaomi Mi 11         | Android              | 154                          | 4.0                            | 761                        | 32                           | 322                    | 42      | Male       | 2                        |
| 4           | Google Pixel 5       | Android              | 239                          | 4.8                            | 1676                       | 56                           | 871                    | 20      | Male       | 3                        |
| 5           | iPhone 12            | iOS                  | 187                          | 4.3                            | 1367                       | 58                           | 988                    | 31      | Female     | 3                        |

*Tabel 1: Gambaran awal Dataset*

### Deskripsi Dataset
Dataset ini terdiri dari **700 sampel** dengan **10 kolom fitur utama** dan 1 kolom target. Semua data telah diproses sehingga tidak memiliki nilai yang hilang, dan kolom **User ID**, yang tidak relevan dengan analisis, telah dihapus.

Berikut adalah deskripsi fitur yang digunakan:
- **Device Model**: Jenis atau model perangkat pengguna (misalnya, Android/iOS).
- **Operating System**: Sistem operasi yang digunakan oleh perangkat.
- **App Usage Time (min/day)**: Total waktu penggunaan aplikasi dalam satu hari (menit).
- **Screen On Time (hours/day)**: Durasi layar perangkat aktif dalam satu hari (jam).
- **Battery Drain (mAh/day)**: Konsumsi baterai perangkat dalam satu hari (mAh).
- **Number of Apps Installed**: Jumlah aplikasi yang terinstal pada perangkat.
- **Data Usage (MB/day)**: Penggunaan data internet dalam satu hari (MB).
- **Age**: Usia pengguna perangkat (tahun).
- **Gender**: Jenis kelamin pengguna (Laki-laki/Perempuan).
- **User Behavior Class**: Kategori perilaku pengguna, yang digunakan sebagai **label target** untuk klasifikasi.

Dataset ini memberikan gambaran menyeluruh tentang kebiasaan penggunaan perangkat, sehingga memungkinkan analisis untuk memahami faktor yang memengaruhi perilaku pengguna secara lebih mendalam. 

### Informasi Dataset
Berikut adalah tabel dengan nama dan informasi terkait data dalam dataset:

| **Nomor** | **Nama Kolom**              | **Tipe Data** | **Jumlah Non-Null** | **Deskripsi**                       |
|-----------|-----------------------------|---------------|---------------------|-------------------------------------|
| 1         | User ID                     | `int64`       | 700                 | Identitas unik untuk setiap pengguna. |
| 2         | Device Model                | `object`      | 700                 | Model perangkat pengguna.          |
| 3         | Operating System            | `object`      | 700                 | Sistem operasi perangkat (Android/iOS). |
| 4         | App Usage Time (min/day)    | `int64`       | 700                 | Total waktu penggunaan aplikasi dalam menit per hari. |
| 5         | Screen On Time (hours/day)  | `float64`     | 700                 | Durasi layar menyala dalam jam per hari. |
| 6         | Battery Drain (mAh/day)     | `int64`       | 700                 | Konsumsi baterai per hari dalam mAh. |
| 7         | Number of Apps Installed    | `int64`       | 700                 | Jumlah aplikasi yang diinstal pada perangkat. |
| 8         | Data Usage (MB/day)         | `int64`       | 700                 | Total penggunaan data internet dalam MB per hari. |
| 9         | Age                         | `int64`       | 700                 | Usia pengguna perangkat.            |
| 10        | Gender                      | `object`      | 700                 | Jenis kelamin pengguna.             |
| 11        | User Behavior Class         | `int64`       | 700                 | Kategori perilaku pengguna, digunakan sebagai target. |

*Tabel 2: Informasi Dataset*

## 4. Data Preparation

### Teknik Data Preparation
Langkah-langkah persiapan data meliputi:
1. **Label Encoder**: Kolom **Device Model** dan **Operating System** diubah menjadi vektor numerik.
```python
# @title Mengubah kolom kategorikal menjadi numerikal menggunakan LabelEncoder
labencoder = preprocessing.LabelEncoder()
df['Operating System'] = labencoder.fit_transform(df['Operating System'])
df['Gender'] = labencoder.fit_transform(df['Gender'])
df
```
- **Apa itu Label Encoding?**
  
- **Mengapa Menggunakan Label Encoding?**
   - Kolom **`Operating System`** dan **`Gender`** adalah tipe data kategorikal dengan nilai unik binary, seperti:
     - `Operating System`: "Android" dan "iOS".
     - `Gender`: "Male" dan "Female".
   - Label Encoding mengubah nilai-nilai kategorikal ini menjadi angka numerik, seperti pada kasus kali ini:
     - "Android" → 0, "iOS" → 1.
     - "Male" → 0, "Female" → 1.
   - Pendekatan ini ideal untuk kolom dengan **nilai kategorikal yang bersifat biner (binary)**, karena:
     - Mudah diterapkan.
     - Efisien untuk model yang tidak bergantung pada hubungan antar-kategori.

- **Keunggulan Label Encoding untuk Binary Categories:**
   - Proses sederhana dan cepat.
   - Tidak memperkenalkan redundansi seperti **One-Hot Encoding** yang menambahkan kolom tambahan.
   - Sangat cocok untuk dataset dengan fitur binary kategorikal.

- **Hasil:**
   Setelah transformasi, kolom `Operating System` dan `Gender` dalam dataset `df` akan berisi nilai numerik, mempermudah algoritma machine learning untuk memproses data tersebut.
  
Berikut adalah tampilan Data setelah dilakukan Label Encoder:
| Index | Device Model        | Operating System | App Usage Time (min/day) | Screen On Time (hours/day) | Battery Drain (mAh/day) | Number of Apps Installed | Data Usage (MB/day) | Age | Gender | User Behavior Class |
|-------|---------------------|------------------|--------------------------|----------------------------|-------------------------|--------------------------|---------------------|-----|--------|--------------------|
| 0     | Google Pixel 5      | 0                | 393                      | 6.4                        | 1872                   | 67                       | 1122                | 40  | 1      | 4                  |
| 1     | OnePlus 9           | 0                | 268                      | 4.7                        | 1331                   | 42                       | 944                 | 47  | 0      | 3                  |
| 2     | Xiaomi Mi 11        | 0                | 154                      | 4.0                        | 761                    | 32                       | 322                 | 42  | 1      | 2                  |
| 3     | Google Pixel 5      | 0                | 239                      | 4.8                        | 1676                   | 56                       | 871                 | 20  | 1      | 3                  |
| 4     | iPhone 12           | 1                | 187                      | 4.3                        | 1367                   | 58                       | 988                 | 31  | 0      | 3                  |
*Tabel 3: Dataset setelah dilakukan Label Encoder*
## 5.  Eksplorasi Data
-


## 6. Modeling

### Algoritma yang Digunakan
Algoritma **Random Forest Classifier** dipilih karena mampu menangani dataset dengan jumlah fitur yang cukup banyak dan bekerja dengan baik pada data klasifikasi.

### Hyperparameter Tuning
Hyperparameter tuning dilakukan untuk mencari kombinasi parameter terbaik dengan menggunakan metode **Grid Search**. Kombinasi parameter yang dicari termasuk:
- **n_estimators**: Jumlah pohon dalam hutan.
- **max_depth**: Kedalaman maksimum dari pohon.
- **min_samples_split**: Jumlah minimum sampel untuk membagi node internal.

## 7. Evaluation and interptetation

### Metrik Evaluasi
Metrik evaluasi yang digunakan adalah:
- **Accuracy**: Mengukur seberapa sering model memberikan prediksi yang benar.
- **Precision**: Mengukur seberapa baik model memprediksi kelas positif dengan benar.
- **Recall**: Mengukur seberapa baik model menangkap semua sampel yang benar-benar positif.
- **F1-Score**: Rata-rata harmonik dari precision dan recall.

### Hasil Proyek
Setelah pelatihan awal, model berhasil mencapai hasil evaluasi yang sangat baik dengan semua metrik bernilai sempurna. Namun, tuning lebih lanjut atau penggunaan algoritma lain seperti **XGBoost** dapat dilakukan untuk memastikan model lebih robust dan tidak overfitting.
