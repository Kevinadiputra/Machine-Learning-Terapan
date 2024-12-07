# **Predictive Analytics Project Report-Kevin Adiputra Mahesa**

## 1. Domain Project

### **Latar Belakang**

Dengan semakin meningkatnya penggunaan perangkat seluler dalam kehidupan sehari-hari, pemahaman mengenai perilaku pengguna perangkat menjadi sangat penting. Pengguna perangkat seluler kini tidak hanya menggunakan perangkat untuk komunikasi, tetapi juga untuk berbagai aktivitas digital lainnya seperti hiburan, belanja online, pendidikan, dan pekerjaan. Seiring dengan perubahan perilaku ini, perangkat seluler kini mengumpulkan sejumlah besar data terkait penggunaan, yang bisa memberikan wawasan berharga tentang preferensi dan kebiasaan pengguna.[[1]](https://databycy.com/2024/10/27/analyzing-user-behavior-based-on-device-characteristics-and-app-usage/)

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
Decision Tree adalah algoritma yang membagi data menjadi beberapa cabang berdasarkan fitur tertentu, dengan tujuan meminimalkan ketidakpastian dalam prediksi. Setiap node dalam pohon mewakili sebuah fitur, dan setiap cabang mewakili keputusan berdasarkan nilai fitur tersebut. Model ini mudah dipahami dan diinterpretasikan.[[2]](https://www.ibm.com/id-id/topics/decision-trees)
- **Random Forest (RF)**
Random Forest adalah metode ensemble yang menggunakan banyak pohon keputusan (decision trees) untuk meningkatkan akurasi prediksi. Setiap pohon dibangun menggunakan subset acak dari data dan fitur, dengan hasil akhirnya ditentukan oleh voting mayoritas dari semua pohon. Ini membantu mengurangi overfitting dan meningkatkan generalisasi.[[3]](https://www.ibm.com/topics/random-forest)
- **Logistic Regression (LG)**
Logistic Regression adalah model statistik yang digunakan untuk klasifikasi biner atau multi-kelas. Model ini menghitung probabilitas bahwa suatu input termasuk dalam suatu kelas tertentu menggunakan fungsi logit. Logistic Regression sangat populer karena kesederhanaannya dalam interpretasi.[[4]](https://www.ibm.com/topics/logistic-regression)
- **K-Nearest Neighbors (KNN)**
KNN adalah algoritma non-parametrik yang mengklasifikasikan data berdasarkan kedekatannya dengan data lain. Setiap data diberi label berdasarkan mayoritas kelas dari \(k\) tetangga terdekatnya. Algoritma ini sederhana namun seringkali efektif untuk dataset kecil dan sederhana.[[5]](https://esairina.medium.com/algoritma-k-nearest-neighbor-knn-penjelasan-dan-implementasi-untuk-klasifikasi-kanker-ff9b7fbe0a4)
- **Support Vector Machine (SVM)**
SVM adalah algoritma klasifikasi yang berusaha menemukan hyperplane terbaik yang memisahkan kelas-kelas data. Tujuannya adalah untuk memaksimalkan margin (jarak) antara kelas-kelas tersebut. SVM sangat efektif untuk data dengan dimensi tinggi dan mampu menangani masalah klasifikasi non-linear dengan kernel trick.[[6]](https://www.ibm.com/id-id/topics/support-vector-machine)
- **AdaBoost**
AdaBoost (Adaptive Boosting) adalah teknik ensemble yang menggabungkan beberapa model lemah (weak learners) untuk membentuk model yang lebih kuat. Algoritma ini memberi bobot lebih pada data yang salah klasifikasi pada iterasi sebelumnya, dengan tujuan memperbaiki kesalahan yang terjadi.[[7]](https://www.analyticsvidhya.com/blog/2021/09/adaboost-algorithm-a-complete-guide-for-beginners/)
- **XGBoost**
XGBoost adalah implementasi optimasi dari Gradient Boosting yang menggunakan banyak pohon keputusan. Teknik ini menggabungkan pembelajaran berbasis boosting dan regularisasi untuk meningkatkan kecepatan dan mengurangi overfitting, menjadikannya salah satu algoritma yang paling populer untuk masalah klasifikasi.[[8]](https://xgboost.readthedocs.io/)
- **Naive Bayes**
Naive Bayes adalah model probabilistik yang menggunakan teorema Bayes untuk klasifikasi. Model ini mengasumsikan bahwa fitur-fitur dalam data bersifat independen, yang menyederhanakan perhitungan probabilitas kelas. Naive Bayes sering digunakan untuk masalah klasifikasi teks, seperti analisis sentimen.[[9]](https://www.ibm.com/topics/naive-bayes)
- **Gradient Boosting**
Gradient Boosting adalah metode ensemble yang membangun model secara bertahap. Setiap model baru berfokus pada kesalahan yang dilakukan oleh model sebelumnya, sehingga model secara iteratif memperbaiki kesalahan prediksi. Teknik ini sangat efektif untuk berbagai macam tugas klasifikasi dan regresi.[[10]](https://www.geeksforgeeks.org/ml-gradient-boosting/)
### Evaluasi Model:
Setiap model ini akan dievaluasi dengan menggunakan **akurasi** untuk menilai seberapa baik prediksi yang dihasilkan dalam memprediksi kelas perilaku pengguna. Selain itu, **confusion matrix** akan digunakan untuk memberikan gambaran lebih mendalam mengenai kesalahan prediksi, termasuk jumlah **false positives** dan **false negatives** untuk masing-masing kelas.

3. **Feature Importance untuk Mengidentifikasi Fitur Utama:**
   - Setelah menentukan model terbaik, **feature importance** akan dilakukan untuk memahami fitur mana yang memiliki pengaruh paling besar terhadap prediksi kelas perilaku pengguna. Proses ini akan mengidentifikasi fitur-fitur seperti **waktu penggunaan aplikasi**, **waktu layar menyala**, dan **konsumsi baterai**, serta faktor-faktor lain yang dapat menjelaskan pola perilaku pengguna dengan lebih baik.
      - **Feature Importance**
Feature importance adalah teknik yang menghitung skor kontribusi masing-masing fitur terhadap kinerja model prediktif, dengan skor yang lebih tinggi menunjukkan fitur yang memiliki dampak lebih besar pada hasil prediksi. Teknik ini membantu dalam memahami data, mengoptimalkan model, dan mengurangi dimensionalitas. [[11]](https://builtin.com/data-science/feature-importance)
   
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
1. **Label Encoder**: Kolom **Gender** dan **Operating System** diubah menjadi vektor numerik.
```python
# @title Mengubah kolom kategorikal menjadi numerikal menggunakan LabelEncoder
labencoder = preprocessing.LabelEncoder()
df['Operating System'] = labencoder.fit_transform(df['Operating System'])
df['Gender'] = labencoder.fit_transform(df['Gender'])
df
```
- **Apa itu Label Encoding?**

  Label encoding adalah teknik dalam pemrosesan data yang digunakan untuk mengubah data kategorikal menjadi representasi numerik. Teknik ini menggantikan setiap kategori unik dalam sebuah kolom dengan angka tertentu, biasanya berdasarkan urutan kemunculan atau tingkatannya.[[12]](https://www.geeksforgeeks.org/ml-label-encoding-of-datasets-in-python/)

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

### 2. One-Hot Encoding pada Kolom Kategorikal Non-Ordinal("Device Model")

**Apa itu One-Hot Encoding?**  
One-Hot Encoding adalah teknik representasi data yang mengonversi variabel kategorikal menjadi format yang lebih mudah dipahami oleh model machine learning. Teknik ini menciptakan kolom biner terpisah untuk setiap kategori unik pada kolom awal. Setiap kolom baru diisi dengan nilai **1** jika kategori tersebut ada, atau **0** jika tidak.[[13]](https://www.geeksforgeeks.org/ml-one-hot-encoding/)

**Mengapa Menggunakan One-Hot Encoding?**  
One-Hot Encoding digunakan untuk kolom kategorikal **non-ordinal**, seperti **Device Model**, yang kategorinya tidak memiliki urutan atau hierarki. Dalam kasus ini, representasi numerikal biasa (seperti Label Encoding) dapat menyesatkan model dengan menyiratkan adanya hubungan ordinal antara kategori. Dengan One-Hot Encoding, hubungan ordinal ini dihindari, memastikan model memberikan bobot yang adil untuk setiap kategori.

**Keunggulan One-Hot Encoding**  
1. **Meningkatkan Akurasi Model**: Representasi biner menghindari bias yang mungkin terjadi akibat interpretasi ordinal pada data.
2. **Kesesuaian dengan Algoritma Machine Learning**: Algoritma seperti regresi linier dan neural network bekerja lebih baik dengan data numerikal yang tidak menyiratkan hubungan ordinal.
3. **Memperjelas Informasi Kategori**: Setiap kategori mendapatkan kolom yang jelas tanpa tumpang tindih.

**Langkah-Langkah Implementasi**
1. **Konversi Kategori ke Representasi Biner**  
   Menggunakan fungsi `pd.get_dummies`, semua nilai unik dari kolom **Device Model** dikonversi menjadi kolom baru yang hanya berisi **0** dan **1**.  
   ```python
   df = df.join(pd.get_dummies(df['Device Model'], dtype=int))
   ```

2. **Penggabungan dengan Dataset Asli**  
   Kolom hasil encoding langsung digabungkan dengan dataset awal.

3. **Penghapusan Kolom Asli**  
   Kolom **Device Model** yang asli dihapus menggunakan fungsi `drop` karena informasinya telah direpresentasikan dalam kolom hasil encoding.  
   ```python
   df = df.drop(columns='Device Model')
   ```

**Hasil**  
Setelah proses ini, dataset memiliki kolom tambahan untuk setiap model perangkat. Berikut adalah contoh hasil setelah One-Hot Encoding diterapkan pada kolom **Device Model**:

| Operating System | App Usage Time (min/day) | Screen On Time (hours/day) | Battery Drain (mAh/day) | Number of Apps Installed | Data Usage (MB/day) | Age | Gender | User Behavior Class | Google Pixel 5 | OnePlus 9 | Samsung Galaxy S21 | Xiaomi Mi 11 | iPhone 12 |
|-------------------|--------------------------|----------------------------|--------------------------|--------------------------|---------------------|-----|--------|--------------------|----------------|-----------|--------------------|--------------|-----------|
| 0                 | 393                      | 6.4                        | 1872                     | 67                       | 1122                | 40  | 1      | 4                  | 1              | 0         | 0                  | 0            | 0         |
| 0                 | 268                      | 4.7                        | 1331                     | 42                       | 944                 | 47  | 0      | 3                  | 0              | 1         | 0                  | 0            | 0         |
| 0                 | 154                      | 4.0                        | 761                      | 32                       | 322                 | 42  | 1      | 2                  | 0              | 0         | 0                  | 1            | 0         |
| 0                 | 239                      | 4.8                        | 1676                     | 56                       | 871                 | 20  | 1      | 3                  | 1              | 0         | 0                  | 0            | 0         |
| 1                 | 187                      | 4.3                        | 1367                     | 58                       | 988                 | 31  | 0      | 3                  | 0              | 0         | 0                  | 0            | 1         |

*Tabel 4: Dataset setelah One-Hot Encoding*

## 5. Exploratory Data Analysis

Pada tahap Exploratory Data Analysis (EDA), dilakukan eksplorasi awal terhadap dataset untuk memahami distribusi data, statistik deskriptif, dan hubungan antar fitur. Berikut adalah penjelasan dan hasil analisis dari data yang diberikan.

### **Melihat Deskripsi Statistik**

Fungsi `describe()` menghasilkan deskripsi statistik seperti **count**, **mean**, **std**, **min**, **25%**, **50% (median)**, **75%**, dan **max** untuk setiap kolom numerik dalam dataset.

| Statistik          | Operating System | App Usage Time (min/day) | Screen On Time (hours/day) | Battery Drain (mAh/day) | Number of Apps Installed | Data Usage (MB/day) | Age   | Gender | User Behavior Class | Google Pixel 5 | OnePlus 9 | Samsung Galaxy S21 | Xiaomi Mi 11 | iPhone 12 |
|---------------------|------------------|--------------------------|----------------------------|--------------------------|--------------------------|---------------------|-------|--------|---------------------|----------------|-----------|--------------------|--------------|-----------|
| **Count**           | 700              | 700                      | 700                        | 700                      | 700                      | 700                 | 700   | 700    | 700                 | 700            | 700       | 700                | 700          | 700       |
| **Mean**            | 0.21             | 271.13                   | 5.27                       | 1525.16                 | 50.68                   | 929.74             | 38.48 | 0.52   | 2.99                | 0.20           | 0.19      | 0.19               | 0.21         | 0.21      |
| **Std Dev**         | 0.41             | 177.20                   | 3.07                       | 819.14                  | 26.94                   | 640.45             | 12.01 | 0.50   | 1.40                | 0.40           | 0.39      | 0.39               | 0.41         | 0.41      |
| **Min**             | 0.00             | 30.00                    | 1.00                       | 302.00                  | 10.00                   | 102.00            | 18.00 | 0.00   | 1.00                | 0.00           | 0.00      | 0.00               | 0.00         | 0.00      |
| **25%**             | 0.00             | 113.25                   | 2.50                       | 722.25                  | 26.00                   | 373.00            | 28.00 | 0.00   | 2.00                | 0.00           | 0.00      | 0.00               | 0.00         | 0.00      |
| **50% (Median)**    | 0.00             | 227.50                   | 4.90                       | 1502.50                 | 49.00                   | 823.50             | 38.00 | 1.00   | 3.00                | 0.00           | 0.00      | 0.00               | 0.00         | 0.00      |
| **75%**             | 0.00             | 434.25                   | 7.40                       | 2229.50                 | 74.00                   | 1341.00           | 49.00 | 1.00   | 4.00                | 0.00           | 0.00      | 0.00               | 0.00         | 0.00      |
| **Max**             | 1.00             | 598.00                   | 12.00                      | 2993.00                 | 99.00                   | 2497.00           | 59.00 | 1.00   | 5.00                | 1.00           | 1.00      | 1.00               | 1.00         | 1.00      |

*Tabel 5 : Tampilan Statistik Data*

#### **Analisis Deskriptif**

1. **Operating System**: Kolom ini berisi nilai biner (0 atau 1) untuk tipe sistem operasi. Mean sebesar 0.21 menunjukkan bahwa mayoritas data memiliki nilai 0.
2. **App Usage Time**: Waktu rata-rata penggunaan aplikasi adalah 271.13 menit per hari, dengan standar deviasi sebesar 177.20 menit, menunjukkan adanya variasi yang signifikan di antara pengguna.
3. **Screen On Time**: Rata-rata waktu layar menyala adalah 5.27 jam per hari, dengan variasi yang relatif moderat.
4. **Battery Drain**: Penggunaan daya rata-rata adalah 1525.16 mAh per hari, dengan maksimum mencapai 2993 mAh.
5. **Age**: Rata-rata usia pengguna adalah 38.48 tahun, dengan kisaran usia 18–59 tahun.
6. **Gender**: Kolom biner yang hampir seimbang (0 untuk perempuan, 1 untuk laki-laki) dengan rata-rata 0.52.
7. **User Behavior Class**: Rata-rata kelas perilaku pengguna adalah 2.99, mendekati median (3), menunjukkan distribusi yang cukup seimbang.

### **Exploratory Data Analysis - Univariate Analysis**

![image](https://github.com/user-attachments/assets/d40fd8ab-cec0-4984-8867-95d159be8a75)  
*Gambar 1: Univariate Analysis (Operating System)*

![image](https://github.com/user-attachments/assets/a13ad5ab-8d86-4f33-9ddc-6852f7872793)  
*Gambar 2: Univariate Analysis (App Usage Time - min/day)*

![image](https://github.com/user-attachments/assets/9f0ebcac-1fa3-4fe2-b9ee-e465201774c6)  
*Gambar 3: Univariate Analysis (Screen On Time - hours/day)*

![image](https://github.com/user-attachments/assets/9e3d95e9-1bd4-42d6-933f-541468717e1d)  
*Gambar 4: Univariate Analysis (Battery Drain - mAh/day)*

![image](https://github.com/user-attachments/assets/23610339-7791-44ff-a922-f4679bdfe450)  
*Gambar 5: Univariate Analysis (Number of Apps Installed)*

![image](https://github.com/user-attachments/assets/aec95395-b717-4246-9fdd-cd1b293c8f76)  
*Gambar 6: Univariate Analysis (Data Usage - MB/day)*

![image](https://github.com/user-attachments/assets/6c6f91c6-e1a1-435c-be4f-df1cbc73b0ff)  
*Gambar 7: Univariate Analysis (Age)*

![image](https://github.com/user-attachments/assets/2599108f-0faf-4408-b042-b64881edda94)  
*Gambar 8: Univariate Analysis (Gender)*

![image](https://github.com/user-attachments/assets/ff5b5854-d894-4329-b4b5-469c2fadf6bd)  
*Gambar 9: Univariate Analysis (User Behavior Class)*

![image](https://github.com/user-attachments/assets/9fffa7eb-de6d-4f01-80ed-0b3e55f37450)  
*Gambar 10: Univariate Analysis (Google Pixel 5)*

![image](https://github.com/user-attachments/assets/9a947174-4057-4b04-9c2c-1cb4aa001a74)  
*Gambar 11: Univariate Analysis (OnePlus 9)*

![image](https://github.com/user-attachments/assets/1df6ae2a-fc5a-40a2-bcfc-6b73f50dd54c)  
*Gambar 12: Univariate Analysis (Samsung Galaxy S21)*

![image](https://github.com/user-attachments/assets/a421f129-4940-419b-963e-b0818ff6b764)  
*Gambar 13: Univariate Analysis (Xiaomi Mi 11)*

![image](https://github.com/user-attachments/assets/6c8aaf7e-3ac9-4ce6-918e-81ae2beb29e1)  
*Gambar 14: Univariate Analysis (iPhone 12)*

Berdasarkan hasil visualisasi **Boxplot dan Histogram** yang ditampilkan, kita dapat menarik beberapa kesimpulan penting mengenai distribusi data dalam dataset ini:

- **Tidak ada outlier pada dataset**: Boxplot menunjukkan bahwa semua nilai berada dalam rentang interkuartil (IQR) yang normal, tanpa adanya nilai ekstrem yang terletak jauh di luar batas atas atau bawah.
  
- **Distribusi data terpusat dengan baik**: Mayoritas data terdistribusi secara simetris, yang berarti bahwa tidak ada kecenderungan data yang menyimpang secara signifikan ke salah satu sisi.

- **Nilai tengah berada di dalam rentang yang diharapkan**: Posisi median yang terlihat di boxplot menunjukkan bahwa nilai tengah data berada dalam kisaran yang wajar, memberikan indikasi bahwa distribusi data tidak terdistorsi.

- **Variabilitas antar data**: Berdasarkan panjang whisker, kita dapat melihat bahwa data memiliki variasi yang cukup seimbang di kedua sisi median, menunjukkan tidak ada ketidakseimbangan signifikan dalam variasi data.

Dari analisis ini, kita dapat menyimpulkan bahwa dataset ini tidak membutuhkan penanganan khusus terkait outlier atau distribusi yang tidak normal, sehingga dapat digunakan lebih lanjut tanpa perlu proses pembersihan data yang rumit.

### **Exploration Data Analysis-Multivariate Analysis**

![image](https://github.com/user-attachments/assets/3235e15f-35b8-46dd-9162-25ecf58b149e)

*Gambar 15: Multivariate Analysis(Matriks korelasi)*

Berdasarkan analisis matriks korelasi yang ditampilkan, kita dapat menyimpulkan beberapa hubungan yang signifikan antar variabel:

1. **Waktu Penggunaan Aplikasi, Waktu Layar, Pengurasan Baterai, Jumlah Aplikasi yang Diinstal, dan Penggunaan Data**: 
   Terdapat korelasi yang kuat dan positif antara variabel-variabel ini. Hal ini menunjukkan bahwa semakin tinggi nilai salah satu variabel, semakin tinggi pula nilai variabel lainnya. Sebagai contoh, pengguna yang menghabiskan lebih banyak waktu untuk menggunakan aplikasi cenderung memiliki **waktu layar yang lebih lama**, **pengurasan baterai yang lebih tinggi**, **jumlah aplikasi yang lebih banyak**, dan **penggunaan data yang lebih tinggi**. Pola ini mengindikasikan bahwa faktor-faktor ini saling mempengaruhi dan dapat digunakan untuk memprediksi perilaku pengguna di masa mendatang.

2. **Waktu Penggunaan Aplikasi, Waktu Layar, dan Pengurasan Baterai dengan Kategori Perilaku Pengguna**: 
   Ditemukan korelasi positif yang cukup kuat antara **waktu penggunaan aplikasi**, **waktu layar**, dan **pengurasan baterai** dengan **kategori perilaku pengguna**. Artinya, pengguna yang memiliki pola penggunaan yang serupa (misalnya dalam hal durasi aplikasi yang digunakan dan pengurasan baterai) cenderung memiliki kategori perilaku pengguna yang serupa pula. Ini memberikan wawasan bahwa pengguna dengan kebiasaan penggunaan serupa mungkin memiliki pola perilaku yang serupa pula dalam hal pengelolaan perangkat dan aplikasinya.

![image](https://github.com/user-attachments/assets/6773e282-c5c1-4d90-87f3-2aa610d76573)

*Gambar 16: Multivariate Analysis(PairPlot)*

### **Interpretasi Analisis Multivariat**

**Pola Umum**

Matriks scatterplot menunjukkan bahwa sebagian besar fitur memiliki korelasi yang sangat lemah atau tidak ada hubungan dengan **User Behavior Class**. Namun, ada beberapa fitur seperti "Operating System" dan "Age" yang menunjukkan sedikit kecenderungan berdasarkan **User Behavior Class**, dengan kemungkinan konsentrasi kelas tertentu di kisaran tertentu.

Sebagian besar fitur lainnya tampaknya tidak memiliki hubungan yang signifikan dengan variabel target.

1. **Operating System**: Terlihat sedikit konsentrasi kelas 1 dan 3 di bagian tengah distribusi dari fitur "Operating System". Meskipun demikian, hubungan ini tidak terlalu kuat.
   
2. **Age**: Ada sedikit indikasi bahwa kelas 2 dan 5 lebih banyak ditemukan di ujung bawah distribusi "Age", meskipun pola ini juga tidak terlalu jelas.
   
3. **Screen Size**: Tidak ada pola atau korelasi yang dapat terlihat antara "Screen Size" dengan **User Behavior Class**.
   
4. **Battery Drain**: Fitur "Battery Drain" terlihat hampir independen dari **User Behavior Class**, tanpa korelasi yang kuat.
   
5. **Number of Apps Installed**: Ada sedikit indikasi bahwa kelas 2 lebih sering muncul di rentang atas distribusi "Number of Apps Installed".
   
6. **Data Usage Efficiency**: Distribusi "Data Usage Efficiency" antar **User Behavior Class** tampaknya acak tanpa pola yang jelas.

## 6. Feature Engineering

Pada bagian ini, dilakukan beberapa langkah untuk mempersiapkan data agar dapat digunakan dalam model machine learning, meliputi pemisahan atribut independen dan dependen, normalisasi data, serta pembagian dataset menjadi data latih dan data uji.

#### **1. Memisahkan Atribut Independen dan Dependen**

Langkah pertama dalam feature engineering adalah memisahkan dataset menjadi dua bagian, yaitu **atribut independen (X)** dan **atribut dependen (Y)**. Atribut dependen dalam proyek ini adalah **User Behavior Class**, yang menjadi target untuk diprediksi, sedangkan atribut independen terdiri dari variabel-variabel yang akan digunakan untuk memprediksi perilaku pengguna.

```python
x = df.drop(columns='User Behavior Class')
y = df['User Behavior Class']
```

Pada kode di atas, `x` berisi seluruh kolom yang ada kecuali kolom **User Behavior Class**, yang kemudian disimpan dalam variabel `y` sebagai target label.

#### **2. Normalisasi Data menggunakan MinMaxScaler**

Normalisasi atau skaling sangat penting untuk memastikan bahwa semua fitur berada dalam skala yang seragam. Hal ini dilakukan untuk menghindari fitur dengan rentang nilai yang lebih besar mendominasi perhitungan model. Dalam hal ini, **MinMaxScaler** digunakan untuk mengubah setiap fitur ke dalam rentang [0, 1].

```python
scalar = MinMaxScaler()
x_scale = scalar.fit_transform(x)
```

Dengan menggunakan `MinMaxScaler`, kita memastikan bahwa setiap kolom dalam data `x` memiliki nilai yang berada dalam rentang yang konsisten, memudahkan model dalam belajar dan meningkatkan performa model.

#### **3. Pembagian Dataset menjadi Data Latih dan Data Uji**

Setelah normalisasi, dataset dibagi menjadi dua bagian: **data latih (training data)** dan **data uji (testing data)**. Pembagian ini bertujuan agar model dapat dilatih pada data latih dan dievaluasi pada data uji yang belum pernah dilihat sebelumnya. Pembagian dilakukan dengan proporsi 80% untuk data latih dan 20% untuk data uji.

```python
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)
```

Selanjutnya, label target **User Behavior Class** yang ada pada `y_train` dan `y_test` dikurangi dengan 1. Hal ini dilakukan karena kelas target pada dataset dimulai dari 1, sedangkan dalam pemodelan machine learning biasanya kelas target dimulai dari 0.

```python
y_train = y_train - 1
y_test = y_test - 1
```
Dengan demikian, dataset siap digunakan untuk tahap pelatihan model, dengan variabel independen yang telah dinormalisasi dan variabel dependen yang sudah disiapkan.

## 7. Modeling

### Algoritma yang Digunakan
Algoritma **Random Forest Classifier** dipilih karena mampu menangani dataset dengan jumlah fitur yang cukup banyak dan bekerja dengan baik pada data klasifikasi.

### Hyperparameter Tuning
Hyperparameter tuning dilakukan untuk mencari kombinasi parameter terbaik dengan menggunakan metode **Grid Search**. Kombinasi parameter yang dicari termasuk:
- **n_estimators**: Jumlah pohon dalam hutan.
- **max_depth**: Kedalaman maksimum dari pohon.
- **min_samples_split**: Jumlah minimum sampel untuk membagi node internal.

## 8. Evaluation and interpretation

### Metrik Evaluasi
Metrik evaluasi yang digunakan adalah:
- **Accuracy**: Mengukur seberapa sering model memberikan prediksi yang benar.
- **Precision**: Mengukur seberapa baik model memprediksi kelas positif dengan benar.
- **Recall**: Mengukur seberapa baik model menangkap semua sampel yang benar-benar positif.
- **F1-Score**: Rata-rata harmonik dari precision dan recall.

### Hasil Proyek
Setelah pelatihan awal, model berhasil mencapai hasil evaluasi yang sangat baik dengan semua metrik bernilai sempurna. Namun, tuning lebih lanjut atau penggunaan algoritma lain seperti **XGBoost** dapat dilakukan untuk memastikan model lebih robust dan tidak overfitting.
