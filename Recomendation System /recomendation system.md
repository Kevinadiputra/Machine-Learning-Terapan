# Laporan Proyek Machine Learning - Kevin Adiputra Mahesa

## Latar Belakang

Di era digital, jumlah informasi yang tersedia terus meningkat dengan sangat cepat. Hal ini menciptakan tantangan besar bagi pengguna untuk menemukan konten yang relevan di tengah pilihan yang begitu banyak. Misalnya, layanan seperti Netflix, Spotify, dan Amazon menghadapi kebutuhan mendesak untuk menyaring pilihan berdasarkan preferensi pengguna untuk memberikan rekomendasi yang tepat.

Netflix, sebagai salah satu platform streaming terkemuka, memanfaatkan sistem rekomendasi berbasis data yang mengintegrasikan pola tontonan dan preferensi pengguna untuk menyajikan rekomendasi yang relevan. Algoritma canggih ini tidak hanya meningkatkan pengalaman pengguna tetapi juga mendorong waktu menonton lebih lama dan memperkuat loyalitas pelanggan.[[1]](https://stratoflow.com/how-netflix-recommendation-algorithm-work/)

Spotify, dalam bidang musik, menggunakan pendekatan serupa dengan fitur seperti *Discover Weekly*. Fitur ini menganalisis kebiasaan mendengarkan pengguna dan menyajikan daftar putar yang disesuaikan, memungkinkan pengguna menemukan musik baru yang sesuai dengan selera mereka. Strategi ini terbukti efektif dalam meningkatkan keterlibatan pengguna dan memperluas cakupan artis serta lagu yang didengarkan.[[2]](https://stratoflow.com/spotify-recommendation-algorithm/)

Dalam dunia e-commerce, Amazon menerapkan *collaborative filtering* untuk merekomendasikan produk berdasarkan riwayat pembelian pengguna dan preferensi pembeli lainnya. Dengan memberikan rekomendasi produk yang relevan, Amazon berhasil meningkatkan tingkat konversi dan kepuasan pelanggan, menjadikan sistem rekomendasi sebagai pilar utama dalam strategi pemasaran digital mereka.[[3]](https://clouddevs.com/go/building-personalized-recommendation-engines/)

### Mengapa Masalah Ini Harus Diselesaikan?

Masalah *information overload* tidak hanya membebani pengguna dalam pengambilan keputusan tetapi juga dapat mengurangi efektivitas platform digital. Jika pengguna merasa kewalahan dengan jumlah informasi yang tidak relevan, mereka cenderung mengurangi waktu interaksi atau bahkan beralih ke platform lain yang menawarkan pengalaman lebih personal. Hal ini dapat berdampak langsung pada pendapatan, loyalitas pelanggan, dan daya saing platform.

Selain itu, pengguna memiliki harapan yang semakin tinggi terhadap pengalaman yang disesuaikan. Sistem rekomendasi yang efisien memungkinkan platform untuk memberikan nilai tambah dengan menghadirkan konten yang sesuai dengan kebutuhan spesifik setiap individu. Dengan demikian, sistem rekomendasi tidak hanya menjadi solusi teknis tetapi juga strategi bisnis utama untuk mempertahankan dan memperluas basis pengguna.

Oleh karena itu, menyelesaikan masalah ini adalah langkah penting untuk menciptakan pengalaman pengguna yang lebih baik, meningkatkan keterlibatan, dan pada akhirnya mendukung pertumbuhan bisnis secara berkelanjutan.

---

## Business Understanding

### Problem Statements

1. **Bagaimana cara memberikan rekomendasi film kepada pengguna yang berdasarkan pada preferensi mereka?**
2. **Bagaimana cara memberikan rekomendasi yang relevan tanpa hanya mengandalkan data rating pengguna?**

### Goals

Tujuan dari proyek ini adalah:
1. Membuat sistem rekomendasi yang memberikan rekomendasi film yang sesuai dengan preferensi pengguna.
2. Membandingkan dua metode rekomendasi (Collaborative Filtering dan Content-Based Filtering) untuk memilih metode yang lebih baik.

### Solution Approach

Untuk mencapai tujuan-tujuan dalam memberikan rekomendasi yang relevan, saya menggunakan dua pendekatan sistem rekomendasi:

1. **Collaborative Filtering**:
   Saya menggunakan algoritma **Singular Value Decomposition (SVD)** untuk melakukan *collaborative filtering*. SVD adalah metode yang efektif untuk menangani masalah sparsity pada data interaksi pengguna, seperti pada rating film atau produk. Dengan menggunakan SVD, kita dapat mendekomposisi matriks preferensi pengguna ke dalam matriks yang lebih kecil, yang memungkinkan kita untuk menemukan pola tersembunyi dalam data dan membuat prediksi mengenai item yang akan disukai oleh pengguna berdasarkan interaksi pengguna lain yang serupa. Pendekatan ini sangat berguna ketika informasi eksplisit tentang item tidak tersedia, tetapi hanya berdasarkan perilaku pengguna sebelumnya (misalnya, rating atau klik).[[4]](https://www.analyticsvidhya.com/blog/2020/12/understanding-singular-value-decomposition/)

2. **Content-Based Filtering**:
   Untuk pendekatan *content-based filtering*, saya menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)** untuk mengukur pentingnya kata-kata dalam deskripsi film atau konten lain yang relevan. Dengan menghitung bobot TF-IDF dari fitur terkait (seperti genre, aktor, atau deskripsi teks lainnya), sistem dapat mengukur kemiripan antara film yang belum ditonton dengan film yang sudah disukai atau ditonton oleh pengguna. Selanjutnya, **cosine similarity** digunakan untuk menghitung kesamaan antara dua vektor TF-IDF, sehingga rekomendasi dapat diberikan berdasarkan kemiripan konten yang telah dilihat. Pendekatan ini sangat efektif ketika kita memiliki informasi konten yang lebih kaya dan dapat mendasarkan rekomendasi pada fitur deskriptif dari item tersebut.[[5]](https://www.geeksforgeeks.org/content-based-filtering-recommender-system-using-python/), [[6]](https://towardsdatascience.com/how-to-build-a-content-based-recommendation-system-using-tf-idf-b419a0424912)

Kedua pendekatan ini dipilih karena mereka saling melengkapi, memberikan solusi yang lebih baik untuk mempersonalisasi pengalaman pengguna dalam memilih konten atau produk yang relevan.

---

## Data Understanding

Pada bagian ini, saya melakukan analisis awal terhadap dataset yang digunakan dalam sistem rekomendasi. Dataset yang digunakan adalah dataset **MovieLens** yang terdiri dari beberapa file terpisah, yaitu *ratings*, *movies*, *links*, dan *tags*. Data ini memberikan informasi tentang film, rating yang diberikan pengguna, dan berbagai metadata terkait film yang ada dalam dataset.

#### Langkah-langkah yang dilakukan dalam Data Understanding:

1. **Data Loading**:
   Pertama-tama, saya memuat dataset *movies* dan *ratings* menggunakan pandas dengan URL yang disediakan. File *movies* berisi informasi tentang film, seperti ID film dan nama film, sementara file *ratings* berisi informasi mengenai rating yang diberikan oleh pengguna untuk setiap film. Dataset ini dapat diakses melalui [MovieLens](https://grouplens.org/datasets/movielens/).

2. **Menampilkan 5 Baris Awal Data**:
   Untuk mendapatkan gambaran awal mengenai data, saya menampilkan 5 baris pertama dari kedua dataset. Ini membantu untuk memverifikasi struktur data dan memastikan bahwa data dimuat dengan benar.
Berikut adalah tabel markdown yang menggambarkan contoh data dari dataset *Movies* dan *Ratings* yang digunakan dalam sistem rekomendasi:

**Tabel Data Movies:**

| movieId | title                              | genres                                             |
|---------|------------------------------------|----------------------------------------------------|
| 1       | Toy Story (1995)                   | Adventure|Animation|Children|Comedy|Fantasy       |
| 2       | Jumanji (1995)                     | Adventure|Children|Fantasy                       |
| 3       | Grumpier Old Men (1995)            | Comedy|Romance                                     |
| 4       | Waiting to Exhale (1995)           | Comedy|Drama|Romance                            |
| 5       | Father of the Bride Part II (1995) | Comedy                                          |

*Tabel 1: Dataset Movies*

**Tabel Data Ratings:**

| userId | movieId | rating | timestamp   |
|--------|---------|--------|-------------|
| 1      | 1       | 4.0    | 964982703   |
| 1      | 3       | 4.0    | 964981247   |
| 1      | 6       | 4.0    | 964982224   |
| 1      | 47      | 5.0    | 964983815   |
| 1      | 50      | 5.0    | 964982931   |

*Tabel 2: Dataset Ratings*

Dengan tabel di atas, kita bisa melihat bagaimana setiap film diwakili oleh ID unik dalam *movieId*, serta informasi terkait genre film tersebut. Sedangkan pada tabel *Ratings*, setiap rating yang diberikan oleh pengguna juga diwakili dengan ID pengguna (*userId*), ID film (*movieId*), rating yang diberikan (*rating*), dan timestamp yang menunjukkan kapan rating tersebut diberikan.

3. **Informasi Dasar Dataset**:
   Selanjutnya, saya memeriksa informasi dasar dari kedua dataset menggunakan `.info()`. Langkah ini memberikan wawasan tentang jumlah kolom, jumlah baris, serta tipe data dari setiap kolom di dalam dataset. Ini penting untuk memverifikasi apakah data sudah terstruktur dengan benar dan apakah tipe data sesuai dengan analisis yang akan dilakukan.

Berikut adalah tabel yang menyajikan informasi dasar mengenai dataset *Movies* dan *Ratings*:

**Tabel Info Dataset Movies:**

| Column  | Non-Null Count | Dtype  |
|---------|----------------|--------|
| movieId | 9742           | int64  |
| title   | 9742           | object |
| genres  | 9742           | object |

*Tabel 3: Informasi Dataset Movies*

Dataset ini memiliki 9742 baris dan 3 kolom, dengan semua nilai pada kolom `movieId`, `title`, dan `genres` tidak memiliki nilai yang hilang (non-null). Kolom `movieId` berisi tipe data integer, sementara `title` dan `genres` berisi tipe data string (object).

**Tabel Info Dataset Ratings:**

| Column   | Non-Null Count | Dtype  |
|----------|----------------|--------|
| userId   | 100836         | int64  |
| movieId  | 100836         | int64  |
| rating   | 100836         | float64|
| timestamp| 100836         | int64  |

*Tabel 4: Informasi Dataset Ratings*

Dataset *Ratings* memiliki 100836 baris dan 4 kolom, dengan semua nilai pada kolom `userId`, `movieId`, `rating`, dan `timestamp` juga tidak memiliki nilai hilang. Kolom `userId`, `movieId`, dan `timestamp` berisi tipe data integer, sedangkan kolom `rating` berisi tipe data float64.

1. **Dataset Movies**:
   - Kolom `movieId` berisi ID unik untuk setiap film.
   - Kolom `title` menyimpan nama film.
   - Kolom `genres` berisi informasi genre film yang terdiri dari satu atau lebih genre yang dipisahkan dengan tanda "|" (misalnya, "Comedy|Drama").

2. **Dataset Ratings**:
   - Kolom `userId` menyimpan ID pengguna yang memberikan rating.
   - Kolom `movieId` mengacu pada ID film yang dinilai oleh pengguna.
   - Kolom `rating` menunjukkan skor rating yang diberikan pengguna untuk film tertentu, dengan skala antara 0 hingga 5.
   - Kolom `timestamp` berisi waktu ketika rating diberikan, dalam format UNIX timestamp.

Dataset ini akan digunakan untuk menganalisis dan membangun sistem rekomendasi berbasis perilaku pengguna dan konten film yang ada.

4. **Cek Missing Values dan Duplikat**:
   Saya juga memeriksa adanya *missing values* dan duplikasi pada kedua dataset. Mengidentifikasi dan menangani data yang hilang atau duplikat sangat penting untuk menjaga kualitas analisis. Berdasarkan hasil pemeriksaan, tidak ditemukan adanya missing values atau duplikasi dalam dataset, sehingga tidak diperlukan penanganan lebih lanjut.

**Missing Values**

Berikut adalah tabel yang menunjukkan jumlah *missing value* pada dataset *Movies* dan *Ratings*:

**Tabel Jumlah Missing Value Dataset Movies:**

| Column  | Missing Value |
|---------|---------------|
| movieId | 0             |
| title   | 0             |
| genres  | 0             |

*Tabel 5: Hasil pengecekkan missing value pada dataset Movies*

**Tabel Jumlah Missing Value Dataset Ratings:**

| Column    | Missing Value |
|-----------|---------------|
| userId    | 0             |
| movieId   | 0             |
| rating    | 0             |
| timestamp | 0             |

*Tabel 6: Hasil pengecekan missing values pada dataset ratings*

- **Dataset Movies**: 
   - Semua kolom (`movieId`, `title`, `genres`) tidak memiliki nilai yang hilang (missing value) pada dataset ini. Hal ini menunjukkan bahwa data yang tersedia lengkap dan tidak memerlukan pembersihan lebih lanjut terkait data yang hilang.

- **Dataset Ratings**:
   - Begitu pula pada dataset *Ratings*, tidak ada nilai yang hilang pada kolom-kolom yang ada (yakni `userId`, `movieId`, `rating`, dan `timestamp`). Dengan demikian, dataset ini juga siap digunakan tanpa perlunya penanganan missing value.

Karena tidak ada missing value pada kedua dataset, kita dapat melanjutkan analisis tanpa perlu melakukan proses imputasi atau pembersihan data terkait nilai yang hilang.

**Duplicate**
Berikut adalah tabel yang menunjukkan jumlah duplikat pada dataset *Movies* dan *Ratings*:

**Tabel Jumlah Duplikat Dataset Movies:**

| Dataset | Jumlah Duplikat |
|---------|------------------|
| Movies  | 0               |

*Tabel 7: Hasil pengecekan duplikasi pada dataset movies*

**Tabel Jumlah Duplikat Dataset Ratings:**

| Dataset | Jumlah Duplikat |
|---------|------------------|
| Ratings | 0               |

*Tabel 8: Hasil pengecekan duplikasi pada dataset ratings*

- **Dataset Movies**:
   - Tidak ditemukan baris data yang duplikat. Artinya, semua entri pada dataset ini unik dan tidak ada data yang perlu dihapus karena duplikasi.

- **Dataset Ratings**:
   - Sama seperti dataset *Movies*, dataset *Ratings* juga tidak mengandung data duplikat. Dengan demikian, tidak ada langkah tambahan yang diperlukan untuk membersihkan data terkait duplikasi.

Dari hasil ini, kedua dataset telah memenuhi syarat untuk digunakan langsung dalam proses analisis atau pembuatan model, tanpa perlu modifikasi lebih lanjut terkait duplikasi data.

### Exploration Data Analysis

#### Deskripsi Statistik Dataset

##### **Deskripsi Statistik Dataset Movies**
| Statistik | MovieId        |
|-----------|----------------|
| Count     | 9,742          |
| Mean      | 42,200.35      |
| Std       | 52,160.49      |
| Min       | 1.00           |
| 25%       | 3,248.25       |
| 50%       | 7,300.00       |
| 75%       | 76,232.00      |
| Max       | 193,609.00     |

*Tabel 9: Deskripsi Statistik Dataset Movies*

---

##### **Deskripsi Statistik Dataset Ratings**
| Statistik | UserId        | MovieId        | Rating         | Timestamp         |
|-----------|---------------|----------------|----------------|-------------------|
| Count     | 100,836       | 100,836        | 100,836        | 100,836           |
| Mean      | 326.13        | 19,435.30      | 3.50           | 1.21e+09          |
| Std       | 182.62        | 35,530.99      | 1.04           | 2.16e+08          |
| Min       | 1.00          | 1.00           | 0.50           | 8.28e+08          |
| 25%       | 177.00        | 1,199.00       | 3.00           | 1.02e+09          |
| 50%       | 325.00        | 2,991.00       | 3.50           | 1.19e+09          |
| 75%       | 477.00        | 8,122.00       | 4.00           | 1.44e+09          |
| Max       | 610.00        | 193,609.00     | 5.00           | 1.54e+09          |

*Tabel 10: Deskripsi Statistik Dataset Ratings*

---

##### Dataset Movies:
- **Count (Jumlah Data):** Dataset *Movies* berisi total 9,742 data.
- **Rentang MovieId:** ID film berkisar antara 1 hingga 193,609, dengan rata-rata ID film 42,200. Hal ini menunjukkan kemungkinan besar terdapat celah pada distribusi ID film.
- **Distribusi Data:** Standar deviasi yang besar (52,160.49) menunjukkan variasi ID film yang cukup luas.

##### Dataset Ratings:
- **Jumlah Data:** Dataset *Ratings* terdiri dari 100,836 data ulasan film.
- **Distribusi Rating:** Rentang nilai rating adalah dari 0.5 hingga 5.0, dengan rata-rata 3.5. Hal ini menunjukkan preferensi pengguna cenderung ke arah rating menengah-atas.
- **UserId dan MovieId:** Data *UserId* dan *MovieId* menunjukkan bahwa ada 610 pengguna unik yang memberikan ulasan terhadap berbagai film.

### Distribusi Rating



## Data Preparation

Sebelum memulai pemodelan, langkah-langkah berikut dilakukan untuk mempersiapkan data:

1. **Penggabungan Dataset**: Saya menggabungkan data `movies.csv` dan `ratings.csv` berdasarkan `movieId` untuk mendapatkan informasi lengkap tentang film dan ratingnya.
   
2. **Penanganan Nilai yang Hilang**: Beberapa data mungkin memiliki nilai yang hilang atau tidak lengkap, oleh karena itu saya memeriksa dan menghapus baris yang tidak memiliki nilai yang lengkap.

3. **Encoding Genre**: Mengubah kolom genre menjadi fitur numerik menggunakan teknik one-hot encoding. Ini memungkinkan model memahami genre sebagai variabel terpisah.

4. **Pembagian Data**: Data dibagi menjadi set pelatihan dan pengujian untuk mengevaluasi model secara efektif.

---

## Modeling

Pada tahap ini, saya mengimplementasikan dua metode sistem rekomendasi:

### 1. **Collaborative Filtering** (Menggunakan KNN)
Model ini mencari pengguna yang memiliki preferensi serupa dengan pengguna yang sedang dianalisis. Rekomendasi diberikan berdasarkan rating yang diberikan oleh pengguna serupa.

### 2. **Content-Based Filtering**
Menggunakan genre film sebagai dasar untuk memberikan rekomendasi. Model ini memberikan film yang serupa dengan film yang telah ditonton oleh pengguna.

---

## Evaluation

### Collaborative Filtering
Untuk evaluasi Collaborative Filtering, saya menggunakan metrik **Root Mean Squared Error (RMSE)**. Nilai RMSE yang lebih rendah menunjukkan model yang lebih akurat dalam memprediksi rating pengguna.

### Content-Based Filtering
Evaluasi Content-Based Filtering didasarkan pada analisis **kemiripan genre**. Rekomendasi yang diberikan oleh model lebih relevan dengan preferensi genre pengguna dan tidak tergantung pada data rating.

---

## Kesimpulan

Setelah melakukan evaluasi, saya dapat menyimpulkan bahwa:
1. **Collaborative Filtering** efektif dalam memberikan rekomendasi yang dipersonalisasi berdasarkan data rating pengguna, namun mengalami masalah **cold start** untuk pengguna baru.
2. **Content-Based Filtering** memberikan rekomendasi yang relevan berdasarkan genre film yang serupa, cocok digunakan dalam situasi dengan data pengguna terbatas.

Metode yang paling tepat bergantung pada konteks aplikasi dan ketersediaan data pengguna.
