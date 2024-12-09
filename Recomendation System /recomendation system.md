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

![image](https://github.com/user-attachments/assets/aaf86776-fa53-4367-aa7a-461a236153cf)

*Gambar 1: Distribusi Rating*

### **Insight Visualisasi Rating Film (Revisi)**

**1. Rating Terbanyak**  
Sebagian besar pengguna memberikan rating di sekitar nilai **4**. Hal ini menunjukkan bahwa mayoritas film dalam dataset berhasil memenuhi ekspektasi pengguna, mencerminkan tingkat kepuasan yang cukup tinggi.

**2. Distribusi Rating**  
Distribusi rating menunjukkan pola **miring ke kanan (right-skewed)**. Artinya, lebih banyak rating berada di sisi nilai tinggi (3-5) dibandingkan dengan nilai rendah (1-2).  
- **Implikasi:** Sebagian besar film dalam dataset dinilai sebagai "bagus" hingga "sangat bagus," menunjukkan kualitas film yang cukup memadai menurut pengguna.

**3. Pola Rating Ekstrem**  
- **Rating Rendah:** Hanya sedikit pengguna memberikan rating di kisaran nilai **1**. Ini mengindikasikan bahwa hanya sedikit film yang dianggap sangat mengecewakan oleh pengguna.  
- **Rating Tinggi:** Rating di kisaran **5** cukup signifikan, tetapi tidak sebanyak rating di sekitar nilai **4**. Hal ini dapat mengindikasikan adanya potensi untuk perbaikan dalam kualitas film agar lebih banyak pengguna memberikan rating sempurna.

**4. Kesimpulan Utama**  
Hasil visualisasi distribusi rating memberikan wawasan bahwa:  
- Mayoritas film berada dalam kategori "memuaskan" hingga "baik."  
- Ada peluang untuk meningkatkan kepuasan pengguna, khususnya dengan fokus pada elemen-elemen yang mendorong lebih banyak rating **5**.

### Top 10 film dengan rating tertinggi

![image](https://github.com/user-attachments/assets/942002ce-e078-453f-b7d4-c8a071b1d71c)

*Gambar 2: Top 10 film dengan rating tertinggi*

Berikut adalah tabel dan insight berdasarkan data Top 10 film dengan rata-rata rating tertinggi:

| **No.** | **Judul Film**                                  | **Rata-rata Rating** |
|---------|------------------------------------------------|-----------------------|
| 1       | Paper Birds (Pájaros de papel) (2010)          | 5.0                   |
| 2       | Act of Killing, The (2012)                     | 5.0                   |
| 3       | Jump In! (2007)                                | 5.0                   |
| 4       | Human (2015)                                   | 5.0                   |
| 5       | L.A. Slasher (2015)                            | 5.0                   |
| 6       | Lady Jane (1986)                               | 5.0                   |
| 7       | Bill Hicks: Revelations (1993)                 | 5.0                   |
| 8       | Justice League: Doom (2012)                    | 5.0                   |
| 9       | Open Hearts (Elsker dig for evigt) (2002)      | 5.0                   |
| 10      | Formula of Love (1984)                         | 5.0                   |

*Table 11: Daftar Top 10 Film dengan Rata-rata Rating Tertinggi*

Berdasarkan visualisasi dan tabel, dapat disimpulkan sebagai berikut: 

1. **Konsistensi Rating Tinggi:**
   - Semua film dalam daftar ini memiliki rata-rata rating sempurna, yaitu **5.0**. Hal ini menunjukkan bahwa film-film tersebut benar-benar disukai oleh semua penggunanya yang memberikan rating.
   - Keberadaan rating sempurna menandakan kualitas yang diakui secara universal oleh komunitas pengguna.

2. **Keanekaragaman Genre dan Tahun Rilis:**
   - Film-film ini berasal dari berbagai tahun rilis, mulai dari tahun **1984** hingga **2015**.
   - Genre film cukup beragam, termasuk dokumenter seperti *The Act of Killing*, animasi superhero seperti *Justice League: Doom*, hingga drama romantis seperti *Open Hearts*.

3. **Indikasi Jumlah Penonton:**
   - Sebagian besar film dengan rating sempurna ini mungkin memiliki basis penonton yang kecil namun sangat puas.
   - Perlu dianalisis lebih lanjut apakah jumlah penonton memengaruhi rata-rata rating sempurna, misalnya dengan memeriksa jumlah total rating yang diberikan.

4. **Relevansi untuk Rekomendasi:**
   - Film-film ini dapat dijadikan pilihan rekomendasi premium untuk pengguna baru yang mencari kualitas terbaik menurut rating pengguna sebelumnya.
   - Platform dapat mempromosikan film-film ini untuk meningkatkan engagement dan kepuasan pengguna.

### Jumlah Rating perfilm

![image](https://github.com/user-attachments/assets/69705ccc-f425-4684-9c40-951b29d3ff47)

*Gambar 3: Distribusi rating perfilm*

Berikut adalah tabel hasil distribusi jumlah rating untuk 10 film dengan jumlah rating terbanyak dan penjelasannya:

| **No.** | **Judul Film**                                  | **Jumlah Rating** |
|---------|------------------------------------------------|--------------------|
| 1       | Forrest Gump (1994)                            | 329                |
| 2       | Shawshank Redemption, The (1994)               | 317                |
| 3       | Pulp Fiction (1994)                            | 307                |
| 4       | Silence of the Lambs, The (1991)               | 279                |
| 5       | Matrix, The (1999)                             | 278                |
| 6       | Star Wars: Episode IV - A New Hope (1977)      | 251                |
| 7       | Jurassic Park (1993)                           | 238                |
| 8       | Braveheart (1995)                              | 237                |
| 9       | Terminator 2: Judgment Day (1991)              | 224                |
| 10      | Schindler's List (1993)                        | 220                |

*Table 12: 10 film dengan rating terbanyak*

1. **Dominasi Film Populer:**
   - Film-film dengan jumlah rating terbanyak adalah film populer dan ikonis, seperti *Forrest Gump* dan *The Shawshank Redemption*. Hal ini mencerminkan pengaruh besar dari popularitas film terhadap banyaknya jumlah rating yang diterima.

2. **Pola Distribusi:**
   - Distribusi jumlah rating menunjukkan pola ekor panjang (long-tail), di mana hanya sedikit film yang memiliki jumlah rating tinggi, sementara mayoritas lainnya memiliki rating yang relatif rendah. Ini mencerminkan perbedaan besar antara film yang sangat dikenal dan film niche atau kurang populer.

3. **Kaitan dengan Tahun Rilis:**
   - Sebagian besar film dalam daftar berasal dari era 1990-an. Ini mungkin mengindikasikan bahwa era tersebut menghasilkan film-film berkualitas tinggi yang hingga saat ini masih dinikmati banyak pengguna.

4. **Kesimpulan Distribusi:**
   - Mayoritas film dalam dataset memiliki jumlah rating yang rendah, mengindikasikan bahwa ada lebih banyak film yang bersifat niche atau memiliki pangsa penonton yang kecil. Namun, film yang berhasil mendapatkan banyak rating biasanya adalah film blockbuster yang ikonis.

### Distribusi Rating user

![image](https://github.com/user-attachments/assets/27e37b56-e9a8-48a6-8040-8d1530a9dc7d)

*Gambar 4: Distribusi Rating per user*

Berikut adalah tabel hasil distribusi rating per pengguna berdasarkan data yang Anda berikan:

| **No.** | **ID Pengguna** | **Jumlah Rating** |
|---------|----------------|--------------------|
| 1       | 414            | 2698              |
| 2       | 599            | 2478              |
| 3       | 474            | 2108              |
| 4       | 448            | 1864              |
| 5       | 274            | 1346              |
| 6       | 610            | 1302              |
| 7       | 68             | 1260              |
| 8       | 380            | 1218              |
| 9       | 606            | 1115              |
| 10      | 288            | 1055              |

*Tabel 13: Distribusi Jumlah Rating per Pengguna dengan Rating Tertinggi*

---

#### **Karakteristik Distribusi**
1. **Distribusi Asimetris:**
   - Sebagian besar pengguna memiliki jumlah rating yang jauh lebih sedikit dibandingkan pengguna yang aktif, menciptakan bentuk distribusi asimetris dengan ekor panjang ke kanan. Pengguna yang sangat aktif secara signifikan memengaruhi total distribusi.

2. **Ekor Kanan:**
   - Pengguna super aktif, seperti pengguna dengan ID 414 (memberikan 2698 rating), menjadi anomali dalam dataset, dengan kontribusi yang sangat besar terhadap data rating keseluruhan.

---

#### **Implikasi Data**
1. **Mayoritas Pengguna Pasif:**
   - Sebagian besar pengguna memberikan sedikit sekali rating. Ini mungkin mengindikasikan:
     - Ketidakaktifan mereka pada platform.
     - Kurangnya antusiasme untuk memberikan penilaian pada film yang telah mereka tonton.
   
2. **Pengguna Super Aktif:**
   - Segmen kecil pengguna yang memberikan jumlah rating tinggi, seperti ID 414 atau 599, menunjukkan keterlibatan yang sangat baik. Pengguna ini bisa dimanfaatkan sebagai:
     - Sumber umpan balik untuk peningkatan kualitas rekomendasi.
     - Fokus analisis untuk memahami kebiasaan pengguna aktif.

### Korelasi jumlah rating dan rating rata-rata

![image](https://github.com/user-attachments/assets/faa9fa08-9baf-42d8-b0a6-2fb0a8a51d45)

*Gambar 5: Korelasi jumlah raiting dan rating rata-rata*

### **Analisis Korelasi: Jumlah Rating vs. Rating Rata-rata**

#### **Deskripsi Korelasi**
Hasil visualisasi korelasi menunjukkan hubungan antara **jumlah rating yang diterima suatu film** dengan **rating rata-rata yang diberikan oleh pengguna**. Korelasi yang ditemukan bersifat **positif lemah**, mengindikasikan bahwa walaupun terdapat kecenderungan, hubungan antara kedua variabel ini tidak terlalu kuat.

---

#### **Temuan Utama**
1. **Tren Umum**  
   - **Kecenderungan Positif:** Data menunjukkan bahwa film dengan jumlah rating lebih banyak cenderung memiliki rating rata-rata yang lebih tinggi. Hal ini dapat diartikan bahwa popularitas film sering kali berbanding lurus dengan persepsi kualitasnya di mata pengguna.  
   - **Indikasi Kepercayaan:** Banyaknya rating juga dapat meningkatkan kepercayaan pengguna lain terhadap nilai rata-rata, yang sering dianggap sebagai cerminan kualitas produk.

2. **Sebaran Data**  
   - Sebagian besar film dalam dataset hanya memiliki sedikit jumlah rating, dengan konsentrasi data di wilayah jumlah rating rendah. Hal ini menunjukkan bahwa banyak film yang mungkin kurang populer atau kurang dikenal oleh pengguna.

3. **Pengecualian**  
   - **Outlier:** Beberapa film dengan jumlah rating tinggi memiliki rating rata-rata yang rendah, atau sebaliknya. Outlier ini mengindikasikan bahwa ada faktor lain, seperti:
     - **Kualitas Subjektif:** Film tertentu mungkin menarik bagi segmen pengguna tertentu tetapi tidak disukai oleh yang lain.  
     - **Bias Rating:** Pengguna yang sangat menyukai atau tidak menyukai film tersebut mungkin memberikan rating ekstrem.
     - **Pengaruh Waktu:** Jumlah rating bisa meningkat seiring waktu, tetapi rata-rata rating bisa dipengaruhi oleh ulasan awal yang dominan.

### **Kesimpulan EDA**

Berdasarkan eksplorasi data yang dilakukan, berikut adalah kesimpulan utama yang dapat diambil:

1. **Distribusi Rating Film**
   - Sebagian besar pengguna memberikan rating di sekitar nilai **4**, menunjukkan bahwa mayoritas film dinilai cukup memuaskan.
   - Rating tinggi (5) dan rendah (1) jarang terjadi, menunjukkan film-film dalam dataset memiliki kualitas yang baik secara keseluruhan.

2. **Popularitas Film**
   - Sebagian besar film hanya memiliki sedikit jumlah rating, dengan distribusi jumlah rating menunjukkan pola **ekor panjang ke kanan**.
   - Film-film yang sangat populer, seperti *Forrest Gump* dan *Shawshank Redemption*, memiliki jumlah rating yang jauh lebih tinggi dibandingkan film lainnya, menunjukkan perbedaan yang signifikan dalam popularitas.

3. **Distribusi Rating per Pengguna**
   - Sebagian besar pengguna hanya memberikan sedikit rating, dengan sebagian kecil pengguna yang sangat aktif (*super users*). Pengguna super aktif ini memiliki kontribusi besar terhadap total rating.

4. **Korelasi Jumlah Rating dan Rating Rata-rata**
   - Terdapat korelasi positif yang lemah antara jumlah rating dan rata-rata rating. Hal ini mengindikasikan bahwa film yang lebih populer cenderung mendapatkan rating rata-rata yang lebih tinggi, meskipun terdapat pengecualian pada beberapa film.

5. **Film dengan Rating Tinggi**
   - Film dengan rata-rata rating tertinggi sering kali adalah film-film niche atau independen yang memiliki daya tarik spesifik terhadap kelompok kecil pengguna, seperti *Paper Birds* atau *Act of Killing*.

## Data Preparation

### **Tahap Persiapan Data untuk Model Collaborative Filtering**

Pada tahap ini, data dipersiapkan untuk melatih dan mengevaluasi model **Collaborative Filtering** menggunakan **Surprise Library**. Langkah-langkah yang dilakukan adalah sebagai berikut:

#### **1. Format Data**
Untuk memastikan data dapat diproses dengan baik oleh Surprise, dilakukan penyesuaian format sebagai berikut:
- **Mengatur Skala Rating**  
  Menggunakan objek `Reader` untuk menentukan skala rating dalam dataset, yaitu dari **0.5 hingga 5.0**. Langkah ini penting agar model memahami struktur data dan rentang nilai rating yang valid.
- **Konversi DataFrame ke Format Surprise**  
  Data rating yang tersimpan dalam DataFrame dikonversi ke format internal Surprise menggunakan fungsi `Dataset.load_from_df`. Format ini diperlukan agar model dapat membaca hubungan antara **pengguna**, **film**, dan **rating** secara optimal.

#### **2. Pembagian Data**
Data dibagi menjadi dua bagian utama untuk memastikan proses pelatihan dan evaluasi model berjalan sesuai:
- **Training Set (80%)**  
  Digunakan untuk melatih model, yaitu mempelajari pola preferensi pengguna berdasarkan data rating historis.
- **Testing Set (20%)**  
  Digunakan untuk menguji kemampuan model memprediksi rating pada data yang belum pernah dilihat selama pelatihan. Evaluasi ini dilakukan untuk mengukur kemampuan generalisasi model.

Proses pembagian data dilakukan menggunakan fungsi `train_test_split`.  
- **Tujuan:**  
  - Menghindari **overfitting**, yaitu ketika model terlalu spesifik pada data pelatihan dan gagal bekerja dengan baik pada data baru.  
  - Menilai performa model pada data yang tidak digunakan dalam proses pelatihan.

### **Transformasi Data untuk Content-Based Filtering**

Pada tahap ini, dilakukan persiapan data untuk model **Content-Based Filtering**, khususnya dengan fokus pada kolom `genres` dalam dataset **movies**. Berikut tabel hasil transformasi:

| **movieId** | **title**                                | **genres**                                   |
|-------------|------------------------------------------|----------------------------------------------|
| 1           | Toy Story (1995)                        | Adventure Animation Children Comedy Fantasy |
| 2           | Jumanji (1995)                          | Adventure Children Fantasy                   |
| 3           | Grumpier Old Men (1995)                 | Comedy Romance                               |
| 4           | Waiting to Exhale (1995)                | Comedy Drama Romance                         |
| 5           | Father of the Bride Part II (1995)      | Comedy                                       |

*Tabel 14: Hasil transformasi data untuk kolom genres*

1. **Transformasi Karakter Pemisah**  
   - Kolom `genres` sebelumnya menggunakan karakter pemisah `|` (contoh: `Adventure|Animation|Children|Comedy|Fantasy`).
   - Karakter tersebut diganti menjadi spasi menggunakan metode `str.replace('|', ' ')`, menghasilkan format seperti `Adventure Animation Children Comedy Fantasy`.

2. **Tujuan Transformasi**
   - **Peningkatan Kualitas Data**: Menghilangkan karakter pemisah yang tidak diperlukan sehingga mempermudah pemrosesan teks.
   - **Kemudahan Analisis Teks**: Format data ini lebih cocok untuk metode berbasis teks, seperti **TF-IDF** (Term Frequency-Inverse Document Frequency), yang digunakan untuk merepresentasikan teks sebagai vektor fitur.
   - **Konsistensi Format**: Memastikan format teks seragam agar model dapat memproses data secara efisien dan akurat.

#### **Keuntungan Transformasi**
- **Kemudahan Pemrosesan**: Format teks yang bersih mendukung pipeline analisis data lebih efektif.
- **Interpretasi Lebih Baik**: Model dapat mengenali hubungan antar genre dengan lebih baik, mendukung pembuatan rekomendasi yang relevan.

---

## Modeling

### **Tahap Modeling: Collaborative Filtering dengan SVD**

Pada tahap ini, dilakukan implementasi model **Collaborative Filtering** menggunakan algoritma **Singular Value Decomposition (SVD)** dari pustaka **Surprise**. 

#### **Pemilihan Model: SVD**
Model **SVD** dipilih karena merupakan algoritma berbasis faktorisasi matriks yang mampu menangani data yang jarang dan menghasilkan rekomendasi personal. Alasan pemilihannya adalah:
1. **Efektivitas**: SVD menguraikan hubungan antara pengguna dan item ke dalam dimensi laten, memungkinkan analisis yang lebih mendalam meskipun data matriks tidak lengkap.
2. **Akurasi**: Algoritma ini sering kali menghasilkan prediksi yang lebih akurat dibandingkan metode berbasis heuristik.
3. **Skalabilitas**: Dengan optimisasi algoritma, SVD dapat menangani dataset besar dengan efisiensi yang baik.

---

#### **Parameter yang Digunakan**
Dalam proyek ini, digunakan **parameter default** dari pustaka **Surprise**, yang meliputi:
- `n_factors`: Dimensi laten (default: 100).
- `n_epochs`: Jumlah iterasi pelatihan (default: 20).
- `lr_all`: Learning rate (default: 0.005).
- `reg_all`: Regularisasi untuk mencegah overfitting (default: 0.02).  
Penggunaan parameter default memastikan model dapat digunakan dengan konfigurasi standar tanpa penyesuaian tambahan.

---

#### **Proses Modeling**
1. **Pelatihan Model**  
   Model dilatih menggunakan **training set** untuk mempelajari pola hubungan laten antara pengguna dan item berdasarkan data rating yang tersedia.

2. **Evaluasi Model**  
   Model diuji pada **testing set** untuk mengukur performa prediksi. Metode evaluasi menggunakan **Mean Absolute Error (MAE)**, yang menghitung rata-rata selisih absolut antara prediksi dan nilai aktual.  

3. **Fungsi Rekomendasi**  
   - Item yang belum dirating oleh pengguna diidentifikasi.  
   - Model memprediksi rating untuk item-item tersebut.  
   - Item dengan prediksi rating tertinggi direkomendasikan kepada pengguna.

---

#### **Kelebihan**
1. **Penanganan Missing Values**: SVD dirancang untuk bekerja pada matriks data yang jarang, sehingga dapat menangani kasus di mana sebagian besar pengguna belum memberikan rating.
2. **Akurasi yang Baik**: Algoritma ini menghasilkan prediksi yang sering kali lebih akurat dibandingkan pendekatan non-faktorisasi.
3. **Generalitas**: SVD dapat digunakan untuk berbagai jenis dataset, tidak hanya data rating tetapi juga data lain yang bersifat matriks.
4. **Efisiensi**: Dengan implementasi optimasi, SVD dapat menangani dataset besar tanpa membutuhkan terlalu banyak memori.

---

#### **Kekurangan**
1. **Ketergantungan pada Parameter**: Performa model sangat bergantung pada pemilihan parameter seperti `n_factors`, `lr_all`, dan `reg_all`. Pengaturan yang kurang tepat dapat menyebabkan underfitting atau overfitting.
2. **Tidak Cocok untuk Dataset Sangat Kecil**: Pada dataset dengan jumlah pengguna atau item yang sangat sedikit, algoritma SVD mungkin kurang efektif karena tidak cukup data untuk pelatihan dimensi laten.
3. **Kompleksitas Komputasi**: Meskipun efisien, SVD tetap memiliki kompleksitas yang lebih tinggi dibandingkan algoritma berbasis heuristik.
4. **Tidak Menangani Data Kontekstual**: SVD tidak mempertimbangkan informasi tambahan seperti waktu atau lokasi, yang bisa relevan dalam beberapa aplikasi rekomendasi.

Dengan kelebihan dan kekurangannya, SVD adalah pilihan yang ideal untuk sistem rekomendasi berbasis data rating, terutama jika dataset yang digunakan cukup besar dan memiliki tingkat sparsity tinggi. Implementasi default memungkinkan pendekatan baseline yang baik sebelum dilakukan penyesuaian parameter lebih lanjut.

### **Rekomendasi untuk User 2**

Berdasarkan model **Collaborative Filtering** menggunakan algoritma **SVD**, berikut adalah **daftar rekomendasi** yang diberikan untuk **User 2** beserta rating yang diprediksi untuk setiap film:

1. **Forrest Gump (1994)** - **Predicted Rating: 4.55**
2. **Singin' in the Rain (1952)** - **Predicted Rating: 4.54**
3. **Lawrence of Arabia (1962)** - **Predicted Rating: 4.48**
4. **Das Boot (1981)** - **Predicted Rating: 4.47**
5. **Amadeus (1984)** - **Predicted Rating: 4.45**

Model **Collaborative Filtering** yang menggunakan **SVD** memprediksi rating untuk film yang belum pernah ditonton oleh pengguna berdasarkan pola perilaku pengguna lain yang mirip. Dalam hal ini, untuk **User 2**, sistem memberikan rekomendasi film dengan **rating yang diprediksi tinggi**, yang berarti bahwa film-film ini kemungkinan akan disukai oleh User 2, berdasarkan preferensi pengguna yang memiliki pola rating yang serupa.

- **Forrest Gump (1994)**, yang memiliki prediksi rating tertinggi (4.55), mungkin merupakan film yang sangat sesuai dengan preferensi User 2 berdasarkan sejarah rating mereka.
- **Singin' in the Rain (1952)** dan **Lawrence of Arabia (1962)** adalah film klasik yang mendapatkan rating prediksi tinggi, menunjukkan bahwa User 2 cenderung menikmati film-film dengan genre atau era serupa.
- **Das Boot (1981)** dan **Amadeus (1984)** juga masuk dalam rekomendasi, menunjukkan bahwa model mempertimbangkan variasi genre (drama perang dan biografi musik) yang mungkin relevan bagi User 2.

### **Content-Based Filtering**

Pada tahap ini, **Content-Based Filtering** digunakan untuk memberikan rekomendasi berdasarkan kesamaan fitur konten, seperti genre film. Berikut adalah tahapan yang dilakukan dalam model ini:

1. **Vektorisasi Genre Menggunakan TF-IDF**  
   Proses pertama adalah melakukan **vektorisasi** pada kolom `genres` menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**. TF-IDF adalah teknik yang digunakan untuk mengubah teks menjadi representasi numerik yang lebih mudah dipahami oleh model. Dalam hal ini, genre setiap film diubah menjadi vektor yang menggambarkan frekuensi relatif dari kata-kata dalam genre tersebut. Berikut adalah hasil matriks **TF-IDF** setelah proses vektorisasi:

| **Film ke-** | **TF-IDF Genre 1** | **TF-IDF Genre 2** | **TF-IDF Genre 3** | **...** | **TF-IDF Genre n** |
|--------------|---------------------|---------------------|---------------------|---------|---------------------|
| 1            | 0.0000              | 0.4168              | 0.5162              | ...     | 0.0000              |
| 2            | 0.0000              | 0.5124              | 0.0000              | ...     | 0.0000              |
| 3            | 0.0000              | 0.0000              | 0.0000              | ...     | 0.0000              |
| ...          | ...                 | ...                 | ...                 | ...     | ...                 |
| m            | 0.5786              | 0.0000              | 0.8156              | ...     | 0.0000              |

*Tabel 15: Hasil Vektroisasi TF-IDF*

   Matriks ini memungkinkan sistem untuk memahami hubungan antara genre satu film dengan genre lainnya dalam bentuk angka, yang kemudian digunakan untuk menghitung kesamaan antar film.

2. **Penghitungan Cosine Similarity**  
   Setelah matriks TF-IDF diperoleh, tahap selanjutnya adalah menghitung **cosine similarity**, yaitu ukuran yang digunakan untuk menilai sejauh mana dua vektor memiliki arah yang sama, dengan nilai antara -1 hingga 1. Nilai lebih dekat ke 1 menunjukkan kesamaan yang lebih tinggi. Hasil **cosine similarity** ini akan membentuk matriks kemiripan antara film satu dengan film lainnya berdasarkan genre.

   Berikut adalah hasil **cosine similarity** untuk beberapa film:

| **Film ke-** | **1**     | **2**     | **3**     | **...** | **m**     |
|--------------|-----------|-----------|-----------|---------|-----------|
| 1            | 1.0000    | 0.8136    | 0.1528    | ...     | 0.2676    |
| 2            | 0.8136    | 1.0000    | 0.0000    | ...     | 0.0000    |
| 3            | 0.1528    | 0.0000    | 1.0000    | ...     | 0.5709    |
| ...          | ...       | ...       | ...       | ...     | ...       |
| m            | 0.2676    | 0.0000    | 0.5709    | ...     | 1.0000    |

*Tabel 16: Hasil Cosine similarity*

- Matriks ini mengukur tingkat kesamaan antar film berdasarkan vektor TF-IDF dari genre mereka.
- Nilai berkisar antara **0** (tidak mirip) hingga **1** (identik).
- Matriks ini digunakan untuk menemukan film yang paling mirip dengan film input.

3. **Rekomendasi Berbasis Konten**  
   Fungsi **content_based_recommendation** digunakan untuk memberikan rekomendasi film berdasarkan kesamaan genre. Fungsi ini akan menerima **judul film** sebagai input dan memberikan **n rekomendasi teratas** berdasarkan kemiripan genre. Prosesnya adalah sebagai berikut:
   
   - Menentukan indeks film berdasarkan judul yang diberikan.
   - Mengambil skor kemiripan (similarity scores) dengan film lainnya.
   - Menyusun skor kemiripan tersebut, mengurutkannya dari yang tertinggi, dan mengambil film-film yang memiliki skor kemiripan tertinggi.
   
   berikut output untuk rekomendasi berbasis konten untuk film *"Toy Story (1995)"* adalah:
   ```
   1. Jumanji (1995)
   2. Grumpier Old Men (1995)
   3. Waiting to Exhale (1995)
   4. Father of the Bride Part II (1995)
   5. Toy Story 2 (1999)
   ```

### **Kelebihan dan Kekurangan Content-Based Filtering**
#### Kelebihan:
- **Rekomendasi Tepat Berdasarkan Konten**: Sistem ini sangat efektif dalam memberikan rekomendasi berdasarkan kesamaan genre atau atribut lainnya, yang memungkinkan film dengan tema yang mirip dapat direkomendasikan.
- **Tidak Bergantung pada Pengguna Lain**: Berbeda dengan collaborative filtering, metode ini tidak membutuhkan data dari pengguna lain untuk menghasilkan rekomendasi, menjadikannya lebih pribadi.
  
#### Kekurangan:
- **Rekomendasi Terbatas**: Karena bergantung hanya pada genre atau fitur konten lainnya, sistem ini dapat memberikan rekomendasi yang terbatas pada kategori yang serupa dan mungkin tidak menyarankan variasi atau genre baru.
- **Masalah dengan Film Baru**: Sistem ini memerlukan data tentang genre film untuk memberikan rekomendasi, sehingga film baru yang belum memiliki banyak informasi genre mungkin tidak bisa mendapatkan rekomendasi yang optimal.


---

## Evaluation

### Evaluasi *Content-Based Filtering*

Untuk evaluasi Collaborative Filtering, saya menggunakan metrik **Root Mean Squared Error (RMSE) dan Mean Absolute Error(MAE)**. Nilai RMSE yang lebih rendah menunjukkan model yang lebih akurat dalam memprediksi rating pengguna.

### Hasil Evaluasi Model Collaborative Filtering

| **Metrik**        | **Fold 1** | **Fold 2** | **Fold 3** | **Fold 4** | **Fold 5** | **Mean** | **Std Dev** |
|--------------------|------------|------------|------------|------------|------------|-----------|-------------|
| **RMSE (testset)** | 0.8741     | 0.8799     | 0.8659     | 0.8686     | 0.8754     | 0.8727    | 0.0050      |
| **MAE (testset)**  | 0.6689     | 0.6752     | 0.6680     | 0.6689     | 0.6728     | 0.6708    | 0.0028      |

*Tabel 17: hasil evaluasi model collaborative filtering*

### Waktu Eksekusi

| **Proses**         | **Fold 1** | **Fold 2** | **Fold 3** | **Fold 4** | **Fold 5** | **Mean** | **Std Dev** |
|---------------------|------------|------------|------------|------------|------------|-----------|-------------|
| **Fit Time (detik)** | 4.32       | 3.92       | 1.55       | 1.62       | 1.60       | 2.60      | 1.25        |
| **Test Time (detik)**| 0.42       | 0.21       | 0.11       | 0.27       | 0.13       | 0.23      | 0.11        |

*Tabel 18: Waktu eksekusi*

### Ringkasan

| **Metrik**        | **Nilai**  |
|--------------------|------------|
| **Mean RMSE**      | 0.8727     |
| **Mean MAE**       | 0.6708     |

*Tabel 19: Ringkasan Hasil evaluasi*

### **Rumus Evaluasi**
1. **Root Mean Square Error (RMSE)**:
   $\[
   RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2}
   \]$
   - **N**: Jumlah data (prediksi).
   - $\(\hat{y}_i\)$: Prediksi model untuk data ke- $\(i\)$.
   - $\(y_i\)$: Nilai sebenarnya untuk data ke- $\(i\)$.

2. **Mean Absolute Error (MAE)**:
   $\[
   MAE = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|
   \]$
   - Sama dengan RMSE, tetapi menggunakan nilai absolut perbedaan.

### **Alasan Penggunaan RMSE dan MAE**
- **RMSE** memberikan bobot lebih besar pada kesalahan besar karena menggunakan kuadrat dari selisih, sehingga lebih sensitif terhadap outlier.
- **MAE** lebih sederhana dan tidak terlalu dipengaruhi oleh outlier, sehingga memberikan gambaran rata-rata kesalahan secara langsung.
- Dengan menggunakan kedua metrik, kita dapat memperoleh gambaran lengkap tentang performa model dalam memprediksi rating.

---

### **Interpretasi Hasil Evaluasi**
#### **RMSE (Cross-Validation)**
- **Mean RMSE**: 0.8727
  - RMSE menunjukkan rata-rata akar kuadrat kesalahan antara prediksi dan nilai sebenarnya.
  - Nilai RMSE yang mendekati 0 menunjukkan performa prediksi yang baik.
  - Dalam kasus ini, RMSE sebesar **0.8727** menunjukkan bahwa rata-rata prediksi model memiliki kesalahan sebesar 0.8727 (skala rating biasanya 1–5).

#### **MAE (Cross-Validation)**
- **Mean MAE**: 0.6708
  - MAE menunjukkan rata-rata kesalahan absolut antara prediksi dan nilai sebenarnya.
  - Dengan nilai MAE sebesar **0.6708**, model menghasilkan kesalahan rata-rata sekitar 0.67 per prediksi.

#### **Variabilitas**
- **Standar Deviasi (Std)**:
  - **RMSE Std**: 0.0050
  - **MAE Std**: 0.0028
  - Standar deviasi yang kecil menunjukkan bahwa performa model konsisten pada setiap lipatan validasi.

#### **Waktu Eksekusi**
- **Waktu Pelatihan**: Rata-rata 2.60 detik.
- **Waktu Pengujian**: Rata-rata 0.23 detik.
  - Model dapat digunakan dengan efisiensi waktu yang baik untuk proses pelatihan dan prediksi.

---

### **Kesimpulan**
- Model **SVD** yang digunakan memiliki performa baik dengan RMSE < 1, yang menunjukkan tingkat akurasi prediksi tinggi.
- **RMSE** lebih sensitif terhadap kesalahan besar, sedangkan **MAE** memberikan nilai rata-rata kesalahan absolut. Kedua metrik menunjukkan kesalahan prediksi yang relatif kecil.
- Konsistensi performa ditunjukkan oleh rendahnya standar deviasi dari hasil cross-validation.

Hasil ini menunjukkan bahwa model cukup andal untuk merekomendasikan item berdasarkan preferensi pengguna.

Berikut adalah penjelasan untuk laporan evaluasi *content-based filtering* yang Anda lakukan:

---

### Evaluasi *Content-Based Filtering*

#### 1. **Tujuan Evaluasi**
Evaluasi bertujuan untuk mengukur tingkat kesesuaian genre antara film asli dengan film yang direkomendasikan oleh model *content-based filtering*. Pendekatan ini dilakukan dengan menghitung persentase kesamaan genre (genre similarity) dari film rekomendasi terhadap film yang dijadikan acuan.

#### 2. **Metode Evaluasi**
- **Dataset yang Digunakan:** 
  - Film asli yang dianalisis: *Toy Story (1995)*.
  - Film yang direkomendasikan: Lima film dengan skor kemiripan tertinggi terhadap film asli.
- **Langkah Evaluasi:**
  1. **Ekstraksi Genre:**
     - Genre dari film asli diambil dan dipisahkan menjadi sebuah himpunan (*set*) untuk mempermudah perbandingan.
     - Genre dari setiap film yang direkomendasikan juga diambil dalam bentuk himpunan.
  2. **Perhitungan Kesamaan:**
     - Menghitung jumlah genre yang sama antara film asli dan film yang direkomendasikan (irisan atau *intersection* dari dua himpunan).
     - Menghitung persentase kesamaan dengan rumus:
       $\[
       \text{Kesamaan} = \frac{\text{Jumlah Genre yang Sama}}{\text{Jumlah Genre Film Asli}} \times 100
       \]$
  3. **Rata-rata Kesamaan:**
     - Setelah menghitung kesamaan untuk semua film rekomendasi, diambil rata-rata dari semua skor kesamaan tersebut.

#### 3. **Hasil Evaluasi**
Berikut adalah hasil perhitungan:

| **Film Rekomendasi**                    | **Genre**                                      | **Kesamaan Genre (%)** |
|------------------------------------------|-----------------------------------------------|-------------------------|
| *Antz (1998)*                            | *Adventure, Animation, Children, Comedy, Fantasy* | 100.00%                |
| *Toy Story 2 (1999)*                     | *Adventure, Animation, Children, Comedy, Fantasy* | 100.00%                |
| *Adventures of Rocky and Bullwinkle (2000)* | *Adventure, Animation, Children, Comedy, Fantasy* | 100.00%                |
| *Emperor's New Groove (2000)*            | *Adventure, Animation, Children, Comedy, Fantasy* | 100.00%                |
| *Monsters, Inc. (2001)*                  | *Adventure, Animation, Children, Comedy, Fantasy* | 100.00%                |

*Tabel 20: Hasil Evaluasi content base Filtering*

**Rata-rata Kesamaan Genre:** 100.00%

#### 4. **Analisis Hasil**
- **Kesesuaian Genre:**
  Semua film rekomendasi memiliki tingkat kesamaan genre sebesar 100% dengan film asli (*Toy Story (1995)*). Hal ini menunjukkan bahwa model bekerja dengan sangat baik dalam merekomendasikan film-film dengan genre yang sama.
- **Kekuatan Model:**
  - Model berhasil menjaga konsistensi rekomendasi berdasarkan informasi genre yang tersedia.
  - Genre seperti *Adventure, Animation, Children, Comedy, Fantasy* yang dimiliki film asli juga ditemukan di semua film rekomendasi.
- **Keterbatasan Model:**
  - Model hanya mempertimbangkan informasi genre sebagai dasar rekomendasi. Ini dapat menyebabkan model mengabaikan faktor lain seperti popularitas, rating, atau preferensi pengguna yang lebih kompleks.

#### 5. **Kesimpulan**
Hasil evaluasi menunjukkan bahwa model *content-based filtering* sangat efektif dalam merekomendasikan film dengan genre yang sama. Namun, untuk meningkatkan relevansi rekomendasi, perlu mempertimbangkan faktor tambahan di luar genre seperti preferensi pengguna atau ulasan film.

## Kesimpulan

Berdasarkan hasil evaluasi dan pencapaian *goals* yang telah ditetapkan, berikut adalah kesimpulan dari proyek ini:

1. **Pembuatan Sistem Rekomendasi**  
   - Sistem rekomendasi berhasil dikembangkan menggunakan dua pendekatan: *Collaborative Filtering* dan *Content-Based Filtering*.
   - *Content-Based Filtering* memberikan rekomendasi film dengan kesesuaian genre yang sangat tinggi (100% rata-rata kesamaan genre untuk film uji). Hal ini menunjukkan keunggulannya dalam memberikan rekomendasi berbasis informasi eksplisit seperti genre film.  
   - *Collaborative Filtering* memberikan hasil yang lebih kompleks dengan evaluasi kuantitatif menggunakan RMSE dan MAE dari data pengguna terhadap preferensi film. Model ini menunjukkan performa baik dengan nilai RMSE rata-rata 0.8727 dan MAE rata-rata 0.6708, menandakan tingkat kesalahan prediksi yang cukup rendah.

2. **Perbandingan Metode Rekomendasi**  
   - *Content-Based Filtering* sangat efektif dalam memastikan film yang direkomendasikan memiliki kesamaan fitur eksplisit dengan film yang telah ditonton pengguna, tetapi terbatas pada informasi metadata seperti genre. Metode ini tidak mampu menangkap preferensi individu yang lebih kompleks.  
   - *Collaborative Filtering* lebih unggul dalam menangkap pola preferensi berbasis data perilaku pengguna secara kolektif. Metode ini lebih cocok untuk skenario di mana data interaksi pengguna tersedia dalam jumlah besar dan beragam.  

3. **Pilihan Metode Terbaik**  
   - Jika tujuan utamanya adalah memberikan rekomendasi berbasis kesamaan fitur film, *Content-Based Filtering* adalah pilihan yang tepat.  
   - Jika fokusnya adalah pada personalisasi berdasarkan pola preferensi pengguna, *Collaborative Filtering* menawarkan solusi yang lebih relevan dan fleksibel.

---

##Daftar Pustaka

1. **How Netflix Recommendation Algorithm Works**  
   Stratoflow. (n.d.). *How Netflix Recommendation Algorithm Works*. Retrieved from [https://stratoflow.com/how-netflix-recommendation-algorithm-work/](https://stratoflow.com/how-netflix-recommendation-algorithm-work/)

2. **Spotify Recommendation Algorithm**  
   Stratoflow. (n.d.). *Spotify Recommendation Algorithm*. Retrieved from [https://stratoflow.com/spotify-recommendation-algorithm/](https://stratoflow.com/spotify-recommendation-algorithm/)

3. **Amazon Recommendation Engine**  
   Clouddevs. (n.d.). *Building Personalized Recommendation Engines*. Retrieved from [https://clouddevs.com/go/building-personalized-recommendation-engines/](https://clouddevs.com/go/building-personalized-recommendation-engines/)

4. **Understanding Singular Value Decomposition**  
   Analytics Vidhya. (2020). *Understanding Singular Value Decomposition*. Retrieved from [https://www.analyticsvidhya.com/blog/2020/12/understanding-singular-value-decomposition/](https://www.analyticsvidhya.com/blog/2020/12/understanding-singular-value-decomposition/)

5. **Content-Based Filtering Using TF-IDF**  
   - GeeksforGeeks. (n.d.). *Content-Based Filtering Recommender System Using Python*. Retrieved from [https://www.geeksforgeeks.org/content-based-filtering-recommender-system-using-python/](https://www.geeksforgeeks.org/content-based-filtering-recommender-system-using-python/)  
   - Towards Data Science. (n.d.). *How to Build a Content-Based Recommendation System Using TF-IDF*. Retrieved from [https://towardsdatascience.com/how-to-build-a-content-based-recommendation-system-using-tf-idf-b419a0424912](https://towardsdatascience.com/how-to-build-a-content-based-recommendation-system-using-tf-idf-b419a0424912)
