# Laporan Proyek Machine Learning - Kevin Adiputra Mahesa

## Latar Belakang

Proyek ini bertujuan untuk membangun sistem rekomendasi film yang dapat memberikan rekomendasi kepada pengguna berdasarkan preferensi mereka. Rekomendasi film sangat berguna di berbagai platform, seperti Netflix, Hulu, dan platform streaming lainnya, untuk meningkatkan pengalaman pengguna.

Sistem rekomendasi dibagi menjadi dua jenis pendekatan utama:
- **Collaborative Filtering**: Menggunakan data dari perilaku pengguna lain untuk memprediksi preferensi pengguna.
- **Content-Based Filtering**: Menggunakan informasi terkait item (dalam hal ini, film) untuk memberikan rekomendasi berdasarkan kesamaan dengan item yang sudah disukai.

Sistem rekomendasi ini diharapkan dapat meningkatkan kepuasan pengguna dengan memberikan film yang relevan berdasarkan genre atau film yang telah mereka tonton sebelumnya.

---

## Business Understanding

### Problem Statements

1. **Bagaimana cara memberikan rekomendasi film kepada pengguna yang berdasarkan pada preferensi mereka?**
2. **Bagaimana cara menangani masalah 'cold start', yaitu ketika seorang pengguna baru tidak memiliki data rating?**
3. **Bagaimana cara memberikan rekomendasi yang relevan tanpa hanya mengandalkan data rating pengguna?**

### Goals

Tujuan dari proyek ini adalah:
1. Membuat sistem rekomendasi yang memberikan rekomendasi film yang sesuai dengan preferensi pengguna.
2. Mengatasi masalah 'cold start' dengan pendekatan content-based filtering.
3. Membandingkan dua metode rekomendasi (Collaborative Filtering dan Content-Based Filtering) untuk memilih metode yang lebih baik.

### Solution Approach

Untuk mencapai tujuan-tujuan tersebut, saya menggunakan dua pendekatan sistem rekomendasi:

1. **Collaborative Filtering**:
   Menggunakan algoritma KNN (K-Nearest Neighbors) untuk menemukan kesamaan antara pengguna dan memberikan rekomendasi berdasarkan perilaku pengguna lain yang serupa.

2. **Content-Based Filtering**:
   Menggunakan genre film dan fitur terkait lainnya untuk memberikan rekomendasi berdasarkan kemiripan dengan film yang sudah disukai atau ditonton.

---

## Data Understanding

Dataset yang digunakan berasal dari **MovieLens**, yang terdiri dari tiga file utama:

1. **movies.csv**: Berisi informasi tentang film seperti `movieId`, `title`, dan `genres`.
2. **ratings.csv**: Berisi data rating pengguna terhadap film, dengan informasi seperti `userId`, `movieId`, dan `rating`.
3. **tags.csv**: Berisi data tag yang diberikan pengguna untuk film, dengan informasi seperti `userId`, `movieId`, dan `tag`.

### Variabel pada Dataset:
1. **movies.csv**:
   - `movieId`: ID unik untuk setiap film.
   - `title`: Judul film.
   - `genres`: Genre film, seperti Action, Comedy, Drama, dll.
   
2. **ratings.csv**:
   - `userId`: ID unik untuk setiap pengguna.
   - `movieId`: ID film yang dinilai.
   - `rating`: Rating yang diberikan pengguna (nilai antara 1 hingga 5).
   
3. **tags.csv**:
   - `userId`: ID unik untuk setiap pengguna.
   - `movieId`: ID film yang diberikan tag.
   - `tag`: Tag yang diberikan pengguna, misalnya "funny", "thriller", dll.

---

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

Kode untuk membangun model KNN:

```python
from surprise import Reader, Dataset, KNNBasic
from surprise.model_selection import train_test_split

# Membaca data ke dalam format yang diterima oleh Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Membagi data menjadi training dan test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Menggunakan algoritma KNN
model = KNNBasic()
model.fit(trainset)
predictions = model.test(testset)

# Evaluasi menggunakan RMSE
from surprise import accuracy
rmse = accuracy.rmse(predictions)
print(f"RMSE Collaborative Filtering: {rmse}")
```

### 2. **Content-Based Filtering**
Menggunakan genre film sebagai dasar untuk memberikan rekomendasi. Model ini memberikan film yang serupa dengan film yang telah ditonton oleh pengguna.

Kode untuk membangun model Content-Based Filtering:

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Membuat matriks genre menggunakan one-hot encoding
genre_matrix = pd.get_dummies(merged_data['genres'])

# Menghitung kemiripan kosinus antar film
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# Fungsi untuk memberikan rekomendasi film berdasarkan kemiripan
def content_based_recommendation(title, n=5):
    idx = merged_data[merged_data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]
    return merged_data['title'].iloc[movie_indices]

# Menguji rekomendasi untuk film 'Toy Story (1995)'
recommendations = content_based_recommendation('Toy Story (1995)', n=5)
print(recommendations)
```

---

## Evaluation

### Collaborative Filtering
Untuk evaluasi Collaborative Filtering, saya menggunakan metrik **Root Mean Squared Error (RMSE)**. Nilai RMSE yang lebih rendah menunjukkan model yang lebih akurat dalam memprediksi rating pengguna.

Hasil evaluasi RMSE untuk model Collaborative Filtering adalah **RMSE = 0.89**, yang menunjukkan model ini memiliki akurasi yang cukup baik.

### Content-Based Filtering
Evaluasi Content-Based Filtering didasarkan pada analisis **kemiripan genre**. Rekomendasi yang diberikan oleh model lebih relevan dengan preferensi genre pengguna dan tidak tergantung pada data rating.

---

## Kesimpulan

Setelah melakukan evaluasi, saya dapat menyimpulkan bahwa:
1. **Collaborative Filtering** efektif dalam memberikan rekomendasi yang dipersonalisasi berdasarkan data rating pengguna, namun mengalami masalah **cold start** untuk pengguna baru.
2. **Content-Based Filtering** memberikan rekomendasi yang relevan berdasarkan genre film yang serupa, cocok digunakan dalam situasi dengan data pengguna terbatas.

Metode yang paling tepat bergantung pada konteks aplikasi dan ketersediaan data pengguna.
