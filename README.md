# Laporan Proyek Machine Learning - Nama Anda

## Project Overview
Pada era digital dimana banyak sekali buku diterbitkan setiap hari, pembaca menghadapi banyak sekali pilihan buku yang membuat menemukan buku baru yang sesuai dengan keinginan mereka susah untuk dilakukan. Masalah ini dapat membuat pembaca enggan mencoba buku baru karena takut bahwa buku tersebut tidak akan cocok dengan selera mereka, menyusahkan mereka membaca literatur baru yang berkualitas, dan mengurangi penjualan buku pada toko. Proyek ini bertujuan untuk mengembangkan sistem rekomendasi buku dengan content-based filtering untuk mengatasi masalah-masalah tersebut. Dengan menganalisa fitur buku seperti penulis, genre, dan konten deskriptif dan menggunakan teknik seperti TF-IDF untuk ekstraksi fitur, sistem ini bertujuan untuk memberikan rekomendasi buku yang relevan pengguna, membantu mereka menemukan bacaan favorit mereka yang berikutnya. Pendekatan ini memanfaatkan fitur bawaan buku itu sendiri dan dapat dengan mudah digunakan dalam sistem rekomendasi hybrid yang juga menggabungkan teknik collaborative filtering.

Referensi:
Rosidah, L., & Dellia, P. (2024). Library Book Recommendation System Using Content-Based Filtering. Iota, 4(1), 46-65. https://pubs.ascee.org/index.php/iota/article/download/693/311/2229


## Business Understanding


### Problem Statements
- Bagaimana cara menemukan buku yang relevan dengan keinginan pembaca?

### Goals
- Membuat sistem rekomendasi yang dapat memberikan rekomendasi buku yang relevan dengan apa yang dicari oleh pengguna.

### Solution statements
- Menggunakan sistem rekomendasi content-based filtering untuk merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna.
- Menggunakan sistem rekomendasi hybrid yang menggunakan sistem sebelumnya dengan teknik collaborative filtering untuk merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah [Books Dataset](https://www.kaggle.com/datasets/abdallahwagih/books-dataset). Dataset ini merupakan kumpulan informasi mengenai buku dengan 6810 baris dan 12 kolom, dengan setiap baris mewakili satu buku dan berbagai fitur buku tersebut.

### Variabel-variabel pada dataset ini adalah sebagai berikut:
- isbn13: International Standard Book Number (ISBN) 13 digit, yaitu kode buku yang unik.

- isbn10: ISBN 10 digit untuk buku-buku tersebut.

- title: Judul buku.

- subtitle: Subjudul buku.

- authors: Penulis buku.

- categories: Genre atau kategori buku.

- thumbnail: URL atau tautan ke gambar sampul buku.

- description: Ringkasan singkat atau deskripsi konten setiap buku.

- published_year: Tahun penerbitan buku.

- average_rating: Rating rata-rata buku dari ulasan pembaca, bernilai dari 0 sampai 5.

- num_pages: Jumlah halaman buku.

- ratings_count: Jumlah total ulasan buku.

Menggunakan df.info(), dapat dilihat informasi dataset sebagai berikut:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6810 entries, 0 to 6809
Data columns (total 12 columns):
| Column           | Non-Null Count | Dtype   |
| ---------------- | -------------- | ------- |
| isbn13           | 6810 non-null  | int64   |
| isbn10           | 6810 non-null  | object  |
| title            | 6810 non-null  | object  |
| subtitle         | 2381 non-null  | object  |
| authors          | 6738 non-null  | object  |
| categories       | 6711 non-null  | object  |
| thumbnail        | 6481 non-null  | object  |
| description      | 6548 non-null  | object  |
| published_year   | 6804 non-null  | float64 |
| average_rating   | 6767 non-null  | float64 |
| num_pages        | 6767 non-null  | float64 |
| ratings_count    | 6767 non-null  | float64 |
dtypes: float64(4), int64(1), object(7)
memory usage: 638.6+ KB

Dapat dilihat bahwa terdapat 6810 baris dari 12 kolom data, dengan kebanyakan kolom memiliki data null atau hilang, dan jenis data berupa int64 (1 fitur), object (7 fitur), dan float64 (4 fitur). Berikutnya kita dapat menggunakan describe() untuk melihat statistik dataset.

|       | isbn13              | published_year | average_rating | num_pages | ratings_count       |
| :---- | :------------------ | :------------- | :------------- | :-------- | :------------------ |
| count | 6.810000e+03        | 6804.000000    | 6767.000000    | 6767.000000    | 6.767000e+03        |
| mean  | 9.780677e+12        | 1998.630364    | 3.933284       | 348.181026    | 2.106910e+04        |
| std   | 6.068911e+08        | 10.484257      | 0.331352       | 242.376783    | 1.376207e+05        |
| min   | 9.780002e+12        | 1853.000000    | 0.000000       | 0.000000    | 0.000000e+00        |
| 25%   | 9.780330e+12        | 1996.000000    | 3.770000       | 208.000000    | 1.590000e+02        |
| 50%   | 9.780553e+12        | 2002.000000    | 3.960000       | 304.000000    | 1.018000e+03        |
| 75%   | 9.780810e+12        | 2005.000000    | 4.130000       | 420.000000    | 5.992500e+03        |
| max   | 9.789042e+12        | 2019.000000    | 5.000000       | 3342.000000    | 5.629932e+06        |

Fungsi describe() memberikan informasi statistik pada kolom numerik, antara lain:

- Count  adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum.

**Univariate Analysis**
Menggunakan teknik analisis satu variabel untuk menganalisa data. Pada proyek ini hanya beberapa variabel yang relevan untuk sistem rekomendasi yang akan dianalisa, yaitu authors, categories, average_rating, dan ratings_count.

![img](https://i.imgur.com/vowjZmN.png)

Dapat dilihat bahwa distribusi 50 penulis buku dengan jumlah buku paling banyak didominasi ketiga penulis teratas (Agatha Christie, Stephen King, dan William Shakespeare), dengan mayoritas penulis tidak membuat lebih dari 15 buku.

![img](https://i.imgur.com/UhQAOP2.png)

Pada distribusi genre, dapat dilihat bahwa mayoritas buku merupakan buku fiksi, dengan Fiction dan Juvenile Fiction menempati hampir setengah dari semua buku. Tentu saja, ini kemungkinan besar karena keduanya merupakan genre dasar yang digabungkan dengan genre lain yang lebih spesifik, seperti Drama, Adventure, atau Mystery.

![img](https://i.imgur.com/TJ7TZdO.png)

Pada distribusi rating, dapat dilihat bahwa distribusi tersebut berbentuk distribusi normal, terpusat pada mean 3.93 seperti terlihat pada df.describe() pada bagian sebelumnya. Terdapat sedikit buku yang memiliki rating kurang dari 3, dengan kenaikan sedikit pada buku dengan rating 0, yang mungkin menandakan rating dari orang yang sangat tidak senang terhadap buku tersebut.

![img](https://i.imgur.com/lRJkDIu.png)

Terlihat pada grafik distribusi bahwa mayoritas buku memiliki jumlah rating yang rendah, dengan distribusi ekstrim yang menandakan sedikit sekali buku yang memiliki lebih dari 1e6 (1 juta) rating.


## Data Preparation
Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Pertama, kita akan drop fitur selain variabel yang relevan untuk sistem rekomendasi, yaitu title, subtitle, authors, categories, description, average_rating, dan ratings_count menggunakan kode berikut:

columns = ['title', 'subtitle', 'authors', 'categories', 'description', 'average_rating', 'ratings_count']
df = df[columns]

**Menangani Missing Value**
Kita lalu akan mengatasi missing value yang tadi telah terlihat dari df.info(). Kita memeriksa ulang missing value pada tiap kolom dengan kode berikut:

df.isnull().sum()

yang menghasilkan tabel berikut:
| Feature        | 0    |
| :------------- | :--- |
| title          | 0    |
| subtitle       | 4429 |
| authors        | 72   |
| categories     | 99   |
| description    | 262  |
| average_rating | 43   |
| ratings_count  | 43   |

Seperti dilihat dari df.info(), terdapat missing value, terutama banyak pada subtitle. Karena subtitle hanyalah bagian kedua dari judul, kita akan menyatukan kedua kolom tersebut. Untuk kolom teks lain, kita akan melakukan imputasi data, menggunakan kode berikut:

df['title'] = df['title'].fillna('') + ' ' + df['subtitle'].fillna('')
df = df.drop('subtitle', axis=1)
df['authors'] = df['authors'].fillna('Author Unknown')
df['description'] = df['description'].fillna('No description available')
df['categories'] = df['categories'].fillna('Unknown')
print(df.isnull().sum())

yang menghasilkan output berikut:
| Feature        | 0    |
| :------------- | :--- |
| title          | 0    |
| authors        | 0    |
| categories     | 0    |
| description    | 0    |
| average_rating | 43   |
| ratings_count  | 43   |

Missing value numerik hanya sedikit, dan mungkin membuat informasi salah ketika diimputasi, jadi mereka akan di drop dengan kode berikut:

df = df.dropna()
print(df.isnull().sum())

yang menghasilkan output berikut:
| Feature        | 0    |
| :------------- | :--- |
| title          | 0    |
| authors        | 0    |
| categories     | 0    |
| description    | 0    |
| average_rating | 0    |
| ratings_count  | 0    |

Maka kita telah mengatasi semua missing value pada data yang akan digunakan.

**Menangani Duplikat**
Kita akan memeriksa keberadaan baris duplikat yang menandakan buku sama terisi lebih dari sekali, menggunakan kode berikut:

duplicate_rows = df[df.duplicated()]
print("Duplicate Rows:")
print(duplicate_rows)

dengan output:

Duplicate Rows:
Empty DataFrame
Columns: [title, authors, categories, description, average_rating, ratings_count]
Index: []

Maka dapat dilihat tidak ada baris duplikat.

**Normalisasi Data**
Data rating yang berbentuk numerik akan dinormalisasi agar memiliki skala sama dengan cosine similarity (yaitu 0-1) untuk sistem rekomendasi hybrid nantinya, menggunakan kode berikut:

scaler = MinMaxScaler()
df['avg_ratings_scaled'] = scaler.fit_transform(df[['average_rating']])

**Pre-processing Teks**
Selanjutnya kita akan memproses data teks agar mudah digunakan oleh model untuk content-based filtering nantinya. Kita akan membersihkan teks dengan menghilangkan tanda baca, mengubah teks ke huruf kecil, dan menghilangkan stopwords. Kita lalu akan melakukan tokenisasi teks (memecah teks menjadi kata individu atau token) dan stemming (mengubah kata menjadi bentuk dasar), menggunakan kode berikut:

def preprocess_text(text):
    text = re.split(r';\s*|;', text)
    text = ' '.join(text)
    text = re.sub(r'[^\w\s]', '', text).lower()
    text = [word for word in text.split() if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text]
    return ' '.join(text)

df['processed_author'] = df['authors'].apply(preprocess_text)
df['processed_categories'] = df['categories'].apply(preprocess_text)
df['processed_description'] = df['description'].apply(preprocess_text)

Kita juga akan menghilangkan whitespace dari depan atau belakang judul, agar lebih mudah digunakan nantinya, menggunakan kode berikut:

df['title'] = df['title'].str.strip()

Berikut adalah sampel 5 baris teratas dari df.head:
![img](https://i.imgur.com/IU9sR0Q.png)
Maka dapat dilihat kolom teks yang dibersihkan telah diubah agar mudah digunakan model nantinya.

**Perbaiki Indeks**
Karena tadi dilakukan drop, beberapa kolom data hilang, namun indeks tiap kolom masih tetap sama. Untuk menghindari masalah, indeks akan kita reset agar konsisten dengan jumlah kolom baru dengan kode berikut:

df = df.reset_index(drop=True)

Untuk memastikan, kita gunakan df.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6767 entries, 0 to 6766
Data columns (total 10 columns):
| Column               | Non-Null Count | Dtype   |
| -------------------- | -------------- | ------- |
| title                | 6767 non-null  | object  |
| authors              | 6767 non-null  | object  |
| categories           | 6767 non-null  | object  |
| description          | 6767 non-null  | object  |
| average_rating       | 6767 non-null  | float64 |
| ratings_count        | 6767 non-null  | float64 |
| avg_ratings_scaled   | 6767 non-null  | float64 |
| processed_author     | 6767 non-null  | object  |
| processed_categories | 6767 non-null  | object  |
| processed_description| 6767 non-null  | object  |
dtypes: float64(3), object(7)

Maka dapat dilihat bahwa indeks sudah disesuaikan dengan jumlah baris yang dikurangi.

## Modeling
Modeling adalah tahapan di mana kita menggunakan sistem rekomendasi untuk menjawab problem statement dari tahap business understanding.

Pertama, kita akan menggunakan TF-IDF vectorizer untuk mengekstrak fitur dari data teks yang sudah diproses sebelumnya. TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang umum dan efektif untuk merepresentasikan data teks secara numerik dengan menangkap pentingnya kata-kata dalam suatu dokumen atau korpus. Ini dilakukan dengan kode berikut:

author_vectorizer = TfidfVectorizer()
categories_vectorizer = TfidfVectorizer()
description_vectorizer = TfidfVectorizer()
author_features = author_vectorizer.fit_transform(df['processed_author'])
categories_features = categories_vectorizer.fit_transform(df['processed_categories'])
description_features = description_vectorizer.fit_transform(df['processed_description'])

Selanjutnya, kita akan menggabungkan fitur-fitur yang sudah kita ekstrak dari data teks menjadi satu vektor dan menghitung cosine similarity dari buku-buku dengan kode berikut:

features = hstack([author_features, categories_features, description_features])
features = features.toarray()
similarity_matrix = cosine_similarity(features)

# Content-based Recommender
Kita lalu dapat mendefinisikan fungsi untuk melakukan rekomendasi buku top-n sesuai dengan nilai cosine similarity antar buku yang mempertimbangkan penulis, genre buku, dan deskripsi singkat tentang buku tersebut sebagai content-based recommendation system. Kita juga akan menghilangkan white space dari awal dan akhir input judul buku agar sistem rekomendasi tetap bekerja bahkan jika terdapat salah spasi di depan atau belakang judul.

    def get_recommendations(book_title, top_n=10):
    """
    Gets book recommendations using cosine similarity.

    Args:
        book_title (str): The title of the book to get recommendations for.
        top_n (int, optional): The maximum number of recommendations to return.

    Returns:
        list: A list of dictionaries containing the title, author, categories, and description of the recommended books.
    """
    # Strip leading/trailing whitespaces from input book_title
    book_title = book_title.strip()

    # Check if the book title exists in the DataFrame
    if book_title in df['title'].values:
        book_index = df[df['title'] == book_title].index[0]

        # Get the similarity scores for the book
        similarity_scores = similarity_matrix[book_index]

        # Sort the scores in descending order and get the top_n indices
        top_indices = similarity_scores.argsort()[::-1][1:top_n+1]

        # Create list of dictionaries to return later
        recommendations = []
        for index in top_indices:
            book_info = {
                'title': df.loc[index, 'title'],
                'author': df.loc[index, 'authors'],
                'categories': df.loc[index, 'categories'],
                'description': df.loc[index, 'description']
            }
            recommendations.append(book_info)
        return recommendations
    else:
        # Return an empty list if the book title is not found
        print(f"Book '{book_title}' not found in the dataset.")
        return []

Berikutnya, kita akan mencoba fungsi tersebut dengan mencari 10 rekomendasi buku yang paling mirip dengan buku yang disenangi. Kali ini digunakan buku misteri Agatha Christie, "A Murder is Announced". Berikut hasil 10 rekomendasi buku dari sistem rekomendasi content-based filtering:

Title: Miss Marple The Complete Short Stories
Author: Agatha Christie
Categories: Detective and mystery stories, English
Description: Miss Marple featured in 20 short stories, published in a number of different collections in Britain and America. Presented here in their order of publication, Miss Marple uses her unique insight to deduce the truth about a series of unsolved crimes.
---
Title: The Thirteen Problems
Author: Agatha Christie
Categories: Detective and mystery stories, English
Description: The Tuesday Night Club is a venue where locals challenge Miss Marple to solve recent crimes... One Tuesday evening a group gathers at Miss Marple's house and the conversation turns to unsolved crimes... The case of the disappearing bloodstains; the thief who committed his crime twice over; the message on the death-bed of a poisoned man which read 'heap of fish'; the strange case of the invisible will; a spiritualist who warned that 'Blue Geranium' meant death... Now pit your wits against the powers of deduction of the 'Tuesday Night Club'.
---
Title: Appointment with Death
Author: Agatha Christie
Categories: Detective and mystery stories
Description: A repugnant Amercian widow is killed during a trip to Petra... Among the towering red cliffs of Petra, like some monstrous swollen Buddha, sat the corpse of Mrs Boynton. A tiny puncture mark on her wrist was the only sign of the fatal injection that had killed her. With only 24 hours available to solve the mystery, Hercule Poirot recalled a chance remark he'd overheard back in Jerusalem: 'You see, don't you, that she's got to be killed?' Mrs Boynton was, indeed, the most detestable woman he'd ever met...
---
Title: The Big Four
Author: Agatha Christie
Categories: Detective and mystery stories
Description: A ruthless international cartel seeks world domination... Framed in the doorway of Poirot's bedroom stood an uninvited guest, coated from head to foot in dust. The man's gaunt face stared for a moment, then he swayed and fell. Who was he? Was he suffering from shock or just exhaustion? Above all, what was the significance of the figure 4, scribbled over and over again on a sheet of paper? Poirot finds himself plunged into a world of international intrigue, risking his life to uncover the truth about 'Number Four'.
---
Title: The Listerdale Mystery
Author: Agatha Christie
Categories: Detective and mystery stories
Description: A selection of mysteries, some light-hearted, some romantic, some very deadly... Twelve tantalizing cases... the curious disappearance of Lord Listerdale; a newlywed's fear of her ex-fiance; a strange encounter on a train; a domestic murder investigation; a wild man's sudden personality change; a retired inspector's hunt for a murderess; a young woman's impersonation of a duchess; a necklace hidden in a basket of cherries; a mystery writer's arrest for murder; an astonishing marriage proposal; a soprano's hatred for a baritone; the case of the rajah's emerald. All have one thing in common: the skilful hand of Agatha Christie.
---
Title: Death in the Clouds
Author: Agatha Christie
Categories: Detective and mystery stories
Description: A woman is killed by a poisoned dart in the enclosed confines of a commercial passenger plane... From seat No.9, Hercule Poirot was ideally placed to observe his fellow air passengers. Over to his right sat a pretty young woman, clearly infatuated with the man opposite; ahead, in seat No.13, sat a Countess with a poorly-concealed cocaine habit; across the gangway in seat No.8, a detective writer was being troubled by an aggressive wasp. What Poirot did not yet realize was that behind him, in seat No.2, sat the slumped, lifeless body of a woman.
---
Title: Hercule Poirot's Casebook
Author: Agatha Christie
Categories: Detective and mystery stories
Description: Here, for the first time in one volume, is the complete collection of fifty stories about Hercule Poirot.
---
Title: The Mysterious Mr. Quin
Author: Agatha Christie
Categories: Detective and mystery stories
Description: A mysterious stranger appears at a New Year's Eve party, becoming the enigmatic sleuthing sidekick to the snobbish Mr Satterthwaite... So far, it had been a typical New Year's Eve house party. But Mr Satterthwaite - a keen observer of human nature - sensed that the real drama of the evening was yet to unfold. So it proved when a mysterious stranger arrived after midnight. Who was this Mr Quin? And why did his presence have such a pronounced effect on Eleanor Portal, the woman with the dyed-black hair?
---
Title: The Hollow
Author: Agatha Christie
Categories: Detective and mystery stories
Description: Lucy Angkatell Invited Hercule Poirot To Lunch. To Tease The Great Detective, Her Guests Stage A Mock Murder Beside The Swimming Pool. Unfortunately, The Victim Plays The Scene For Real. As His Blood Drips Into The Water, John Christow Gasps One Final Word: Henrietta . In The Confusion, A Gun Sinks To The Bottom Of The Pool. Poirot S Enquiries Reveal A Complex Web Of Romantic Attachments. It Seems Everyone In The Drama Is A Suspect And Each A Victim Of Love.
---
Title: Murder in Mesopotamia
Author: Agatha Christie
Categories: Detective and mystery stories
Description: An archaeologist's wife is murdered on the shores of the River Tigris in Iraq... It was clear to Amy Leatheran that something sinister was going on at the Hassanieh dig in Iraq; something associated with the presence of 'Lovely Louise', wife of celebrated archaeologist Dr Leidner. In a few days' time Hercule Poirot was due to drop in at the excavation site. But with Louise suffering from terrifying hallucinations, and tension within the group becoming almost unbearable, Poirot might just be too late...
---
Maka dapat dilihat bahwa novel misteri Agatha Christie menghasilkan rekomendasi buku-buku misteri, dengan buku rekomendasi juga oleh Agatha Christie, menandakan bahwa model dapat mempertimbangkan penulis dan genre seperti diharapkan.

Sistem rekomendasi content-based filtering seperti ini memiliki kelebihan sebagai berikut:
- Lebih mudah didesain dan diimplementasikan.
- Tidak memerlukan input pengguna lain untuk merekomendasikan konten.
- Mudah merekomendasikan konten baru selama data deskripsinya tersedia.

Namun, terdapat juga kekurangan:
- Masalah 'cold start' yang berarti sistem tidak dapat merekomendasikan konten secara akurat jika pengguna belum memiliki konten yang dia senangi.
- Cenderung merekomendasikan item yang sangat mirip dengan preferensi sebelumnya.
- Bergantung pada kualitas metadata konten (e.g. deskripsi singkat buku/acara).

# Hybrid Recommender
Untuk mengatasi sebagian kekurangan tersebut, kali ini kita akan menggunakan hybrid recommendation system, yang menggunakan fungsi mirip dengan content-based namun menambahkan weighted linear function untuk memperhitungkan rating buku dari pembaca. Pertama kita akan mendefinisikan bobotnya dengan kode berikut:

weight_similarity = 0.7
weight_rating = 0.3

Fungsi baru untuk rekomendasi hybrid sangat mirip dengan fungsi sebelumnya, dengan tambahan modifikasi untuk menggunakan rating dan bobot yang sudah didefinisikan, dan memfilter buku rekomendasi agar mendapatkan buku dengan yang sudah dirating minimal 1000 kali. Modifikasi kode sebagai berikut:

    def get_recommendations_weighted(book_title, top_n=10):
    """
    Gets book recommendations using a weighted average of cosine similarity and scaled ratings.

    Args:
        book_title (str): The title of the book to get recommendations for.
        top_n (int, optional): The maximum number of recommendations to return.

    Returns:
        list: A list of dictionaries containing the title, author, categories, and description of the recommended books.
    """
    # Strip leading/trailing whitespaces from input book_title
    book_title = book_title.strip()

    # Check if the book title exists in the DataFrame
    if book_title in df['title'].values:
        book_index = df[df['title'] == book_title].index[0]

        # Get the similarity scores for the book
        similarity_scores = similarity_matrix[book_index]

        # Find weighted scores using cosine similarity and ratings
        ratings = df['avg_ratings_scaled'].values
        weighted_scores = (weight_similarity * similarity_scores +
                           weight_rating * ratings)

        # Filter indices based on ratings_count, filtering books with less than 1000 ratings
        filtered_indices = [index for index in weighted_scores.argsort()[::-1][1:]
                             if df.iloc[index, df.columns.get_loc('ratings_count')] >= 1000]

        # Get top indices after filtering
        top_indices = filtered_indices[:top_n]

        # Create list of dictionaries to return
        recommendations = []
        for index in top_indices:
            # Use .iloc to access data based on integer position
            book_info = {
                'title': df.iloc[index, df.columns.get_loc('title')],
                'author': df.iloc[index, df.columns.get_loc('authors')],
                'genre': df.iloc[index, df.columns.get_loc('categories')],
                'description': df.iloc[index, df.columns.get_loc('description')],
            }
            recommendations.append(book_info)

        return recommendations
    else:
        print(f"Book '{book_title}' not found in the dataset.")
        return []

Akhirnya, kita akan mencoba melihat 10 hasil rekomendasi pada buku yang sama seperti tadi, "A Murder is Announced".

Title: Miss Marple The Complete Short Stories
Author: Agatha Christie
Genre: Detective and mystery stories, English
Description: Miss Marple featured in 20 short stories, published in a number of different collections in Britain and America. Presented here in their order of publication, Miss Marple uses her unique insight to deduce the truth about a series of unsolved crimes.
---
Title: The Thirteen Problems
Author: Agatha Christie
Genre: Detective and mystery stories, English
Description: The Tuesday Night Club is a venue where locals challenge Miss Marple to solve recent crimes... One Tuesday evening a group gathers at Miss Marple's house and the conversation turns to unsolved crimes... The case of the disappearing bloodstains; the thief who committed his crime twice over; the message on the death-bed of a poisoned man which read 'heap of fish'; the strange case of the invisible will; a spiritualist who warned that 'Blue Geranium' meant death... Now pit your wits against the powers of deduction of the 'Tuesday Night Club'.
---
Title: Hercule Poirot's Casebook
Author: Agatha Christie
Genre: Detective and mystery stories
Description: Here, for the first time in one volume, is the complete collection of fifty stories about Hercule Poirot.
---
Title: Appointment with Death
Author: Agatha Christie
Genre: Detective and mystery stories
Description: A repugnant Amercian widow is killed during a trip to Petra... Among the towering red cliffs of Petra, like some monstrous swollen Buddha, sat the corpse of Mrs Boynton. A tiny puncture mark on her wrist was the only sign of the fatal injection that had killed her. With only 24 hours available to solve the mystery, Hercule Poirot recalled a chance remark he'd overheard back in Jerusalem: 'You see, don't you, that she's got to be killed?' Mrs Boynton was, indeed, the most detestable woman he'd ever met...
---
Title: Murder in Mesopotamia
Author: Agatha Christie
Genre: Detective and mystery stories
Description: An archaeologist's wife is murdered on the shores of the River Tigris in Iraq... It was clear to Amy Leatheran that something sinister was going on at the Hassanieh dig in Iraq; something associated with the presence of 'Lovely Louise', wife of celebrated archaeologist Dr Leidner. In a few days' time Hercule Poirot was due to drop in at the excavation site. But with Louise suffering from terrifying hallucinations, and tension within the group becoming almost unbearable, Poirot might just be too late...
---
Title: Death in the Clouds
Author: Agatha Christie
Genre: Detective and mystery stories
Description: A woman is killed by a poisoned dart in the enclosed confines of a commercial passenger plane... From seat No.9, Hercule Poirot was ideally placed to observe his fellow air passengers. Over to his right sat a pretty young woman, clearly infatuated with the man opposite; ahead, in seat No.13, sat a Countess with a poorly-concealed cocaine habit; across the gangway in seat No.8, a detective writer was being troubled by an aggressive wasp. What Poirot did not yet realize was that behind him, in seat No.2, sat the slumped, lifeless body of a woman.
---
Title: The Hollow
Author: Agatha Christie
Genre: Detective and mystery stories
Description: Lucy Angkatell Invited Hercule Poirot To Lunch. To Tease The Great Detective, Her Guests Stage A Mock Murder Beside The Swimming Pool. Unfortunately, The Victim Plays The Scene For Real. As His Blood Drips Into The Water, John Christow Gasps One Final Word: Henrietta . In The Confusion, A Gun Sinks To The Bottom Of The Pool. Poirot S Enquiries Reveal A Complex Web Of Romantic Attachments. It Seems Everyone In The Drama Is A Suspect And Each A Victim Of Love.
---
Title: The Mysterious Mr. Quin
Author: Agatha Christie
Genre: Detective and mystery stories
Description: A mysterious stranger appears at a New Year's Eve party, becoming the enigmatic sleuthing sidekick to the snobbish Mr Satterthwaite... So far, it had been a typical New Year's Eve house party. But Mr Satterthwaite - a keen observer of human nature - sensed that the real drama of the evening was yet to unfold. So it proved when a mysterious stranger arrived after midnight. Who was this Mr Quin? And why did his presence have such a pronounced effect on Eleanor Portal, the woman with the dyed-black hair?
---
Title: The Big Four
Author: Agatha Christie
Genre: Detective and mystery stories
Description: A ruthless international cartel seeks world domination... Framed in the doorway of Poirot's bedroom stood an uninvited guest, coated from head to foot in dust. The man's gaunt face stared for a moment, then he swayed and fell. Who was he? Was he suffering from shock or just exhaustion? Above all, what was the significance of the figure 4, scribbled over and over again on a sheet of paper? Poirot finds himself plunged into a world of international intrigue, risking his life to uncover the truth about 'Number Four'.
---
Title: Spider's Web A Novel
Author: Charles Osborne;Agatha Christie
Genre: Detective and mystery stories
Description: A new 'Christie for Christmas' -- a full-length novel adapted from her acclaimed play by Charles Osborne Following BLACK COFFEE and THE UNEXPECTED GUEST comes the final Agatha Christie play novelisation, bringing her superb storytelling to a new legion of fans. Clarissa, the wife of a Foreign Office diplomat, is given to daydreaming. 'Supposing I were to come down one morning and find a dead body in the library, what should I do?' she muses. Clarissa has her chance to find out when she discovers a body in the drawing-room of her house in Kent. Desperate to dispose of the body before her husband comes home with an important foreign politician, Clarissa persuades her three house guests to become accessories and accomplices. It seems that the murdered man was not unknown to certain members of the house party (but which ones?), and the search begins for the murderer and the motive, while at the same time trying to persuade a police inspector that there has been no murder at all... SPIDER'S WEB was written in 1954 specifically for Margaret Lockwood and opened first at the Theatre Royal Nottingham before moving to the Savoy Theatre in London on 14 December 1954. With THE MOUSETRAP and WI
---

Maka bisa dilihat hybrid recommender menghasilkan rekomendasi yang mirip dengan content-based recommender dalam mempertimbangkan penulis dan genre buku, namun menghasilkan rekomendasi yang agak berbeda dengan model yang hanya menggunakan content based filtering.

Sistem rekomendasi hybrid seperti ini memiliki kelebihan sebagai berikut:
- Mengurangi masalah 'cold start' dengan teknik collaborative filtering.
- Menggabungkan pendekatan yang berbeda dapat menghasilkan rekomendasi yang lebih beragam dan akurat.
- Mengurangi ketergantungan sistem content-based pada metadata konten.

Namun, terdapat juga kekurangan:
- Lebih rumit dalam desain dan implementasinya.
- Perlu dilakukan eksperimen untuk optimisasi bobot.
- Lebih mahal secara komputasi.

## Evaluation
Pada proyek ini, kita akan menggunakan metrik precision dan recall, menggunakan confusion matrix sebagai ilustrasi. Berikut penjelasan singkat mengenai metrik yang digunakan:

1. Confusion Matrix
Confusion Matrix adalah tabel sederhana yang menunjukkan seberapa baik kinerja sistem dengan membandingkan prediksinya dengan hasil aktual. Matriks ini membagi prediksi menjadi empat kategori: prediksi yang benar untuk kedua kelas (True Positive dan True Negative) dan prediksi yang salah (False Positive dan False Negative).
- True Positive (TP): Model secara akurat memprediksi kelas positif.
- True Negative (TN): Model secara akurat memprediksi kelas negatif.
- False Positive (FP): Model secara tidak akurat memprediksi kelas positif (Type I error).
- False Negative (FN): Model secara tidak akurat memprediksi kelas negatif (Type II error).

2. Precision
Precision mengukur proporsi prediksi positif yang sebenarnya positif. 
$$\text{Precision} = \frac{TP}{TP + FP}$$
Di mana:
• $$(TP)$$ mewakili jumlah positif benar.
• $$(FP)$$ mewakili jumlah positif salah.

3. Recall
Recall mengukur proporsi prediksi positif aktual yang diidentifikasi dengan benar oleh model.
$$\text{Recall} = \frac{TP}{TP + FN}$$
Dimana:
• $$(TP)$$ mewakili jumlah positif benar.
• $$(FN)$$ mewakili jumlah negatif palsu.

Pada tahap ini, kita mengevaluasi model dengan metrik tertentu untuk menilai performanya. Untuk mengevaluasi kedua sistem rekomendasi (content-based dan hybrid), kita akan menggunakan confusion matrix untuk melihat relevansi buku yang direkomendasikan. Kali ini buku relevan akan didefinisikan sebagai buku dengan cosine similarity 0.5 keatas dan rating 3 keatas.

Berikut adalah confusion matrix content based dan hybrid recommender bagi buku "A Murder is Announced" yang telah digunakan tadi:

![img](https://i.imgur.com/h1pfAmU.png)

![img](https://i.imgur.com/vh2jNVD.png)

Dapat dilihat dari kedua confusion matrix diatas bahwa kedua sistem rekomendasi telah berhasil merekomendasikan 10 buku yang memiliki cosine similarity minimal 0.5 dan rating minimal 3 secara sempurna, yang menunjukkan bahwa kedua sistem rekomendasi merekomendasikan 10 buku yang mirip dengan buku "A Murder is Announced" dengan rating rata-rata minimal 3, yang kemungkinan besar adalah rekomendasi baik bagi pembaca.

Selanjutnya, kedua sistem dievaluasi dengan buku lain, "The Dead Zone" oleh Stephen King, untuk memastikan performa model baik.

![img](https://i.imgur.com/5Mchd68.png)

![img](https://i.imgur.com/8uAMTGD.png)

Maka dapat dilihat dari kedua confusion matrix diatas bahwa kedua sistem rekomendasi memiliki performa yang baik, dengan content based merekomendasikan 2 false negative, dan hybrid system merekomendasikan 1 false negative, menandakan hybrid recommender dapat lebih menemukan buku relevan karena bobot pada fungsi rekomendasinya.

Ini berarti mayoritas dari 10 buku rekomendasi kedua sistem berupa rekomendasi baik, dengan 2 buku dari content based recommender dan 1 buku dari hybrid recommender berupa rekomendasi yang tidak relevan.

Dari confusion matrix kedua buku tersebut, kita dapat menghitung metrik precision dan recall, menggunakan rumus yang telah didefinisikan tadi dengan kode berikut:

    def calculate_precision_recall(TP, FP, FN):
        """Calculates precision and recall."""
        if TP + FP == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
    
        if TP + FN == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        return precision, recall

Menggunakan fungsi tersebut untuk menghitung precision dan recall, kita mendapatkan:

Content-Based Filtering (A Murder is Announced): Precision = 1.00, Recall = 1.00
Hybrid Filtering (A Murder is Announced): Precision = 1.00, Recall = 1.00
Content-Based Filtering (The Dead Zone): Precision = 1.00, Recall = 0.80
Hybrid Filtering (The Dead Zone): Precision = 1.00, Recall = 0.90

Precision yang tinggi berarti sistem merekomendasikan buku yang relevan dan menghindari buku yang tidak relevan.

Recall yang tinggi berarti sistem dapat menemukan sebagian besar buku yang relevan.

Dari evaluasi di atas, dapat disimpulkan bahwa kedua sistem rekomendasi kita baik dalam menemukan dan merekomendasikan buku yang relevan bagi pembaca, dengan hybrid recommender menunjukkan performa recall lebih baik pada buku "The Dead Zone".

Mari mengevaluasi hasil proyek sesuai dengan statement awal:
### Problem Statements
- Bagaimana cara menemukan buku yang relevan dengan keinginan pembaca?

### Goals
- Membuat sistem rekomendasi yang dapat memberikan rekomendasi buku yang relevan dengan apa yang dicari oleh pengguna.

### Solution statements
- Menggunakan sistem rekomendasi content-based filtering untuk merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna.
- Menggunakan sistem rekomendasi hybrid yang menggunakan sistem sebelumnya dengan teknik collaborative filtering untuk merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna.

**1. Apakah kita telah membuat dan menggunakan sistem rekomendasi content-based filtering dan sistem rekomendasi hybrid untuk merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna?**

Kita telah membuat 2 sistem rekomendasi yang dapat merekomendasikan buku yang relevan dengan apa yang dicari oleh pengguna secara akurat, dengan sistem rekomendasi content-based filtering memiliki performa recall lebih buruk dibandingkan dengan sistem rekomendasi hybrid pada buku "The Dead Zone":

Content-Based Filtering (A Murder is Announced): Precision = 1.00, Recall = 1.00
Hybrid Filtering (A Murder is Announced): Precision = 1.00, Recall = 1.00
Content-Based Filtering (The Dead Zone): Precision = 1.00, Recall = 0.80
Hybrid Filtering (The Dead Zone): Precision = 1.00, Recall = 0.90

Hasil precision sempurna menunjukkan bahwa kedua sistem rekomendasi berhasil menemukan dan merekomendasikan buku yang relevan, menghindari buku yang tidak relevan pada 10 rekomendasi buku teratas. Perbedaan performa kedua model terlihat pada rekomendasi buku jika pengguna mencari rekomendasi berdasarkan pada buku "The Dead Zone" oleh Stephen King, dengan Recall sistem rekomendasi content-based filtering bernilai 0.8, dan sistem rekomendasi hybrid bernilai 0.9. Nilai recall yang lebih baik pada hybrid filtering menunjukkan bahwa pengunaan bobot rating pada sistem hybrid membantunya memberikan rekomendasi yang lebih menangkap buku yang sebenarnya relevan bagi pengguna.
