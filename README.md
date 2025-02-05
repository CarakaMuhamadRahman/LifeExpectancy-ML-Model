# Laporan Proyek Machine Learning - Caraka Rahman

## Domain Proyek - Kesehatan

Berkembangnya teknologi informasi khususnya dalam bidang Artificial Intelligence(AI) telah memberikan dampak yang sangat signifikan terhadap kehidupan sehari-hari, baik itu dampak positif maupun negatif. Contoh dampak positif dari penerapan teknologi berbasis AI adalah dapat mempermudah manusia dalam melakukan pekerjaan di berbagai sektor, seperti pertanian, kedokteran, pendidikan, dll. Walaupun telah memberikan banyak dampak positif, tetapi tampaknya perkembangan AI ini belum mampu memberikan dampak optimal pada negara-negara terbelakang, seperti pada benua Afrika, khususnya untuk meningkatkan angka harapan hidup(Life Expectancy).

Life Expectancy telah menjadi masalah utama untuk negara-negara pada benua Africa untuk waktu yang cukup lama, sehingga untuk menangani masalah life expectancy ini harus diketahui terlebih dahulu apa faktor yang mempengaruhi life expectancy. Setelah mengetahui faktor-faktor tersebut, selanjutnya harus dibuat suatu sistem AI, dalam hal ini ML Model yang dapat memprediksi nilai life expectancy berdasarkan faktor-faktor yang mempengaruhinya, untuk dapat mencegah penurunan yang lebih extreme.

Sehingga fokus dari submission ini adalah membuat ML Model Linear Regression untuk memprediksi life expectancy menggunakan berbagai macam Algorithm seperti K-NN, Random Forest, dan AdaBoost.

## Business Understanding

### Problem Statements
Berdasarkan kondisi yang telah diuraikan sebelumnya, maka ML Model Linear Regression ini dibuat untuk menjawab beberapa permasalahan berikut:
- Dari serangkaian fitur yang ada, fitur manakah yang paling berpengaruh terhadap angka harapan hidup(Life Expectancy)?
- Manakah Algoritma yang paling optimal untuk memprediksi Life Expectancy?

### Goals
Berdasarkan problem statement yang telah diuraikan di atas, maka berikut goals atau tujuan dari dibangunnya model Linear Regression ini:
- Mengetahui fitur yang paling berkorelasi atau berpengaruh terhadap pengembangan model.
- Membangun ML Model yang dapat memprediksi angka Life Expectancy seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution statements
Untuk mencapai/meraih goals di atas, berikut merupakan tahapan-tahapan yang akan dilakukan:
- Melakukan Data Analysis sedalam mungkin agar dapat memahami data untuk mengetahui fitur mana yang paling berpengaruh terhadap fitur Life Expectancy.
- Melakukan eksperimen menggunakan tiga Algoritma Machine Learning berbeda untuk memperoleh model yang paling optimal, Algoritma yang akan digunakan diantaranya K-NN, Random Forest, dan AdaBoost.

## Data Understanding
Untuk Datasets yang digunakan pada eksperimen ini, akan menggunakan Datasets yang berasal dari World Health Organization(WHO) .Datasets ini terdiri dari 2938 jumlah data dengan jumlah atribut/fitur sebanyak 21. Berikut link untuk mengunduh Datasets [Life Expectancy WHO](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

### Penjelasan Fitur:
- Country : Fitur ini merepresentasikan negara-negara dari berbagai macam benua. Jumlah negara pada Datasets ini terdiri dari 193 negara.

- Year : Fitur ini merepresentasikan tahun pengamatan dari suatu data. Pada Datasets ini, tahun yang diamati adalah dari tahun 2000 hingga 2015.
- Status : Fitur ini merepresentasikan kondisi atau status dari suatu negara, yang mana fitur ini hanya memiliki dua values, yaitu Developing(Berkembang) dan Developed(Maju).
- Life Expectancy : Fitur ini merupakan Label/Target dari ML Model yang akan dikembangkan. Fitur ini merepresentasikan angka harapan hidup dari suatu negara.
- Infant Deaths : Fitur ini merepresentasikan angka kematian bayi per 1000 populasi.
- Alcohol : Fitur ini merepresentasikan jumlah konsumsi alkohol dalam liter untuk seseorang yang berumur di atas 15 tahun.
- Percentage Expenditure : Fitur ini merepresentasikan pengeluaran untuk kesehatan sebagai persentase dari produk domestik bruto per kapita(%).
- Hepatitis B : Fitur ini merepresentasikan persentase cakupan imunisasi Hepatitis B(HepB3) pada anak usia satu tahun.
- Measles : Fitur ini merepresentasikan jumlah kasus campak yang dilaporkan per 1000 populasi.
- BMI : Fitur ini merepresentasikan rata-rata Body Mass Index(Index Massa Tubuh) dari keseluruhan populasi.
- under-five deaths : Fitur ini merepresentasikan jumlah kematian balita per 1000 populasi.
- Polio : Fitur ini merepresentasikan cakupan imunisasi Polio(Pol3) pada anak usia satu tahun(%).
- Total Expenditure : Fitur ini merepresentasikan pengeluaran pemerintah secara umum untuk kesehatan.
- Diphtheria : Fitur ini merepresentasikan cakupan imunisasi difteri tetanus toksoid dan pertusis(DTP3) pada anak usia satu tahun(%).
- HIV/AIDS : Fitur ini merepresentasikan angka kematian akibat HIV/AIDS per 1000 kelahiran hidup (0-4 tahun).
- GDP : Fitur ini merepresentasikan produk domestik bruto per kapita dalam USD.
- Population : Fitur ini merepresentasikan jumlah populasi dalam suatu negara.
- Thinness 1-19 Years : Fitur ini merepresentasikan prevalensi kekurusan di antara anak-anak dan remaja untuk usia 10 hingga 19 tahun(%).
- Thinness 5-9 Years : Fitur ini merepresentasikan prevalensi kekurusan di antara anak-anak untuk usia 5 hingga 9 tahun(%).
- Income Composition of Resources(ICR) : Fitur ini merepresentasikan Human Development Index(HDI) dalam hal ICR(Index mulai dari 0 hingga 1).
- Schooling : Fitur ini merepresentasikan jumlah tahun bersekolah.

### Data Analysis

Berikut merupakan kondisi awal Datasets sebelum dilakukan **Data Preprocessing**:

| Missing Values | Duplicate Values |
| :------------: | :------------:   |
| 2.553          |0

**A. Univariate Analysis**:
- Categorical Features
  
  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/Top10Countries.png "Top 10 Countries")

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/Status.png "Status")

  _Penjelasan:_
  - Dari Data Visualization di atas, terlihat jelas bahwa **Afghanistan** merupakan salah satu negara dengan jumlah sample data terbanyak pada Datasets.

  - Sedangkan Status negara **Berkembang/Developing** memiliki sample data lebih banyak dibandingkan dengan negara **Maju/Developed**.

- Numerical Features

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/HistogramOfNumericalValues.png "Histogram")

  _Penjelasan:_
  - Dari Histogram di atas terlihat jelas bahwa pada beberapa feature memiliki kondisi **Right/Left Skewed**. Untuk menangani masalah ini, akan diimplementasi metode **IQR(Interquartile Range)** pada proses Outliers Removal.

**B. Multivariate Analysis**:
- Categorical Features

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/LifeExpectancyAverageTop5Countries.png "LifeExpAverageOverTop5Countries")

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/LifeExpectancyAverageOverStatus.png "LifeExpAverageOverStatus")

  _Penjelasan:_
  - Dari Barplot di atas, terlihat bahwa negara **New Zealand** pada tahun **2010** menjadi negara dengan angka harapan hidup tertinggi dibandingkan dengan negara lainnya, dengan rata-rata angka harapan hidup sekitar 90 tahun.
  
  - Sedangkan negara maju(**Developed Country**), memiliki rata-rata angka harapan hidup lebih tinggi dibandingkan dengan negara berkembang(**Developing Country**).

- Numerical Features

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/PairPlotNumericalFeatures.png  "PairPlotNumericalFeatures")

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/Heatmap.png "Heatmap")

  _Penjelasan:_
  - Berdasarkan Pairplot dan Heatmap di atas, dapat disimpulkan bahwa fitur Year, under-five deaths, thinness 1-19 years, dan Measles memiliki hubungan/korelasi yang lemah terhadap fitur Life Expectancy.

  Berikut merupakan kondisi data **sebelum** dilakukan proses Outliers Removal :

  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/OutliersRemoval-1.png "Before")
  ![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/OutliersRemoval-2.png "Before")
  
  _Penjelasan:_
    - Dari Boxplot di atas, terlihat bahwa pada fitur **Infant Deaths**, **Percentage Expanditure**, dan **Population** terdapat Outliers, yang mana jika ini tidak ditangani maka akan berpengaruh pada performa model yang dibangun.

## Data Preparation
Berikut merupakan tahapan Data Preprocessing & Feature Engineering yang dilakukan pada project ini :
  1. **Handle Missing Values**, merupakan proses yang dilakukan untuk menangani data yang kosong/hilang. Pada project ini nilai yang kosong tersebut diisi dengan nilai **Mean**(Rata-Rata).

  2. **Remove Whitespaces**, merupakan proses yang dilakukan untuk menghapus spasi. Pada project ini penghapusan spasi dilakukan pada awal dan akhir kalimat nama features pada Datasets.
  3. **Dropping Column**, merupakan proses yang dilakukan untuk menghapus atribut/fitur yang berkorelasi lemah terhadap fitur **Life Expectancy**. Berdasarkan hasil analysis menggunakan **pairplot** dan **Heatmap**, maka fitur yang akan dihapus adalah Year, under-five deaths, thinness 1-19 years, dan Measles.
  4. **Mapping Country to Continent**, merupakan proses yang dilakukan untuk memperkecil lingkup informasi mengenai negara dengan cara membuat feature baru yaitu **Continent**, ini dilakukan karena jumlah negara yang terdapat pada Datasets ini terdiri dari 193 negara. Masing-masing negara akan di mapping ke benuanya masing-masing. Sebagai contoh negara **Korea** akan di mapping ke benua **Asia**.
  5. **Dropping feature(Country)**, merupakan proses yang dilakukan untuk menghapus feature **country**. Feature ini dihapus setelah pembuatan feature Continent.
  6. **Label Encoder**, merupakan proses yang dilakukan untuk merubah data kategorik menjadi numerik. Pada project ini proses label encoder di implementasi pada fitur **Status** dan **Continent**.
  7. **Outliers Removal**, merupakan proses yang dilakukan untuk menghilangkan outliers pada Datasets. Pada project ini akan diimpelemntasi metode **IQR(Interquartile Range)** untuk menangani outliers, dikarenakan ada beberapa feature yang memiliki kondisi **Left/Right Skewed**.
  8. **Train-Test Split**, merupakan proses yang dilakukan untuk membagi 2 Datasets menjadi data latih dan uji. Pada project ini Proportion pembagian datasets menjadi 90% Data Latih dan 10% Data Uji. berikut detail pembagiannya :

    | Total Data | train set | test set |
    | ---------- | --------- | -------- |
    |    2938    |   2644    |   294    |

  9. **Standardrization**, merupakan proses yang dilakukan untuk merubah values pada setiap features agar memilki skala yang sama.

Berikut merupakan **hasil** dari proses Outliers Removal :

![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/OutliersRemovalResult-1.png "After")
![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/OutliersRemovalResult-2.png "After")

_Penjelasan:_
  - Dari Boxplot di atas, terlihat jelas bahwa outliers yang sebelumnya terdapat pada fitur **Infant Deaths**, **Percentage Expanditure**, dan **Population** setelah dilakukan proses Outliers Removal menggunakan teknik **Capping**, menjadi hilang.

Berikut merupakan hasil dari pembuatan fitur **Continent** berdasarkan average dari **Life Expectancy**.

![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/LifeExpectancyAverageOverContinent.png "Continent")

_Penjelasan:_
  - Dari Bar Plot di atas terlihat bahwa angka harapan hidup tertinggi berada di benua eropa, sedangkan terendahnya berada di benua Africa.

## Modeling
Dalam membangun ML Model Linear Regression, project ini akan menggunakan 3 Algoritma berbeda, kemudian menentukan Algoritma mana yang memiliki performa terbaik. Algoritma yang akan diuji diantaranya K-NN, Random Forest, dan AdaBoost.

  1. K-NN

      K-NN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan **‘kesamaan fitur’** untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

      KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Itulah mengapa algoritma ini dinamakan K-Nearest Neighbor (sejumlah k tetangga terdekat). K-NN bisa digunakan untuk kasus Classification dan Regression.

2. Random Forest

      Random Forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.

      Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori **Ensemble Learning**. **Ensemble Learning** adalah suatu model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian.

3. AdaBoost

      AdaBoost merupakan Algorithm yang bekerja dengan membangun model dari data latih. Kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik.

      Awalnya, semua kasus dalam data latih memiliki weight atau bobot yang sama. Pada setiap tahapan, model akan memeriksa apakah observasi yang dilakukan sudah benar? Bobot yang lebih tinggi kemudian diberikan pada model yang salah sehingga mereka akan dimasukkan ke dalam tahapan selanjutnya. Proses iteratif ini berlanjut sampai model mencapai akurasi yang diinginkan.

Berikut nilai params yang akan digunakan untuk setiap Algoritma-nya:
  
  1. **K-NN**

  - | Params | Value | Keterangan |
    | :------: | :-----: | :----- 
    |  n_neighbors  | 10  | Jumlah Tetangga terdekat yang akan dicari. |

  2. **Random Forest**

  - | Params | Value | Keterangan |
    | :------: | :-----: | :----- |
    |  n_estimators  | 50  | Jumlah Trees (pohon) di Forest. |
    |  max_depth     | 16  | kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. |
    |  random_state  | 55  | Digunakan untuk mengontrol random number generator yang digunakan. | 
    |  n_jobs        | -1  | jumlah job (pekerjaan) yang digunakan secara paralel. |

  3. **AdaBoost**

  - | Params | Value | Keterangan |
    | :------: | :-----: | :----- | 
    |  learning_rate  | 0.05  | Bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting. |
    |  random_state   | 55    | Digunakan untuk mengontrol random number generator yang dikembangkan. |

## Evaluation
Setelah dilakukan proses Training, berikut merupakan hasil dari Training seluruh Model :

| Algorithm | train_mse |
| :------:  | :-----:   |
|  K-NN     | 8.013816	|
|  Random Forest  | 0.671714  |
|  AdaBoost | 14.070159	|

Berdasarkan hasil training, terlihat bahwa Algoritma **Random Forest** memiliki nilai mean squared error(mse) paling kecil, itu berarti Algoritma Random Forest lah yang memiliki performa paling baik diantara Algoritma pembanding lainnya. Untuk memastikannya lebih lanjut, ketiga model ini akan di evaluasi menggunakan data uji(**test set**).

Langkah selanjutnya adalah masing-masing model akan di evaluasi menggunakan data uji menggunakan metrics **Mean Squared Error(MSE)**. MSE adalah suatu metrics yang umum digunakan pada kasus Linear Regression, secara teknis selisih(**difference**) antara nilai sebenarnya dengan nilai prediksi disebut **error**. MSE bekerja dengan cara meghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Berikut merupakan persamaan dari MSE :

![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/MSE-Formulas.png "MSE")

_Keterangan:_
- N : Jumlah Datasets
- y_pred_i : Nilai Prediksi
- y_i : Nilai Sebenarnya

Berikut merupakan hasil dari pengujian model terhadap data uji:

| Algorithm | test_mse |
| :------:  | :-----:   |
|  K-NN     | 0.009676	|
|  Random Forest  | 0.004237  |
|  AdaBoost | 	0.015997	|

Dari hasil pengujian di atas, terlihat bahwa Algorithm Random Forest menunjukkan hasil **mse** lebih baik dibandingkan Algorithm lainnya, dengan nilai **mse** sebesar 0.004237. Untuk mempermudah pemahaman, berikut merupakan **Line Chart** nya.

![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/ModelEvaluation.png "Evaluation")

**Features** yang paling berpengaruh terhadap pelatihan model Random Forest dapat dilihat pada bar plot berikut, yang diurutkan secara **Descending**.

![Alt text](https://raw.githubusercontent.com/CarakaMuhamadRahman/LifeExpectancy-ML-Model/refs/heads/main/Images/FeatureImportance.png "FeatureImportance")

Sehingga dapat disimpulkan bahwa Algorithm yang paling optimal dalam melakukan prediksi terhadap Life Expectancy adalah **Random Forest**, dan **HIV/AIDS** merupakan feature yang paling berpengaruh selama pelatihan ML Model.