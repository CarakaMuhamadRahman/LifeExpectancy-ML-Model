# %% [markdown]
# <h3>Nama: Caraka Rahman</h3>

# %% [markdown]
# <h1>Import Necessary Libraries</h1>

# %% [markdown]
# <p>Di bawah ini merupakan libraries yang akan digunakan pada project ini.</p>

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import pylab as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2, convert_continent_code_to_continent_name
from sklearn.preprocessing import LabelEncoder

# %% [markdown]
# <h1>Load Datasets</h1>

# %% [markdown]
# <p>Datasets yang digunakan pada project ini merupakan Life Expectancy Data yang berasal dari <strong>World Health Organization(WHO)</strong>.</p>
# 
# <p>Function <strong>read_csv</strong> dari pandas digunakan untuk membaca file <strong>.csv</strong>.</p>

# %%
df = pd.read_csv('Life Expectancy Data.csv')
df = df.drop(columns=['Adult Mortality'])

df.info()
df.head()

# %% [markdown]
# <p>Dari informasi di atas dapat dilihat bahwa Datasets memiliki data berjumlah 2938 data, dengan jumlah feature sebanyak 21.</p>

# %% [markdown]
# <p>Kode di bawah digunakan untuk mengetahui ada berapa total negara yang terdapat pada Datasets.</p>

# %%
df['Country'].value_counts()

# %% [markdown]
# <p>Hasil dari fungsi <strong>value_counts()</strong> di atas dapat dilihat bahwa jumlah negara pada Datasets berjumlah 193 negara.</p>

# %%
# country_dict = {}
# num = 1

# for country in df['Country']:
#     if country not in country_dict:
#         country_dict[country] = num
#     elif country in country_dict:
#         continue
#     num += 1

# print(country_dict)

# %%
# country_dict['Zimbabwe']

# %% [markdown]
# <h1>1. Data Preprocessing</h1>

# %% [markdown]
# <h2>1.1 Data Cleaning</h2>

# %% [markdown]
# <h3>1.1.1 Checking Missing Values</h3>

# %% [markdown]
# <p>Checking Missing Values dilakukan untuk mengecek kondisi suatu Datasets, apakah terdapat data yang kosong atau tidak. Jika terdapat data yang kosong, maka akan dilakukan pengisian data menggunakan metode tertentu.</p>

# %%
# Before Handle Missing Values
df.isna().sum()

# %% [markdown]
# <p>Dari hasil pengecekan di atas, ternyata pada beberapa features terdapat missing values.</p>

# %% [markdown]
# <p>Berikut merupakan features yang memilki missing values.</p>

# %%
null_columns = df.columns[df.isna().any()]

null_columns

# %% [markdown]
# <p>Berdasarkan kode di atas, dapat disimpulkan bahwa ada sebanyak 13 features yang memiliki missing values.</p>
# <p>Selanjutnya akan dilakukan proses <strong>Imputasi</strong> untuk mengganti nilai kosong tersebut, menggunakan nilai <strong>mean</strong>.</p>

# %%
# Impute Using Mean
df.fillna(df[null_columns].mean(), inplace=True)

# %% [markdown]
# <p>Berikut merupakan hasil dari proses imputasi, dapat dilihat bahwa features yang sebelumnya terdapat missing values, setelah dilakukan proses imputasi menggunakan nilai mean total missing values nya menjadi 0.</p>

# %%
# After Handle Missing Values(Using Mean)
print(f'Total of Missing Values:\n{df.isna().sum()}\n')

# %% [markdown]
# <p>Kode di bawah berfungsi untuk mengecek jika terdapat data duplicate pada Datasets.</p>

# %% [markdown]
# <h3>1.1.2 Checking Duplicate Values</h3>

# %%
print(f'Jumlah Data Duplicate: {len(df)-len(df.drop_duplicates())}')

# %% [markdown]
# <p>Dari output di atas terlihat bahwa tidak ada data Duplicate pada Datasets.</p>

# %% [markdown]
# <p>Dikarenakan pada beberapa penamaan features terdapat spasi di awal dan akhir, maka untuk mempermudah proses pengerjaan spasi-spasi tersebut akan dihapus menggunakan function <strong>strip()</strong></p>

# %%
df.columns = df.columns.str.strip()

df.head()

# %% [markdown]
# <h3>Append DataFrame</h3>

# %%
# def appendCountries(index):
#     df = df_categories[df_categories['index'] == index]
#     countryValue = None
#     for i, values in df.iterrows():
#         countryValue = values['Country']
    
#     return countryValue

# def appendStatus(index):
#     df = df_categories[df_categories['index'] == index]
#     statusValue = None
#     for i, values in df.iterrows():
#         statusValue = values['Status']
    
#     return statusValue

# %%
# df['Country'] = df['index'].apply(appendCountries)
# df['Status'] = df['index'].apply(appendStatus)

# df.head(10)

# %%
# df.head(10)

# %%
# # Remove 'Index' Feature
# df = df.drop(columns=['index'])

# df.head()

# %% [markdown]
# <h1>2. Data Analysis</h1>

# %% [markdown]
# <p>Selanjutnya akan dilakukan proses Data Analysis yang dibagi menjadi 2 bagian, Univariate dan Multivariate Analysis.</p>

# %% [markdown]
# <h2>2.1 Univariate Analysis</h2>

# %% [markdown]
# <p>Univariate Analysis merupakan proses analisis data yang hanya berfokus pada 1 fitur saja.</p>

# %% [markdown]
# <h4>2.1.1 Categorical Features</h4>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengelompokkan nama feature kategorik dan numerik, yang dipisahkan dengan 2 variables berbeda, yaitu numerical_features dan categorical_features.</p>
# <p>Nantinya variables ini akan digunakan untuk mempermudah proses Data Analysis.</p>

# %%
numerical_features = []
categorical_features = ['Country', 'Status']

for col in df.columns:
    if df[col].dtype != 'object':
        numerical_features.append(col)

print(f'Jumlah feature numerik: {len(numerical_features)}\n')
print(f'Jumlah feature kategorik: {len(categorical_features)}')

# %% [markdown]
# <p>Dari hasil di atas dapat dilihat bahwa features numerik berjumlah 19 dan kategorik berjumlah 2.</p>

# %% [markdown]
# <h5>A. Country</h5>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk menampilakan 10 besar negara yang paling banyak muncul pada Datasets.</p>

# %%
feature_country = categorical_features[0]
count = df[feature_country].value_counts()
percent = 100 * df[feature_country].value_counts(normalize=True)
df_country = pd.DataFrame({'Total Sample': count, 'Percentage': percent.round(3)})
df_country = df_country.reset_index()
df_country = df_country.sort_values(by='Percentage', ascending=False)

print(df_country.head(10))
# count.plot(kind='bar', title='Top 5 Countries')
df_country = df_country.head(10)
plt.figure(figsize=(100,100))
df_country.plot.bar(x='Country', y='Percentage', rot=90)
pl.suptitle('Top 10 Countries')
plt.xticks(fontsize=7)

plt.show()

# %% [markdown]
# <p>Dari Bar Plot di atas, dapat dilihat bahwa 10 besar negara yang paling banyak muncul pada Datasets diantaranya Afghanistan, Azerbaijan, Bahrain, Algeria, dll.</p>

# %% [markdown]
# <h5>B. Status</h5>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk menampilkan jumlah sample Data pada masing-masing status(Developing & Developed)</p>

# %%
feature_status = categorical_features[1]
count = df[feature_status].value_counts()
percent = 100 * df[feature_status].value_counts(normalize=True)
df_status = pd.DataFrame({'Total Sample': count, 'Percentage': percent.round(2)})

print(df_status)
count.plot(kind='bar', title=feature_status)

# %% [markdown]
# <p>Dari Bar Plot di atas, terlihat bahwa jumlah sample data pada Status Developing(82.57%) lebih banyak dibandingkan dengan status Developed(17.43%).</p>

# %% [markdown]
# <h4>2.1.2 Numerical Features</h4>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk melihat persebaran nilai mengggunakan Histogram pada setiap features nya.</p>

# %%
df.hist(bins=50, figsize=(20,20))
pl.suptitle('Histogram of Numerical Features', fontsize=32)
plt.show()

# %% [markdown]
# <p>Dari Histogram di atas, terlihat bahwa pada beberapa feature memilki kondisi <strong>Left/Right Skewed</strong>, sebagai contoh pada fitur Alcohol dan Polio. Tentu kondisi ini tidak baik jika tidak ditangani dengan baik, maka untuk menanganinya akan diimplementasi metode IQR(InterQuartile Range).</p>

# %% [markdown]
# <h2>2.2 Multivariate Analysis</h2>

# %% [markdown]
# <p>Multivariate Analysis merupakan proses analisis data terhadap 2 atau lebih features.</p>

# %% [markdown]
# <h4>2.2.1 Categorical Features</h4>

# %% [markdown]
# <h4>A. Country</h4>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengurutkan DataFrame berdasarkan fitur Life Expectancy yang diurutkan secara Descending. Kemudian mengambil 5 negara teratas berdasarkan nilai Life Expectancy nya.</p>

# %%
top_5_country = df.sort_values(by='Life expectancy', ascending=False)
top_5_country = top_5_country.head(5)
top_5_country['Country-Year'] = top_5_country['Country'] + '-' + top_5_country['Year'].astype(str)

top_5_country

# %% [markdown]
# <p>Dari hasil pengurutan di atas, terlihat bahwa New Zealand, Finland, Belgium, Spain, dan Sweden merupakan negara dengan nilai Life Expectancy terbesar.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menampilkan Bar Plot mengenai Rata-Rata Life Expectancy berdasarkan 5 negara teratas yang memiliki nilai Life Expectancy tertinggi.</p>

# %%
sns.catplot(x='Country-Year', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, data=top_5_country, palette='husl')
plt.title('Life Expectancy Average Over Top 5 Countries')

# %% [markdown]
# <h4>B. Status</h4>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menampilkan Bar Plot mengenai Rata-Rata Life Expectancy berdasarkan feature Status.</p>

# %%
sns.catplot(x='Status', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, data=df, palette='husl')
plt.title('Life Expectancy Average Over Status')

# %% [markdown]
# <p>Berdasarkan Bar Plot di atas dapat dilihat bahwa negara yang maju(Developed) memilki rata-rata nilai Life Expectancy yang lebih besar dibandingkan dengan negara berkembang(Developing).</p>

# %% [markdown]
# <h4>2.2.2 Numerical Features</h4>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk melihat korelasi antar features, terutama dengan feature label yaitu <strong>Life Expectancy</strong>.</p>

# %%
sns.pairplot(df, diag_kind='kde')

# %% [markdown]
# <p>Dari Visualization di atas, terdapat beberapa feature yang memiliki korelasi yang lemah terhadap feature Life Expectancy, contohnya <strong>Year</strong> dan <strong>Measles</strong>.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat korelasi antar features menggunakan Heatmap.</p>

# %%
# Heatmap Visualization
plt.figure(figsize=(12,6))
correlation_matrix = df[numerical_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrix Correlation of Numeric Features', size=20)

# %% [markdown]
# <p>Berikut merupakan beberapa insight yang didapatkan dari 2 Visualizations di atas :</p>
# <ul>
#     <li>Infant Deaths = -0.2 (Correlation Lemah)</li>
#     <li>Measles = -0.16 (There is no patterns)</li>
#     <li>Under-Five Deaths = -0.22 (Correlation Lemah)</li>
#     <li>HIV/AIDS = -0.56 (Correlation Lemah)</li>
#     <li>Population = -0.02 (There is no patterns)</li>
#     <li>1-19 Years = -0.47 (There is no patterns)</li>
#     <li>5-9 Years = -0.47 (Correlation tinggi)</li>
# </ul>
# 
# 
# 
# 
# 
# 
# 

# %% [markdown]
# <h2>1.1 Data Cleaning(Continued)</h2>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menghapus feature yang memilki korelasi lemah terhadap feature Life Expectancy.</p>

# %%
df = df.drop(columns=['Year','under-five deaths','thinness  1-19 years','Measles'])

# %%
removed_features = ['Year','under-five deaths','thinness  1-19 years','Measles']

print(f'Num of Features Before Remove: {len(numerical_features)}')

for feature in removed_features:
    if feature in numerical_features:
        numerical_features.remove(feature)

print(f'Num of Features After Remove: {len(numerical_features)}')

# %% [markdown]
# <p>Berdasarkan hasil di atas, dapat dilihat bahwa jumlah numerical features yang sebelumnya berjumlah 19 setelah dilakukan penghapusan menjadi berjumlah 15 features.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat jumlah negara yang terdapat pada Library <strong>pycountry_convert</strong>.</p>

# %%
from pycountry_convert import map_countries

map_countries = map_countries()

print(len(map_countries))

# %% [markdown]
# <p>Dari hasil di atas terlihat bahwa jumlah negara pada Library pycountry_convert berjumlah 460 negara.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melakukan mapping dari country ke continent.</p>

# %%
def map_country2continent(country_name):
    try:
        code= country_alpha2_to_continent_code(country_name_to_country_alpha2(country_name))
        continent = convert_continent_code_to_continent_name(code)
        return continent
    except Exception as e:
        if str(e) == '"Invalid Country Alpha-2 code: \'TL\'"':
            continent = 'Asia'
            return continent

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengeksekusi function <strong>map_country2continent</strong>.</p>

# %%
df['Country'].replace('Republic of Korea', 'South Korea', inplace=True)
df['Continent'] = df['Country'].apply(map_country2continent)

df = df.drop(columns=['Country'])

# %%
df.head()

# %% [markdown]
# <p>DataFrame di atas merupakan hasil dari eksekusi function <strong>map_country2continent</strong>. Dimana dibuat satu feature baru bernama <strong>Continent</strong>. Kemudian menghapus feature <strong>Country</strong>, dikarenakan sudah tidak akan digunakan lagi.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat value unique pada feature <strong>Continent</strong> yang baru saja dibuat.</p>

# %%
df['Continent'].unique()

# %% [markdown]
# <p>Terlihat bahwa negara yang sebelumnya berjumlah 193, berhasil di Mapping kedalam 6 benua berbeda.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat Rata-Rata nilai Life Expectancy di tiap benuanya.</p>

# %%
sns.catplot(x='Continent', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, hue='Continent', data=df, palette='husl')
plt.title('Life Expectancy Average Over Continent')

# %% [markdown]
# <p>Dari Bar Plot di atas terlihat jelas bahwa Rata-Rata Life Expectancy <strong>terendah</strong> berada di benua Africa, dan <strong>tertinggi</strong> berada di benua Eropa.</p>

# %% [markdown]
# <h3>Label Encoder</h3>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melakukan convert dari data kategorik menjadi numerik pada feature <strong>Continent</strong> dan <strong>Status</strong>.</p>

# %%
df['Continent'] = LabelEncoder().fit_transform(df['Continent'])
df['Status'] = LabelEncoder().fit_transform(df['Status'])

# %%
df.head()

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat value pada features Status dan juga Continent setelah dilakukan Label Encoder.</p>

# %%
df['Status'].value_counts()

# %%
df['Continent'].value_counts()

# %% [markdown]
# <h2>1.2 Data Transformation</h2>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengambil seluruh features pada DataFrame dan di masukkan kedalam variable df_features.</p>

# %%
df_features = df.columns

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menampilkan value pada variable df_features yang berisi nama-nama features pada DataFrame.</p>

# %%
df_features

# %% [markdown]
# <p>Output di atas merupakan features yang terdapat pada DataFrame.</p>

# %% [markdown]
# <h3>1.2.1 Outliers Removal</h3>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk mengidentifikasi Outliers pada seluruh features pada DataFrame.</p>

# %%
df_melted1 = pd.melt(df, value_vars=df_features[0:8])

plt.figure(figsize=(20, 20))
sns.boxplot(x='variable', y='value', data=df_melted1)


# %%
df_melted2 = pd.melt(df, value_vars=df_features[8::])

plt.figure(figsize=(20,20))
sns.boxplot(x='variable', y='value', data=df_melted2)             
                                    

# %% [markdown]
# <p>Dari Boxplot di atas, terlihat bahwa pada fitur <strong>Infant Deaths</strong>, <strong>Percentage Expanditure</strong>, dan <strong>Population</strong> terdapat Outliers, yang mana jika ini tidak ditangani maka akan berpengaruh pada performa model yang dibangun.</p>

# %% [markdown]
# <h4>Interquartile Range(IQR)</h4>
# <pre>Batas Bawah = (Q1-1.5) * IQR
# Batas Atas = (Q3+1.5) * IQR</pre>

# %% [markdown]
# <p>Function di bawah berfungsi untuk menghasilkan Outliers Threshold, menghitung Outlier Percentage pada setiap features nya, dan mengecek apakah pada suatu feature terdapat Outliers atau tidak.</p>

# %%
# Outliers Removal Equation
def outliersThreshold(DataFrame, columns, q1=0.25, q3=0.75):
    Q1 = DataFrame[columns].quantile(q1)
    Q3 = DataFrame[columns].quantile(q3)
    IQR = Q3 - Q1
    up_limit = Q3 + 1.5 * IQR
    low_limit = Q1 - 1.5 * IQR
    
    return low_limit,up_limit

# Outlier Percentage
def outliersPercentage(DataFrame, columns):
    low_limit,up_limit = outliersThreshold(DataFrame, columns)
    outliers = [x for x in DataFrame[columns] if x > up_limit or x < low_limit]
    print(columns)
    print(f'Outliers Percentage: {(len(outliers)/(DataFrame[columns].shape[0])) * 100} %')
    print('-----------------------------------')

# Checking Outliers
def checkOutliers(DataFrame, columns):
    low_limit,up_limit = outliersThreshold(DataFrame, columns)
    outliers = (DataFrame[columns] > up_limit) | (DataFrame[columns] < low_limit)

    if outliers.any():
        return True
    else:
        return False

# %% [markdown]
# <p>Berikut merupakan contoh dari implementasi function <strong>outliersPercentage</strong>.</p>

# %%
for col in df[numerical_features]:
    outliersPercentage(df, col)

# %% [markdown]
# <p>Setelah function <strong>outliersPercentage</strong> dijalankan, terlihat bahwa pada beberapa features terdapat Outliers, ditandai dengan persentase yang lebih dari 0%.</p>

# %% [markdown]
# <p>Function di bawah ini berfungsi untuk mengimpelemntasi teknik Outliers Removal, diantaranya ada <strong>Trimming</strong> dan <strong>Capping</strong>.</p>

# %%
def capping(DataFrame, DataFrame2, columns):
    low_limit, up_limit = outliersThreshold(DataFrame, columns)

    DataFrame.loc[(DataFrame[columns] < low_limit), columns] = low_limit
    DataFrame.loc[(DataFrame[columns] > up_limit), columns] = up_limit
    # DataFrame2.loc[(DataFrame2[columns] < low_limit), columns] = low_limit
    # DataFrame2.loc[(DataFrame2[columns] > up_limit), columns] = up_limit

def trimming(DataFrame, columns):
    df = DataFrame

    for col in columns:
        low_limit, up_limit = outliersThreshold(df, col)
        df = df[df[col] < up_limit]
        df = df[df[col] > low_limit]

    return df

# %% [markdown]
# <p>Di bawah ini merupakan kode untuk menjalankan function <strong>capping</strong>.</p>

# %%
# df = trimming(df, numerical_features)

for col in df[numerical_features]:
    capping(df, df, col)

# %% [markdown]
# <p>Di bawah ini merupakan kode untuk melihat Dimension dari DataFrame setelah diimplementasi Outliers Removal.</p>

# %%
df.shape

# %% [markdown]
# <p>Dari Dimension di atas terlihat bahwa tidak ada perubahan pada jumlah Data setelah dilakukan Outliers Removal, ini dikarenakan yang diimplementasi merupakan metode Capping, yang mana bekerja dengan cara Imputasi, sehingga tidak membuang Outliersnya.</p>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk melihat persentase Outliers setelah dilakukan Outliers Removal.</p>

# %%
for col in df[numerical_features]:
    outliersPercentage(df, col)

# %% [markdown]
# <p>Dari output di atas, terlihat bahwa persentase outliers pada setiap features menjadi <strong>0%</strong>. Itu berarti penerapan Outliers Removal berjalan dengan sukses.</p>

# %% [markdown]
# <h3>1.2.2 Outliers Removal Results</h3>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk mengidentifikasi Outliers pada seluruh features setelah diimplementasi Outliers Removal.</p>

# %%
df_melted1 = pd.melt(df, value_vars=df_features[0:8])

plt.figure(figsize=(20, 20))
sns.boxplot(x='variable', y='value', data=df_melted1)

# %%
df_melted2 = pd.melt(df, value_vars=df_features[8::])

plt.figure(figsize=(20,20))
sns.boxplot(x='variable', y='value', data=df_melted2)  

# %% [markdown]
# <p>Dari Boxplot di atas, terlihat bahwa outliers pada features <strong>Infant Deaths</strong>, <strong>Percentage Expanditure</strong>, dan <strong>Population</strong> telah berhasil ditangani.</p>

# %% [markdown]
# <h1>3. Model Selection</h1>
# <ul>
#     <li>K-Nearest Neighbor</li>
#     <li>Random Forest</li>
#     <li>Boosting Algorithm</li>
# </ul>

# %% [markdown]
# <h2>3.1 Train-Test Split</h2>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk membagi Datasets menjadi Data Latih dan Data Uji, dengan Proportion <strong>90% Data Latih</strong> dan <strong>10% Data Uji</strong>.</p>

# %%
X = df.drop(columns='Life expectancy')
y = df['Life expectancy']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=123)

print(f'Total sample in whole dataset: {len(X)}')
print(f'Total sample in train dataset: {len(X_train)}')
print(f'Total sample in test dataset: {len(X_test)}')

# %% [markdown]
# <p>Setelah dijalankan, terlihat bahwa jumlah data pada Data Latih sebanyak <strong>2644</strong> data, dan pada data uji sebanyak <strong>294</strong> data.</p>

# %% [markdown]
# <h2>1.2 Data Transformation(Continued)</h2>

# %% [markdown]
# <h3>1.2.2 Standardrization</h3>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menampilkan nama-nama features yang merupakan data numerik.</p>

# %%
numerical_features

# %% [markdown]
# <p>Output di atas merupakan nama-nama features yang merupakan data numerik.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk menghapus value <strong>Life expectancy</strong> dari suatu List <strong>numerical_features</strong>. Penghapusan dilakukan karena <strong>Life Expectancy</strong> merupakan feature Label/Target.</p>

# %%
numerical_features.remove('Life expectancy')

# %% [markdown]
# <p>Berikut merupakan jumlah element pada List <strong>numerical_features</strong> setelah dilakukan penghapusan.</p>

# %%
len(numerical_features)

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengimplementasi Standardrization pada data latih.</p>

# %%
scaler = StandardScaler()

scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

# %% [markdown]
# <p>Output di atas merupakan hasil dari implementasi Standardrization, dimana bisa dilihat bahwa seluruh values berada di skala yang sama.</p>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk membuat suatu DataFrame bernama <strong>models</strong> yang nantinya digunakan untuk menyimpan nilai mean-squared error hasil dari proses pelatihan dan pengujian.</p>

# %%
models = pd.DataFrame(index=['KNN','RandomForest','Boosting'],
                      columns=['train_mse', 'test_mse'])

# %% [markdown]
# <h2>3.1 K-Nearest Neighbor</h2>

# %% [markdown]
# <p>Berikut merupakan code untuk Training ML Model menggunakan Algorithm K-Nearest Neighbor.</p>

# %%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['KNN','train_mse'] = mean_squared_error(y_pred=knn.predict(X_train), y_true=y_train)

# %% [markdown]
# <p>Berikut merupakan nilai mse setelah proses pelatihan.</p>

# %%
models.head()

# %% [markdown]
# <h2>3.2 Random Forest</h2>

# %% [markdown]
# <p>Berikut merupakan code untuk Training ML Model menggunakan Algorithm Random Forest.</p>

# %%
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['RandomForest','train_mse'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

# %% [markdown]
# <p>Berikut merupakan nilai mse setelah proses pelatihan.</p>

# %%
models.head()

# %% [markdown]
# <h2>3.3 Adaptive Boosting(AdaBoost)</h2>

# %% [markdown]
# <p>Berikut merupakan code untuk Training ML Model menggunakan Algorithm AdaBoost.</p>

# %%
from sklearn.ensemble import AdaBoostRegressor

adaboost = AdaBoostRegressor(learning_rate=0.05, random_state=55)
adaboost.fit(X_train, y_train)

models.loc['Boosting','train_mse'] = mean_squared_error(y_pred=adaboost.predict(X_train), y_true=y_train)

# %% [markdown]
# <p>Berikut merupakan nilai mse setelah proses pelatihan.</p>

# %%
models.head()

# %% [markdown]
# <h1>4. Model Evaluation</h1>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk mengimplementasi Standardrization pada data uji.</p>

# %%
# Standardrization Test Set
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# %% [markdown]
# <p>Berikut merupakan hasil dari impelemntasi Standardrization terhadap data latih.</p>

# %%
X_test[numerical_features]

# %% [markdown]
# <p>Bisa dilihat bahwa sekarang seluruh values berada di skala yang sama.</p>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk melakukan evaluasi model menggunakan data uji.</p>

# %%
model_dict = {'KNN': knn,
              'RandomForest': RF,
              'Boosting': adaboost}

for name, model in model_dict.items():
    models.loc[name,'test_mse'] = mean_squared_error(y_pred=model.predict(X_test), y_true=y_test)/1e3

models.head()

# %% [markdown]
# <p>Dari hasil evaluasi model terhadap data latih dan uji, terlihat bahwa model dengan Algoritma <strong>Random Forest</strong> lah yang memiliki performa paling optimal berdasarkan metrics mse.</p>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melakukan reset index.</p>

# %%
models = models.reset_index(names=['Model'])
models.head()

# %% [markdown]
# <p>Output di atas merupakan kondisi dari DataFrame <strong>"models"</strong> setelah dilakukan reset index.</p>

# %% [markdown]
# <p>Kode di bawah berfungsi untuk menampilkan Line Chart dari performa model.</p>

# %%
plt.figure(figsize=(8,6))
sns.pointplot(x='Model', y='test_mse', data=models)
plt.xticks(rotation = 90)
plt.title('Model Comparison: MSE')
plt.tight_layout()
plt.show()

# %% [markdown]
# <p>Dari Line Chart di atas, dapat disimpulkan bahwa model dengan Algorithm <strong>RandomForest</strong> merupakan model yang paling optimal berdasarkan metrics mse.</p>

# %% [markdown]
# <h2>4.1 Feature Importance</h2>

# %% [markdown]
# <p>Kode di bawah ini berfungsi untuk melihat feature mana saja kah yang paling berperan selama pelatihan Model, kemudian mengurutkannya secara Descending.</p>

# %%
feature_importance = RF.feature_importances_

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

feature_importance

# %% [markdown]
# <p>Dari DataFrame <strong>feature_importance</strong> di atas, dapat dilihat bahwa feature <strong>HIV/AIDS</strong> merupakan fitur yang paling berperan selama proses pelatihan model.</p>

# %% [markdown]
# <p>Berikut merupakan kode untuk menampilkan Bar Plot dari <strong>feature_importance<strong>.</p>

# %%
feature_importance.plot.bar(x='Feature', y='Importance')
plt.title('Feature Importance of Random Forest Algorithm')

# %% [markdown]
# <h2>4.2 Prediction</h2>

# %% [markdown]
# <p>Berikut merupakan kode untuk melakukan prediksi.</p>

# %%
prediction = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test[:1]}

for name,model in model_dict.items():
    pred_dict[f'{name}_pred'] = model.predict(prediction).round(1)

pred_dict

# %% [markdown]
# <p>Dari hasil prediksi ketiga model di atas, terlihat bahwa model RandomForest lah yang menghasilkan prediksi paling mendekati nilai aslinya dibandingkan dengan 2 model lainnya. Ini berarti bahwa hasil dari perhitungan mse sudah tepat.</p>


