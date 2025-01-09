# %% [markdown]
# <h3>Nama: Caraka Rahman</h3>

# %% [markdown]
# <h1>Import Necessary Libraries</h1>

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

# %%
df = pd.read_csv('Life Expectancy Data.csv')
df = df.drop(columns=['Adult Mortality'])

df.info()
df.head()

# %%
df['Country'].value_counts()

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

# %%
# Before Handle Missing Values
df.isna().sum()

# %%
null_columns = df.columns[df.isna().any()]

null_columns

# %%
# Impute Using Mean
df.fillna(df[null_columns].mean(), inplace=True)

# %%
# After Handle Missing Values(Using Mean)
print(f'Total of Missing Values:\n{df.isna().sum()}\n')

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
# <h2>2.1 Univariate Analysis</h2>

# %% [markdown]
# <h4>2.1.1 Categorical Features</h4>

# %%
numerical_features = []
categorical_features = ['Country', 'Status']

for col in df.columns:
    if df[col].dtype != 'object':
        numerical_features.append(col)


# %% [markdown]
# <h5>A. Country</h5>

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
# <h5>B. Status</h5>

# %%
feature_status = categorical_features[1]
count = df[feature_status].value_counts()
percent = 100 * df[feature_status].value_counts(normalize=True)
df_status = pd.DataFrame({'Total Sample': count, 'Percentage': percent.round(2)})

print(df_status)
count.plot(kind='bar', title=feature_status)

# %% [markdown]
# <h4>2.1.2 Numerical Features</h4>

# %%
df.hist(bins=50, figsize=(20,20))
pl.suptitle('Histogram of Numerical Features', fontsize=32)
plt.show()

# %% [markdown]
# <h2>2.2 Multivariate Analysis</h2>

# %% [markdown]
# <h4>2.2.1 Categorical Features</h4>

# %% [markdown]
# <h4>A. Country</h4>

# %%
top_5_country = df.sort_values(by='Life expectancy', ascending=False)
top_5_country = top_5_country.head(5)
top_5_country['Country-Year'] = top_5_country['Country'] + '-' + top_5_country['Year'].astype(str)

top_5_country

# %%
sns.catplot(x='Country-Year', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, data=top_5_country, palette='husl')
plt.title('Life Expectancy Average Over Top 5 Countries')

# %% [markdown]
# <h4>B. Status</h4>

# %%
sns.catplot(x='Status', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, data=df, palette='husl')
plt.title('Life Expectancy Average Over Status')

# %% [markdown]
# <h4>2.2.2 Numerical Features</h4>

# %%
sns.pairplot(df, diag_kind='kde')

# %%
# Heatmap Visualization
plt.figure(figsize=(12,6))
correlation_matrix = df[numerical_features].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matrix Correlation of Numeric Features', size=20)

# %% [markdown]
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

# %%
df = df.drop(columns=['Year','under-five deaths','thinness  1-19 years','Measles'])

# %%
removed_features = ['Year','under-five deaths','thinness  1-19 years','Measles']

print(f'Num of Features Before Remove: {len(numerical_features)}')

for feature in removed_features:
    if feature in numerical_features:
        numerical_features.remove(feature)

print(f'Num of Features After Remove: {len(numerical_features)}')

# %%
from pycountry_convert import map_countries

map_countries = map_countries()

print(len(map_countries))

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

# %%
df['Country'].replace('Republic of Korea', 'South Korea', inplace=True)
df['Continent'] = df['Country'].apply(map_country2continent)

df = df.drop(columns=['Country'])

# %%
df.head()

# %%
df['Continent'].unique()

# %%
sns.catplot(x='Continent', y='Life expectancy', kind='bar', dodge=False, height=4, aspect=3, hue='Continent', data=df, palette='husl')
plt.title('Life Expectancy Average Over Continent')

# %% [markdown]
# <h3>Label Encoder</h3>

# %%
df['Continent'] = LabelEncoder().fit_transform(df['Continent'])
df['Status'] = LabelEncoder().fit_transform(df['Status'])

# %%
df.head()

# %%
df['Status'].value_counts()

# %%
df['Continent'].value_counts()

# %% [markdown]
# <h2>1.2 Data Transformation</h2>

# %%
df_features = df.columns

# %%
df_features

# %% [markdown]
# <h3>1.2.1 Outliers Removal</h3>

# %%
df_melted1 = pd.melt(df, value_vars=df_features[0:8])

plt.figure(figsize=(20, 20))
sns.boxplot(x='variable', y='value', data=df_melted1)


# %%
df_melted2 = pd.melt(df, value_vars=df_features[8::])

plt.figure(figsize=(20,20))
sns.boxplot(x='variable', y='value', data=df_melted2)             
                                    

# %% [markdown]
# <h4>Interquartile Range(IQR)</h4>
# <pre>Batas Bawah = (Q1-1.5) * IQR
# Batas Atas = (Q3+1.5) * IQR</pre>

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

# %%
for col in df[numerical_features]:
    outliersPercentage(df, col)

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

# %%
# df = trimming(df, numerical_features)

for col in df[numerical_features]:
    capping(df, df, col)

# %%
df.shape

# %%
for col in df[numerical_features]:
    outliersPercentage(df, col)

# %% [markdown]
# <h3>1.2.2 Outliers Removal Results</h3>

# %%
df_melted1 = pd.melt(df, value_vars=df_features[0:8])

plt.figure(figsize=(20, 20))
sns.boxplot(x='variable', y='value', data=df_melted1)

# %%
df_melted2 = pd.melt(df, value_vars=df_features[8::])

plt.figure(figsize=(20,20))
sns.boxplot(x='variable', y='value', data=df_melted2)  

# %% [markdown]
# <h1>3. Model Selection</h1>
# <ul>
#     <li>K-Nearest Neighbor</li>
#     <li>Random Forest</li>
#     <li>Boosting Algorithm</li>
# </ul>

# %% [markdown]
# <h2>3.1 Train-Test Split</h2>

# %%
X = df.drop(columns='Life expectancy')
y = df['Life expectancy']

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.1, random_state=123)

print(f'Total sample in whole dataset: {len(X)}')
print(f'Total sample in train dataset: {len(X_train)}')
print(f'Total sample in test dataset: {len(X_test)}')

# %% [markdown]
# <h2>1.2 Data Transformation(Continued)</h2>

# %% [markdown]
# <h3>1.2.2 Standardrization</h3>

# %%
numerical_features

# %%
numerical_features.remove('Life expectancy')

# %%
len(numerical_features)

# %%
scaler = StandardScaler()

scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

# %%
models = pd.DataFrame(index=['KNN','RandomForest','Boosting'],
                      columns=['train_mse', 'test_mse'])

# %% [markdown]
# <h2>3.1 K-Nearest Neighbor</h2>

# %%
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['KNN','train_mse'] = mean_squared_error(y_pred=knn.predict(X_train), y_true=y_train)

# %%
models.head()

# %% [markdown]
# <h2>3.2 Random Forest</h2>

# %%
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['RandomForest','train_mse'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

# %%
models.head()

# %% [markdown]
# <h2>3.3 Adaptive Boosting(AdaBoost)</h2>

# %%
from sklearn.ensemble import AdaBoostRegressor

adaboost = AdaBoostRegressor(learning_rate=0.05, random_state=55)
adaboost.fit(X_train, y_train)

models.loc['Boosting','train_mse'] = mean_squared_error(y_pred=adaboost.predict(X_train), y_true=y_train)

# %%
models.head()

# %% [markdown]
# <h1>4. Model Evaluation</h1>

# %%
# Standardrization Test Set
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# %%
X_test[numerical_features]

# %%
model_dict = {'KNN': knn,
              'RandomForest': RF,
              'Boosting': adaboost}

for name, model in model_dict.items():
    models.loc[name,'test_mse'] = mean_squared_error(y_pred=model.predict(X_test), y_true=y_test)/1e3

models.head()

# %%
models = models.reset_index(names=['Model'])
models.head()

# %%
plt.figure(figsize=(8,6))
sns.pointplot(x='Model', y='test_mse', data=models)
plt.xticks(rotation = 90)
plt.title('Model Comparison: MSE')
plt.tight_layout()
plt.show()

# %% [markdown]
# <h2>4.1 Feature Importance</h2>

# %%
feature_importance = RF.feature_importances_

feature_importance = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

feature_importance

# %%
feature_importance.plot.bar(x='Feature', y='Importance')
plt.title('Feature Importance of Random Forest Algorithm')

# %% [markdown]
# <h2>4.2 Prediction</h2>

# %%
prediction = X_test.iloc[:1].copy()
pred_dict = {'y_true': y_test[:1]}

for name,model in model_dict.items():
    pred_dict[f'{name}_pred'] = model.predict(prediction).round(1)

pred_dict


