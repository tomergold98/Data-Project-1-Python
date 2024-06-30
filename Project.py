import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
sns.set_theme(style="dark")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

#1.2
dframe1 = pd.read_csv('input1_df.csv')

#1.3
print(dframe1.info())

#1.5
print()
print("Null values for each column: ")
print(dframe1.isnull().sum())

#1.8
print()
print(dframe1.head())
print(dframe1.tail())
print(dframe1.iloc[515:520])

#1.9
print()
print("Descriptive statistics Of the database: ")
print(dframe1.describe())

#1.10
#1.2
dframe2 = pd.read_csv('input2_df.csv')

#1.3
print(dframe2.info())

#1.5
print()
print("Null values for each column: ")
print(dframe2.isnull().sum())

#1.8
print()
print(dframe2.head())
print(dframe2.tail())
print(dframe2.iloc[515:520])

#1.9
print()
print("Descriptive statistics Of the database: ")
print(dframe2.describe())

#1.11
print()
outer_join_df = pd.merge(dframe1,dframe2 , on ='title', how= 'outer')
print("Null values for each column: ")
print(outer_join_df.isnull().sum())

#1.12
df1 = outer_join_df[['type','release_year','IMDb']]
#df1.to_csv('df1.csv')
#print(df1.dtypes)

df2 = outer_join_df[['title','Rotten Tomatoes']]
#df2.to_csv('df2.csv')
#print(df2.dtypes)

#2 2.1
def char_finder(date_frame, series_name):
    cnt=0
    print(series_name)
    for row in date_frame[series_name]:
        try:
            float(row)
        except ValueError:
            print(date_frame.loc[cnt,series_name], "-> at row: "+ str(cnt))
        cnt+=1

print()
char_finder(df1,'release_year')
char_finder(df1,'IMDb')
char_finder(df2,'Rotten Tomatoes')

def char_fixer(data_frame, series_name):
    cnt=0
    for row in data_frame[series_name]:
        try:
            float(row)
            pass
        except ValueError:
            data_frame = data_frame[data_frame.index != cnt]
        cnt+=1
    data_frame[series_name] = pd.to_numeric(data_frame[series_name], errors='coerce').astype('float64')
    data_frame.reset_index(drop=True, inplace=True)
    return data_frame

df1 = char_fixer(df1, 'release_year')
print()
print(df1.dtypes)
# df1.to_csv('df10.csv')

df1 = char_fixer(df1, 'IMDb')
print()
print(df1.dtypes)
#df1.to_csv('df15.csv')

df2 = char_fixer(df2, 'Rotten Tomatoes')
print()
print(df2.dtypes)
#df2.to_csv('df20.csv')

#2.2
def num_finder(data_frame, series_name):
    cnt=0
    print(series_name)
    for row in data_frame[series_name]:
        try:
            int(float(row))

        except ValueError:
            if row=='True' or row=='False':
                print(data_frame.loc[cnt, series_name], "-> at row:" + str(cnt))
            else:
                pass
        else:
            print(data_frame.loc[cnt, series_name], "-> at row:" + str(cnt))

        cnt+=1

print()
num_finder(df1,'type')
num_finder(df2,'title')

def num_fixer(data_frame, series_name):
    cnt=0
    for row in data_frame[series_name]:
        try:
            int(float(row))
        except ValueError:
            if row=='True' or row=='False':
                data_frame.drop([cnt], inplace=True)
            elif row == 'NaN':
                data_frame.loc[cnt, series_name] = np.nan
            else:
                pass
        else:
            data_frame.drop([cnt], inplace=True)
        cnt+=1

    data_frame[series_name] = data_frame[series_name].astype('string', errors='raise')
    data_frame.reset_index(drop=True, inplace=True)

num_fixer(df1, 'type')
print()
print(df1.dtypes)
#df1.to_csv("df25.csv")

num_fixer(df2, 'title')
print()
print(df2.dtypes)
#df2.to_csv("df30.csv")

#2.3
overall_mean = df1['release_year'].mean()
#print(overall_mean)
df1['release_year'].fillna(overall_mean, inplace=True)
#df1.to_csv('AVG.csv')

overall_mean1 = df1["IMDb"].mean()
#print(overall_mean1)
df1["IMDb"].fillna(overall_mean1,inplace=True)
#df1.to_csv('AVG1.csv')

overall_mean2 = df2['Rotten Tomatoes'].mean()
#print(overall_mean2)
df2['Rotten Tomatoes'].fillna(overall_mean2, inplace=True)
#df2.to_csv('AVG2.csv')

overall_mode1 = df1['type'].mode().iloc[0]
#print(overall_mode1)
df1['type'].fillna(overall_mode1, inplace=True)
#df1.to_csv('mode.csv')

overall_mode2 = df2['title'].mode().iloc[0]
#print(overall_mode2)
df2['title'].fillna(overall_mode2, inplace=True)
#df2.to_csv('mode2.csv')

#2.4
max = df1["release_year"].max()
#print(max)
result = []
for number in df1["release_year"]:
    result.append(abs(number)/max)
df1["release_year_norm"] = result
#df1.to_csv("norm.csv")

#2.5
duplicates = df1[df1.duplicated('type')]
print()
print(duplicates)
df1 = df1.drop_duplicates()
df1.reset_index(drop=True, inplace=True)
#df1.to_csv("normDup.csv")

#3
#1
df1.groupby('type').count().plot.pie(y = 'release_year', autopct='%.2f')
plt.title("Quantity of movies\TV shows by year ")
plt.show()

#2
columns = ['type', 'release_year']
df1[columns].hist(stacked=False,alpha=0.5)
plt.xlabel("Release year")
plt.ylabel("Number of types")
plt.title("Types by years ")
plt.show()

#3
sns.boxplot(x='release_year', y='type', data=df1, orient='h', width=0.6, showfliers=False)
sns.stripplot(x='release_year', y='type', data=df1, orient='h', color='black', alpha=0.5)
plt.title("Centralization for Movies and TV shows")
plt.show()

#4
scatter_matrix(df1, figsize=(10,10))
plt.tight_layout()
plt.show()

#5
df1.hist()
plt.tight_layout()
plt.show()

#6
plt.figure(figsize=(8, 6))
sc = plt.scatter(df1['IMDb'], df1['release_year'], c=range(len(df1)), cmap='viridis', s=100, alpha=0.8)
plt.colorbar(sc, label='Data Point Index')
plt.xlabel('IMDb')
plt.ylabel('release year')
plt.title('Sactter graph by years')
plt.grid(True)
plt.show()

#7
axs = df1.plot.area(figsize=(12, 4), subplots=True)
plt.show()

#8
plt.figure(figsize=(8, 6))
plt.bar(df1['release_year'], df1['IMDb'], color='skyblue')
plt.xlabel('Release year')
plt.ylabel('IMDb')
plt.title('Rate by years')
plt.grid(axis='y')
plt.show()

#9
fig = px.scatter(df1, x="release_year", y="IMDb", color="type")
fig.show()

#10
fig = px.box(df1, x='IMDb', y='type')
fig.show()

#4.1
X = df1.iloc[:, [1,2]].values
print(X)

wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('wcss')
plt.show()

kmeans = KMeans(n_clusters=5, n_init='auto', random_state=42)

y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

plt.scatter(X[y_kmeans == 0,0], X[y_kmeans == 0,1], s=100, c='red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1,0], X[y_kmeans == 1,1], s=100, c='blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2,0], X[y_kmeans == 2,1], s=100, c='green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3,0], X[y_kmeans == 3,1], s=100, c='cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4,0], X[y_kmeans == 4,1], s=100, c='magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[: , 1], s=200, c= 'black',label= 'Centroids')

plt.title('Clusters')
plt.legend()
plt.show()

#4.2
x = df1['release_year']
y = df1['IMDb']

a,b = np.polyfit(x,y,deg = 1)
y_est = a*x+b
y_err = x.std() * np.sqrt(1/len(x)+ (x-x.mean()) ** 2/np.sum((x-x.mean())**2))

fig, ax =  plt.subplots()
ax.plot(x,y_est,'-')
ax.fill_between(x,y_est - y_err, y_est + y_err, alpha = 0.2)
ax.plot(x,y,'o', color ='tab:brown')
fig.show()
plt.waitforbuttonpress()

