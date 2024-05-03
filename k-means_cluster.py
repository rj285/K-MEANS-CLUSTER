from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
# %matplotlib inline

df = pd.read_csv("income.csv")
head1 = df.head()
print("INCOME DATA:- ",head1)

plt.scatter(df.Age, df['Income($)'])
plt.xlabel('AGE')
plt.ylabel('Income($)')
plt.savefig('income_data_plotting.png')
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("PREDICTED DATA:- ",y_predicted)

df['cluster'] = y_predicted
head2 = df.head()
print("CLUSTER ADDED TO THE DATA:- ",head2)

centroid_1 = km.cluster_centers_
print("CENTROID:- ",centroid_1)

df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age,df1['Income($)'],color='green',label='income1')
plt.scatter(df2.Age,df2['Income($)'],color='red',label='income2')
plt.scatter(df3.Age,df3['Income($)'],color='black',label='income3')
plt.scatter(km.cluster_centers_ [:,0], km.cluster_centers_ [:,1], color='purple', marker='*', label='centroid')
plt.legend()
plt.savefig('income_data_centroid(1).png')
plt.show()


scaler = MinMaxScaler()

scaler.fit(df[['Income($)']])
df['Income($)'] = scaler.transform(df[['Income($)']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])

print(df.head())

plt.scatter(df.Age,df['Income($)'])
plt.show()

km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print("PREDICTED DATA:- ",y_predicted)

centroid_2 = km.cluster_centers_
print("CENTROID:- ",centroid_2)

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'],color='green',label='income1')
plt.scatter(df2.Age,df2['Income($)'],color='red',label='income2')
plt.scatter(df3.Age,df3['Income($)'],color='black',label='income3')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.savefig('income_data_centroid(2).png')
plt.show()

sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.savefig('sum of squared error.png')
plt.show()