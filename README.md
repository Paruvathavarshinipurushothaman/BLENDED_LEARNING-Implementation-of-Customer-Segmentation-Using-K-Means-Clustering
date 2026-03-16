# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Collect the **Mall Customers dataset** containing features such as age, annual income, and spending score.
2. Preprocess the data by selecting relevant features and normalizing the values if required.
3. Choose the optimal number of clusters (K) using the **Elbow Method**.
4. Apply the **K-Means clustering algorithm** to group customers based on their purchasing behavior.
5. Analyze and visualize the clusters to understand different customer segments.

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
data = pd.read_csv('CustomerData.csv')
print(data.head())
print(data.columns)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
plt.figure(figsize=(10,5))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
```

## Output:
<img width="808" height="201" alt="image" src="https://github.com/user-attachments/assets/5f2344f2-24b8-40db-81f6-0eb9652b1a81" />
<img width="1041" height="493" alt="image" src="https://github.com/user-attachments/assets/2a09edf8-67b1-432a-a45b-3fe34432dc98" />
<img width="1152" height="637" alt="image" src="https://github.com/user-attachments/assets/6b49fd85-321f-4041-9544-fc075c06851e" />

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
