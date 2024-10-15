import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load Data
data = pd.read_csv('velocityx_data.csv')

# Data Cleaning
data.dropna(inplace=True)  # Drop missing values

# Descriptive Statistics
print(data.describe())

# Correlation Analysis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Clustering
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(data[['Fan Challenges Completed', 'Virtual Merchandise Purchases', 'Time on Live 360']])

# Visualize Clusters
sns.scatterplot(data=data, x='Fan Challenges Completed', y='Virtual Merchandise Purchases', hue='Cluster')
plt.title('User Clusters Based on Engagement')
plt.show()
