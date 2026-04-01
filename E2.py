# 2. Correlation Matrix, Heatmap, pairplot for california dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns = housing.feature_names)

# print("Basic Information on the datasets")
# print(df.info())

df['MedHouseVal'] = housing.target
print("First 5 elements:")
print(df.head())

correlation_matrix = df.corr()
print("Correlation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt='.2f', annot=True)
plt.suptitle("Heatmap for correlation Matrix")
plt.show()

# selected_features = ['MedInc', 'HouseAge', 'AveRooms ', 'AveBedrms', 'Population', 'AveOccup']
cols = df.columns
sns.pairplot(df[list(cols[0:5])], diag_kind='kde', height=1.2, aspect=1.1)
plt.suptitle("Pair Plot for selected features", y=1.08)
plt.show()
