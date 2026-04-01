# 1. Histogram, Box Plot, Detect outliers for califronia housing dataset
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names) 

print("Dataset Shape:", df.shape)  
print("\nSummary Statistics:")
print(df.describe())  

df.hist(bins=30, figsize=(12, 10), layout=(4, 2), edgecolor='black')
plt.suptitle('Histogram of California Housing Features', fontsize=14)
plt.show()

print("\nDistribution Analysis (Skewness):")
for col in df.columns:
    skewness = df[col].skew()
    print(f"{col} - skewness: {skewness:.2f}") 

plt.figure(figsize=(12, 10))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Box Plot of California Housing Features")
plt.show()

print("\nOutlier detection using IQR method:")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")
