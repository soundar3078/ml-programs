import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

housing_df = pd.read_csv('housing.csv')
numerical_features = housing_df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.histplot(housing_df[feature],kde=True,bins=30,color='blue')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))
for i, feature in enumerate(numerical_features):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=housing_df[feature],color='orange')
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

print("Outliers detection:")
outliers_summary={}
for feature in numerical_features:
    Q1=housing_df[feature].quantile(0.25)
    Q3=housing_df[feature].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=housing_df[(housing_df[feature] < lower_bound) | (housing_df[feature] > upper_bound)]
    outliers_summary[feature]=len(outliers)
    print(f'{feature}:{len(outliers)}outliers')

print("\n Dataset summary:")
print(housing_df.describe())