import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('housing.csv')
numeric_data = data.select_dtypes(include=['float64','int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True,cmap='coolwarm',fmt='.2f',linewidth='0.5')
plt.title('Correlation matrix of california housing dataset')
plt.show()

sample_data=numeric_data.sample(n=1000,random_state=42)

pairplot=sns.pairplot(sample_data,diag_kind='kwe',plot_kws={'alpha':0.5})
pairplot.fig.suptitle('pair plot of california housing features(sample)',y=1.22)
plt.show()