import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X_values=np.random.rand(100,1)
y_labels=np.array(['Class1' if x <= 0.5 else 'Class2' for x in X_values.flatten()])

X_train=X_values[:50]
y_train=y_labels[:50]
X_test=X_values[50:]
y_test=y_labels[50:]

def euclidean_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))

def knn_classify(X_train,y_train,X_test,k):
    y_pred=[]
    for test_points in X_test:
        distances=[euclidean_distance(test_points,train_point) for train_point in X_train]
        k_indices=np.argsort(distances)[:k]
        k_nearest_labels=[y_train[i] for i in k_indices]
        most_common_label=max(set(k_nearest_labels),key=k_nearest_labels.count)
        y_pred.append(most_common_label)
    return np.array(y_pred)

k_values=[1, 2, 3, 4, 5, 20, 30]
plt.figure(figsize=(12,8))

for i, k in enumerate(k_values,1):
    y_pred=knn_classify(X_train,y_train,X_test,k)
    plt.subplot(3,3,i)
    plt.scatter(X_test,y_test,color='red',marker='x',label='predicted label')
    plt.title(f'KNN with k={k}')
    plt.xlabel('X value')
    plt.ylabel('Class label')
    plt.legend(loc='best')
    plt.grid(True)

plt.tight_layout()
plt.show()

for k in k_values:
    y_pred=knn_classify(X_train,y_train,X_test,k)
    accuracy=np.mean(y_pred==y_test)
    print(f'Accuracy for k={k}: {accuracy:.2f}')
        