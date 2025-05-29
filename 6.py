import numpy as np
import matplotlib.pyplot as plt

def lwr(x,X,y,tau):
    W = np.diag(np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * tau**2)))
    theta = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
    return x @ theta

np.random.seed(42)
X=np.linspace(0,2*np.pi,100)
y=np.sin(X) + 0.1*np.random.randn(100)
X_bias=np.c_[np.ones_like(X),X]

x_test=np.linspace(0,2*np.pi,200)
x_test_bias = np.c_[np.ones_like(x_test), x_test]
tau=0.5


y_pred = np.array([lwr(xi, X_bias, y, tau) for xi in x_test_bias])
plt.figure(figsize=(10,6))
plt.scatter(X,y,color='blue',label=f'LWR Fit (tau={tau})',linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Locally weighted regression')
plt.legend()
plt.grid(alpha=0.3)
plt.show()