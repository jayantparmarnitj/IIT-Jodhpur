
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import time

np.random.seed(42)

# ---------- Data ----------
digits = load_digits()
X = digits.data.astype(np.float64)
y = digits.target.reshape(-1,1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False, categories='auto')
Y = encoder.fit_transform(y)

X_train, X_test, Y_train, Y_test, y_train_labels, y_test_labels = train_test_split(
    X, Y, y.ravel(), test_size=0.2, random_state=42, stratify=y
)

# ---------- Neural Network ----------
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01,
                 activation='relu', momentum=0.0, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.activation = activation
        self.momentum = momentum
        
        # Xavier/Glorot initialization
        limit1 = np.sqrt(6 / (input_size + hidden_size))
        self.W1 = np.random.uniform(-limit1, limit1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        limit2 = np.sqrt(6 / (hidden_size + output_size))
        self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        
        # momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
    
    def _relu(self, Z): return np.maximum(0, Z)
    def _relu_deriv(self, Z): return (Z > 0).astype(float)
    def _sigmoid(self, Z): return 1.0 / (1.0 + np.exp(-Z))
    def _sigmoid_deriv(self, Z):
        A = self._sigmoid(Z); return A * (1 - A)
    def _softmax(self, Z):
        Zs = Z - np.max(Z, axis=1, keepdims=True)
        ex = np.exp(Zs)
        return ex / np.sum(ex, axis=1, keepdims=True)
    
    def _forward(self, X):
        Z1 = X.dot(self.W1) + self.b1
        if self.activation == 'sigmoid':
            A1 = self._sigmoid(Z1)
        else:
            A1 = self._relu(Z1)
        Z2 = A1.dot(self.W2) + self.b2
        A2 = self._softmax(Z2)
        return {'X': X, 'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    
    def _loss(self, A2, Y):
        m = Y.shape[0]
        eps = 1e-12
        return -np.sum(Y * np.log(A2 + eps)) / m
    
    def _backward(self, cache, Y):
        m = Y.shape[0]
        X = cache['X']; Z1 = cache['Z1']; A1 = cache['A1']; A2 = cache['A2']
        dZ2 = (A2 - Y) / m
        dW2 = A1.T.dot(dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(self.W2.T)
        if self.activation == 'sigmoid':
            dZ1 = dA1 * self._sigmoid_deriv(Z1)
        else:
            dZ1 = dA1 * self._relu_deriv(Z1)
        dW1 = X.T.dot(dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def _update(self, grads):
        beta = self.momentum
        # velocity update
        self.vW1 = beta * self.vW1 - self.lr * grads['dW1']
        self.vb1 = beta * self.vb1 - self.lr * grads['db1']
        self.vW2 = beta * self.vW2 - self.lr * grads['dW2']
        self.vb2 = beta * self.vb2 - self.lr * grads['db2']
        # apply
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2
    
    def fit(self, X, Y, X_val=None, Y_val=None, epochs=100, verbose=False):
        history = {'loss': [], 'val_acc': []}
        for ep in range(1, epochs+1):
            cache = self._forward(X)
            loss = self._loss(cache['A2'], Y)
            grads = self._backward(cache, Y)
            self._update(grads)
            history['loss'].append(loss)
            if X_val is not None and Y_val is not None:
                preds = self.predict(X_val)
                history['val_acc'].append(np.mean(preds == np.argmax(Y_val, axis=1)))
            else:
                history['val_acc'].append(None)
            if verbose and (ep == 1 or ep % 100 == 0):
                if X_val is not None:
                    print(f"Epoch {ep}/{epochs} loss={loss:.4f} val_acc={history['val_acc'][-1]:.4f}")
                else:
                    print(f"Epoch {ep}/{epochs} loss={loss:.4f}")
        return history
    
    def predict_proba(self, X):
        return self._forward(X)['A2']
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# ---------- Hyperparameter sweep ----------
hidden_sizes = [8, 16, 32, 64]
learning_rates = [0.1, 0.01, 0.001]
results = []
start = time.time()
for h in hidden_sizes:
    for lr in learning_rates:
        nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=h, output_size=Y_train.shape[1],
                           learning_rate=lr, activation='relu', momentum=0.9, seed=42)
        hist = nn.fit(X_train, Y_train, X_val=X_test, Y_val=Y_test, epochs=100, verbose=False)
        preds = nn.predict(X_test)
        acc = np.mean(preds == y_test_labels)
        results.append({'hidden_size': h, 'learning_rate': lr, 'test_accuracy': acc})
        print(f"hidden={h:2d}, lr={lr:.3f} -> test_acc={acc:.4f}")
end = time.time()
print("Sweep time: {:.2f}s".format(end-start))

df = pd.DataFrame(results).pivot(index='hidden_size', columns='learning_rate', values='test_accuracy')
print("\nHyperparameter table:\n", df)

# ---------- Retrain best config ----------
best = df.stack().idxmax()  # (hidden_size, lr)
best_h, best_lr = best[0], best[1]
print(f"\nBest config: hidden={best_h}, lr={best_lr}")

best_nn = NeuralNetwork(input_size=X_train.shape[1], hidden_size=best_h, output_size=Y_train.shape[1],
                        learning_rate=best_lr, activation='relu', momentum=0.9, seed=1)
history_long = best_nn.fit(X_train, Y_train, X_val=X_test, Y_val=Y_test, epochs=600, verbose=True)

# plots
plt.figure(figsize=(8,5))
plt.plot(range(1, 601), history_long['loss'])
plt.title("Training Loss vs Epochs (best model)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, 601), history_long['val_acc'])
plt.title("Test Accuracy vs Epochs (best model)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()

final_preds = best_nn.predict(X_test)
final_acc = np.mean(final_preds == y_test_labels)
print(f"Final test accuracy after 600 epochs: {final_acc:.4f}")
