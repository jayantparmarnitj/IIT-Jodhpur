from sklearn.neighbors import KNeighborsClassifier
from .base import BaseModel

class KNNModel(BaseModel):
    """Wrapper for k-Nearest Neighbors classifier."""
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
