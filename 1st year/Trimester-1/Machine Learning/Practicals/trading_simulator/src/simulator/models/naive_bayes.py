from sklearn.naive_bayes import GaussianNB
from .base import BaseModel

class NaiveBayesModel(BaseModel):
    """Wrapper for Gaussian Naive Bayes."""
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
