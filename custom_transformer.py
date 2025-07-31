from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class CountEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Ensure X is treated as a Series for value_counts()
        if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            X_series = X.iloc[:, 0]
        else:
            X_series = X # Assume X is already a Series if not a single-column DataFrame

        self.freqs_ = X_series.value_counts().to_dict()
        return self

    def transform(self, X):
        if self.freqs_ is None:
            raise RuntimeError("CountEncoder has not been fitted yet.")

        # Ensure X is treated as a Series for mapping
        if isinstance(X, pd.DataFrame) and X.shape[1] == 1:
            X_series = X.iloc[:, 0]
        else:
            X_series = X # Assume X is already a Series if not a single-column DataFrame

        # Use .map() on the Series to apply the frequency dictionary.
        # .fillna(0) will handle values in X_series that were not present during fit.
        transformed_data = X_series.map(self.freqs_).fillna(0)

        return transformed_data.values.reshape(-1, 1)