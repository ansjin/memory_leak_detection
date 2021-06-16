import warnings  # `do not disturbe` mode
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from math import sqrt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')


class MemLeakDetectionAlgorithmPolyFit:
    def __init__(self, min_window=60, resampling_time_resolution='5min',  smoothing_window='1h'):
        """Constructor takes the parameters of the algorithm.py as arguments"""
        self.min_window = min_window
        self.best_window = np.inf
        self.best_rmse = 0.1
        self.resampling_time_resolution = resampling_time_resolution
        self.smoothing_window = smoothing_window

    def pre_processing(self, df):
        df = pd.DataFrame(df["Value"])
        df = df.resample(self.resampling_time_resolution).median()
        df.dropna(inplace=True)
        # Smooth timeseries
        ts = df.rolling(self.smoothing_window, min_periods=1).median()
        return df

    def fit(self, X, y=None):
        """
        Fit the algorithm.py using historic data

        This function optionally fine-tunes the algorithm.py for this particular time series using it's historic data.

        :param X: The history data, as a pd.Series with timestamps as index
        :param y: Optional labels, currently not used
        :return:  The algorithm.py. We do not distinguish between algorithm.py and fitted model. Everything the algorithm.py
                  learned should be stored inside the class.
        """
        # dataset = self.pre_processing(X)
        dataset = pd.DataFrame(X.Value.values)
        inc_window = 1
        dataset.columns = ['value']
        window = self.min_window
        max_window = len(dataset.value.values)
        # if self.min_window < 0.2 * len(dataset.value.values):
        #    self.min_window = int(0.2 * len(dataset.value.values))

        while window <= max_window:
            temp_dataset = dataset.copy()
            temp_dataset = temp_dataset.tail(window)
            s = temp_dataset.value.values

            coefficients, residuals, _, _, _ = np.polyfit(range(len(temp_dataset.index)), temp_dataset.value,
                                                          1, full=True)
            mse = residuals[0] / (len(temp_dataset.index))
            nrmse = np.sqrt(mse) / (temp_dataset.value.max() - temp_dataset.value.min())

            if nrmse <= self.best_rmse and coefficients[0] > 0:
                self.best_rmse = nrmse
                self.best_window = window
            window = window + inc_window
        return self

    def predict(self, X):
        """
        Detect anomalies in a time series

        This function is used for the anomaly detection

        :param X: The time series, as a pd.Series with timestamps as index
        :return: A list with the same length as the time series, with 0's marking normal values and 1's marking
                 anomalies
        """
        # dataset = self.pre_processing(X)
        dataset = pd.DataFrame(X.Value.values)

        if self.best_window != np.inf:
            temp_dataset = dataset.copy()
            temp_dataset = temp_dataset.tail(self.best_window)
            temp_dataset.columns = ['value']
            coefficients, residuals, _, _, _ = np.polyfit(range(len(temp_dataset.index)), temp_dataset.value,
                                                          1, full=True)
            mse = residuals[0] / (len(temp_dataset.index))
            nrmse = np.sqrt(mse) / (temp_dataset.value.max() - temp_dataset.value.min())
            dataset['label'] = 0
            dataset['trend'] = 0
            dataset.label.iloc[temp_dataset.index] = 1
            dataset.trend.iloc[temp_dataset.index] = [coefficients[0]*x + coefficients[1] for x in range(len(temp_dataset.index))]
        else:
            dataset['label'] = 0
            dataset['trend'] = 0

        return dataset.loc[dataset['trend'] > 0]
