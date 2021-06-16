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


class MemLeakDetectionAlgorithm:
    def __init__(self, min_window=60, resampling_time_resolution='5min',  smoothing_window='1h'):
        """Constructor takes the parameters of the algorithm.py as arguments"""
        self.min_window = min_window
        self.best_period = np.inf
        self.best_window = np.inf
        self.best_r2 = -np.inf
        self.req_trend_r2 = 0.8
        self.resampling_time_resolution = resampling_time_resolution
        self.smoothing_window = smoothing_window

    def pre_processing(self, df):
        df = pd.DataFrame(df["Value"])
        df = df.resample(self.resampling_time_resolution).median()
        df.dropna(inplace=True)
        # Smooth timeseries
        ts = df.rolling(self.smoothing_window, min_periods=1).median()
        return df

    def do_linear_regression(self, x, degree):
        df = x.reshape(-1, 1)
        df_X = np.reshape(range(0, len(x)), (-1, 1))
        df_Y = df
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.1, random_state=42)

        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg = linear_model.LinearRegression(normalize=True)
        regr = Pipeline([
            ("poly_features", polybig_features),
            ("std_scaler", std_scaler),
            ("lin_reg", lin_reg),
        ])
        regr.fit(X_train, y_train)

        # Make predictions using the testing set
        y_pred = regr.predict(df_X)
        rms = sqrt(mean_squared_error(df_Y, y_pred))
        r2 = r2_score(df_Y, y_pred)
        return regr, df_X, rms, r2

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
        inc_window = 50
        dataset.columns = ['value']
        window = self.min_window
        max_window = len(dataset.value.values)
        if self.min_window < 0.2 * len(dataset.value.values):
            self.min_window = int(0.2 * len(dataset.value.values))

        while window <= max_window:
            temp_dataset = dataset.copy()
            temp_dataset = temp_dataset.tail(window)
            s = temp_dataset.value.values
            t = np.linspace(0, 1, len(s))
            fft = np.fft.fft(s)
            T = t[1] - t[0]  # sampling interval
            N = s.size
            # 1/T = frequency
            f = np.linspace(0, 1 / T, N)
            kk = pd.DataFrame(index=f[:N // 2], data=np.abs(fft)[:N // 2], columns=['value'])
            kk = kk.sort_values(by=['value'], ascending=False)
            probable_periods = kk.head(10).index
            probable_periods = [int(i) for i in probable_periods]
            for period in probable_periods:
                if period != 0:
                    result = seasonal_decompose(temp_dataset, model='multiplicative', freq=period)
                    trend = result.trend
                    trend.dropna(inplace=True)
                    r, df_X, rms, r2 = self.do_linear_regression(trend.values, 1)

                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        list(range(0, len(trend.value.values))),
                        trend.value.values)
                    if r2 >= self.best_r2 and slope > 0 and r2 > self.req_trend_r2:
                        self.best_r2 = r2
                        self.best_period = period
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
        if self.best_period != np.inf:
            temp_dataset = dataset.copy()
            temp_dataset = temp_dataset.tail(self.best_window)
            result = seasonal_decompose(temp_dataset, model='multiplicative', freq=self.best_period)
            trend = result.trend
            trend.dropna(inplace=True)
            r, df_X, rms, r2 = self.do_linear_regression(trend.values, 1)
            dataset['label'] = 0
            dataset['trend'] = 0
            dataset.label.iloc[range(trend.index[0], len(dataset.index))] = 1
            dataset.trend.iloc[range(trend.index[0], len(dataset.index))] = \
                r.predict(np.reshape(range(0, len(dataset.index) - trend.index[0]), (-1, 1))).reshape(1, len(
                    range(trend.index[0], len(dataset.index))))[0]
        else:
            dataset['label'] = 0
            dataset['trend'] = 0

        return dataset.loc[dataset['trend'] > 0]
