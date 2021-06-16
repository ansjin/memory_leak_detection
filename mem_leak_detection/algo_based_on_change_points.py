import warnings  # `do not disturbe` mode
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings('ignore')


class MemLeakDetectionAlgorithmChangePoints:

    def __init__(self, min_slope=0, min_r2=0.8, max_change_points=10, min_window_size=60, delta_r2=0.05,
                 resampling_time_resolution='5min',  smoothing_window='1h',  seasonality=False):
        """Constructor takes the parameters of the algorithm.py as arguments"""
        self.min_slope = min_slope
        self.min_r2 = min_r2
        self.best_r2 = min_r2
        self.best_initial_index = 0
        self.max_change_points = max_change_points
        self.change_point_indexes = []
        self.resampling_time_resolution = resampling_time_resolution
        self.delta_r2 = delta_r2
        self.min_window_size = min_window_size
        self.seasonality = seasonality
        self.smoothing_window = smoothing_window

    def pre_processing(self, df):
        df = pd.DataFrame(df["Value"])
        df = df.resample(self.resampling_time_resolution).median()
        df.dropna(inplace=True)
        # Smooth timeseries
        # ts = df.rolling(self.smoothing_window, min_periods=1).median()
        return df

    @staticmethod
    def do_linear_regression(x):
        df = x.reshape(-1, 1)
        df_X = np.reshape(range(0, len(x)), (-1, 1))
        df_Y = df

        lin_reg = linear_model.LinearRegression(normalize=True)
        lin_reg.fit(df_X, df_Y)

        # Make predictions using the testing set
        y_pred = lin_reg.predict(df_X)
        r2 = r2_score(df_Y, y_pred)
        return lin_reg, r2

    def seasonality_identification(self, X):

        lin_reg, series_r2 = MemLeakDetectionAlgorithmChangePoints.do_linear_regression(X)
        best_r2 = series_r2
        probable_periods = [24]
        seasonality = False
        best_trend = 0

        for model_type in ['additive', 'multiplicative']:
            for period in probable_periods:
                if not (2*period > len(X)):
                    result = seasonal_decompose(pd.DataFrame(X, columns=['value']), model=model_type, freq=period)
                    trend = result.trend.copy()
                    trend.dropna(inplace=True)
                    r, trend_r2 = self.do_linear_regression(trend.values)
                    if trend_r2 > best_r2:
                        best_r2 = trend_r2
                        seasonality = True
                        best_trend = result.trend
        return seasonality, best_trend  #best_model_type, best_r2, best_period

    def get_change_point_indices(self, X, method='modified_z_score'):
        """
       get_change_point_indices

       This function find the change points indexes present in the given data

       :param X: The history data, as a pd.Series with timestamps as index
       :param method: The type of method from ['max_change_points', 'z_score', 'modified_z_score']
       :return:  The indexes of the change points.
       """
        # Step1 : Calculate the first order difference and make a dataframe of the values.
        diff_dataset = pd.DataFrame(X.diff())
        # Step 2 : Calculate the modulus of those values as we are interested in their absolute value
        diff_dataset[0] = np.absolute(diff_dataset.values)
        # Step 3 : Save the original indexes of the dataframe
        orig_indexes = diff_dataset.index
        # Step 4 : Reset the index of the dataframe
        diff_dataset.reset_index(inplace=True)

        # Step 5 (important step) : Determine the indexes of the change point
        diff_dataset = diff_dataset.dropna()
        ys = diff_dataset[0].values
        indexes = []
        if method == 'max_change_points':
            # Method 1 : Based on the default set maximum number of change points to be calculated
            indexes = diff_dataset.sort_values([0], ascending=0).head(self.max_change_points).index
        elif method == 'z_score':
            # Method 2 : Taking the change points based on the z-score
            # as z_score find the anomaly points in the given data so here the same method is applied on first order
            # difference of time series to get the anomaly points in the difference and those points are regarded as the
            # change points.
            threshold = 3
            mean_y = np.mean(ys)
            stdev_y = np.std(ys)
            z_scores = [(y - mean_y) / stdev_y for y in ys]
            diff_scores = pd.DataFrame(np.abs(z_scores))
            indexes = diff_scores[diff_scores[0] > threshold].index
        elif method == 'modified_z_score':
            # Method 3 : Taking the change points based on the modified_z_score
            # This method uses median instead of mean as in the above method, therefore it is more robust
            threshold = 3.5
            median_y = np.median(ys)
            median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
            modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                                 for y in ys]
            diff_scores = pd.DataFrame(np.abs(modified_z_scores))
            indexes = diff_scores[diff_scores[0] > threshold].index
        # TODO: Other methods to be added here

        # Step 6 : Check for if all the change points are greater than minimum window size from the last index.

        indexes = np.sort(indexes)
        final_change_point_indexes = list()
        for idx in indexes :
            if np.absolute(idx - len(orig_indexes)) > self.min_window_size :
                final_change_point_indexes.append(int(idx))

        # Step 7 : Append the last and beginning of the series
        final_change_point_indexes = np.append(final_change_point_indexes, int(len(orig_indexes) - 1))
        final_change_point_indexes = np.append(final_change_point_indexes, int(0))
        final_change_point_indexes = [int(i) for i in final_change_point_indexes]
        final_change_point_indexes = np.sort(final_change_point_indexes)

        # Step 8 : Get the original indexes
        final_change_point_indexes = orig_indexes[final_change_point_indexes]

        # Step 9 : Return the change point indexes
        return final_change_point_indexes

    def get_predicted_values(self, dataset):
        dataset = pd.DataFrame(dataset)
        predicted_values = pd.DataFrame(np.empty(len(dataset.index)))
        predicted_values.index = dataset.index
        predicted_values['trend'] = 0
        predicted_values['label'] = 0

        # The last point is the end point of series so instead we take the second last point to get an initial window
        # from second last to last
        idx = len(self.change_point_indexes) - 2
        final_index = dataset.index[len(dataset.index) - 1]

        # set the initial to final too, if there is no perfect line fit
        # then there would be no element in the series to predict values
        self.best_initial_index = final_index

        # Step 1 : Check for the best line fit starting from the back and hopping by the change point indexes

        while idx >= 0:
            val = self.change_point_indexes[idx]
            series = dataset[dataset.columns[0]].loc[val:final_index].values

            if len(series) > 1:
                lin_reg, r2 = MemLeakDetectionAlgorithmChangePoints.do_linear_regression(series)
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    list(range(0, len(series))), series)

                # Check if the line is best fit and also see if the line can be in between the adjusted delta_r2
                if (r2 >= self.best_r2 or r2 >= (self.best_r2 - self.delta_r2)) and slope > self.min_slope:
                    self.best_r2 = r2
                    self.best_initial_index = val

            idx = idx - 1

        # Step 2 : Get the series for which the line is best fit
        series = dataset[dataset.columns[0]].loc[self.best_initial_index:final_index].values

        # Step 3 : Check if the series has more than 1 element
        if len(series) > 1:
            # Step 4 : Do the linear regression on the series and get the predicted values
            lin_reg, r2 = MemLeakDetectionAlgorithmChangePoints.do_linear_regression(series)
            regr_series_predicted_values = lin_reg.predict(np.array(range(0, len(series))).reshape(-1, 1))
            predicted_values['trend'][self.best_initial_index:final_index] = \
            regr_series_predicted_values.reshape(1, regr_series_predicted_values.shape[0])[0]
            predicted_values['label'][self.best_initial_index:final_index] = 1

        # Step 5 : Return the predicted values
        return predicted_values

    def fit(self, X, y=None):
        """
        Fit the algorithm.py using historic data

        This function optionally fine-tunes the algorithm.py for this particular time series using it's historic data.

        :param X: The history data, as a pd.Series with timestamps as index
        :param y: Optional labels, currently not used
        :return:  The algorithm.py. We do not distinguish between algorithm.py and fitted model. Everything the algorithm.py
                  learned should be stored inside the class.
        """

        if len(X) == 0:
            # log.warning('Empty timeseries provided for training. Skipping.')
            return self

        # print(len(X))
        X = self.pre_processing(X)
        X_copy = X.copy()
        if self.seasonality:
            seasonality, trend = self.seasonality_identification(X_copy.values)
            print(seasonality)
            if seasonality:
                X_copy.values[0:len(trend.index)] = trend.value.values

        change_point_indexes = self.get_change_point_indices(X_copy)
        self.change_point_indexes = change_point_indexes
        return self

    def predict(self, X):
        """
        Detect anomalies in a time series

        This function is used for the anomaly detection

        :param X: The time series, as a pd.Series with timestamps as index
        :return: A list with the same length as the time series, with 0's marking normal values and 1's marking
                 anomalies
        """
        X = self.pre_processing(X)
        dataset = pd.DataFrame(X.values)
        predicted_df = self.get_predicted_values(X)
        dataset['trend'] = predicted_df['trend'].values
        dataset['label'] = predicted_df['label'].values

        return dataset.loc[dataset['trend'] > 0]
