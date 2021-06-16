import functools
import logging
import typing
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error

import datetime
from statsmodels.tsa.seasonal import seasonal_decompose


#from iforesight2.algorithms.daily_seasonality_detect.seasonality_detection import check_daily_seasonality
#from iforesight2.algorithms.profiles import TimeSeriesProfiles
from .base import  make_uniform_result_group, AnomalousGroup, AnomalousPoint
#from iforesight2.algorithms.exceptions import InvalidDataProperties
#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class InvalidDataProperties(Exception):

    # Constructor or Initializer
    def __init__(self, value, string_val):
        self.value = value
        self.string_val = string_val

        # __str__ is to print() the value

    def __str__(self):
        return repr(self.value + ' ' + self.string_val)


class Precog():
    """
    Precog - Detect if a metric will exceed a limit in the near future due to a trend

    Precog is a trend detection algorithm for positives trends. It detects linear trends only, and calculates if the
    trend will increase to a given limit within a certain 'critical' time. Precog was primarily designed for detecting
    memory leaks from memory usage data.

    Parameters
    ----------
    - 'req_trend_duration': Minimal required duration for a trend to be recognised. In time units.
    - 'req_trend_r2': Minimal closeness to a straight line for a trend. Between 0 and 1.
    - 'limit': The upper limit for the metric.
    - 'min_value': The minimal value the time series must have to detect a problem.
    - 'critical_time': A trend is reported as critical only if it is predicted to reach the limit before this time.
                       In time units.
    - 'smoothing_window': The strength of the smoothing applied to the metric before trend detection. In time units.

    Change Log
    ----------
    - Version 0.3
        + Detect seasonality in training data and disable trend detection when present
        + Add training phase which detects normal trend durations and slopes, and does not report similar trends during
          prediction.
        + Add 'min_value' parameter to turn off detection when metric is low
        + Also specify the smoothing_window parameter as time
    - Version 0.2
        + Use a timestamps to specify windows and times instead of points
        + Remove max_window parameter & always use the complete training time series
        + Always use a fixed number of windows per time series
    - Version 0.1
        + Initial version of the algorithm
    """
    name = 'precog'
    version = 0.3
    default_config = {
        'req_trend_duration': '6h',
        'req_trend_r2': 0.8,
        'limit': 100.0,
        'min_value': 40.0,
        'critical_time': '7d',
        'smoothing_window': '1h',
        'max_change_points': 10,
        'seasonality_detection': False,
        'req_trend_r2_delta': 0.05,
        'resampling_time_resolution': '5min',
        'min_time_seasonality_detection': '3d',
        'min_separation_duration': '6h',
        'top_max_n_values': 10,
        'use_max_values_as_filter': True
    }

    requires_timestamps = True

    def __init__(self,
                 req_trend_duration=default_config['req_trend_duration'],
                 req_trend_r2=default_config['req_trend_r2'],
                 limit=default_config['limit'],
                 min_value=default_config['min_value'],
                 critical_time=default_config['critical_time'],
                 smoothing_window=default_config['smoothing_window'],
                 max_change_points=default_config['max_change_points'],
                 seasonality_detection=default_config['seasonality_detection'],
                 req_trend_r2_delta=default_config['req_trend_r2_delta'],
                 resampling_time_resolution=default_config['resampling_time_resolution'],
                 min_time_seasonality_detection=default_config['min_time_seasonality_detection'],
                 min_separation_duration=default_config['min_separation_duration'],
                 top_max_n_values=default_config['top_max_n_values'],
                 use_max_values_as_filter=default_config['use_max_values_as_filter'],
                 ):

        self.req_trend_duration_str = req_trend_duration
        self.critical_time_str = critical_time

        self.n_windows = 100
        self.req_trend_r2 = req_trend_r2
        self.limit = limit
        self.min_value = min_value

        self.smoothing_window = smoothing_window
        self.seasonality_detected = False

        # Change points Related
        self.max_change_points = max_change_points
        self.change_point_indexes = []
        self.change_point_indexes_predict = []
        self.seasonality_detection = seasonality_detection
        self.req_trend_r2_delta = req_trend_r2_delta
        self.resampling_time_resolution = resampling_time_resolution
        self.min_time_seasonality_detection = min_time_seasonality_detection
        self.min_separation_duration = min_separation_duration

        # Maximum Values Filter Related
        self.top_max_n_values = top_max_n_values
        self.use_max_values_as_filter = use_max_values_as_filter
        self.max_n_values = []

        # Values learned turing training
        self.max_training_duration_str = req_trend_duration
        self.max_training_slope = 0
        self.found_trends = []


        # For visualization
        self.fit_trends = []
        self.prediction_trend = None

    #@classmethod
    #def is_applicable(cls, profiles: typing.Set[TimeSeriesProfiles]) -> bool:
     #   if TimeSeriesProfiles.is_seasonal in profiles:
     #       return False

     #   if TimeSeriesProfiles.has_negatives in profiles:
     #       return False
     #
      #  return True

    @property
    def min_separation_window(self) -> pd.Timedelta:
        return self._conv_to_timedelta(self.min_separation_duration)

    @property
    def min_window(self) -> pd.Timedelta:
        return self._conv_to_timedelta(self.req_trend_duration_str)

    @property
    def critical_time(self) -> pd.Timedelta:
        return self._conv_to_timedelta(self.critical_time_str)

    @property
    def max_training_duration(self) -> pd.Timedelta:
        return self._conv_to_timedelta(self.max_training_duration_str)

    @functools.lru_cache()
    def _conv_to_timedelta(self, t: str) -> pd.Timedelta:
        return pd.Timedelta(t)

    def pre_process(self, ts: pd.Series) -> pd.Series:
        # Reduce number of points to one every five minutes
        ts = ts.resample(self.resampling_time_resolution).median().interpolate()

        # Smooth timeseries
        ts = ts.rolling(self.smoothing_window, min_periods=1).median()
        return ts

    @staticmethod
    def do_linear_regression(x: pd.Series):
        # Use seconds as x-axis
        train_predictors = ((x.index.max() - x.index) // pd.Timedelta('1min')).values.astype(np.float) * -1
        # reshape(-1, 1) converts an (n)-array into an (nx1)-array
        train_predictors = train_predictors.reshape(-1, 1)
        train_response = x.values.astype(np.float).reshape(-1, 1)

        lin_reg = linear_model.LinearRegression()
        lin_reg.fit(train_predictors, train_response)

        slope = lin_reg.coef_[0]
        train_prediction = lin_reg.predict(train_predictors)
        r2 = r2_score(train_response, train_prediction)

        #estimators = [('RANSAC', RANSACRegressor(random_state=42)), ]


        return r2, slope, train_prediction.reshape(-1)

    def check_seasonality(self, x, probable_periods):

        series_r2, slope, reg_line = self.do_linear_regression(x)
        best_r2 = series_r2
        seasonality = False
        best_trend = 0

        for model_type in ['additive', 'multiplicative']:
            for period in probable_periods:
                if not (2*period > len(x.index)):
                    result = seasonal_decompose(pd.DataFrame(x.values, columns=['value']), model=model_type, freq=period)
                    trend = result.trend.copy()
                    trend.dropna(inplace=True)
                    trend_r2, trend_slope, reg_trend_line = self.do_linear_regression(trend)
                    if trend_r2 > best_r2:
                        best_r2 = trend_r2
                        seasonality = True
                        best_trend = result.trend

        return seasonality, best_trend

    def calc_exit_time(self, current_value, slope) -> pd.Timedelta:
        if slope <= 0:
            return pd.Timedelta.max
        return pd.Timedelta(int((self.limit - current_value) / slope * 60), 's')

    def check_input(self, ts: pd.Series):
        if not isinstance(ts, pd.Series):
            raise InvalidDataProperties(f"{self.name} requires a timeseries with timestamps")

        if not isinstance(ts.index, pd.DatetimeIndex):
            raise TypeError('Input must be indexed by DatetimeIndex.')

        if len(ts) == 0:
            raise InvalidDataProperties(self.name, f"Timeseries does not contain any points!")

        if ts.max() > self.limit:
            raise InvalidDataProperties(self.name, f"Values are higher than the limit ({self.limit})")

    def get_change_point_indices(self, x: pd.Series, method='modified_z_score'):
        """
       get_change_point_indices

       This function find the change points indexes present in the given data

       :param x: The history data, as a pd.Series with timestamps as index
       :param method: The type of method from ['max_change_points', 'z_score', 'modified_z_score']
       :return:  The indexes of the change points.
       """
        # Step1 : Calculate the first order difference and make a dataframe of the values.
        diff_dataset = pd.DataFrame(x.diff())
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

        indexes = np.array(indexes).tolist()

        for idx in indexes:
                final_change_point_indexes.append(int(idx))
        # Step 7 : Append the last and beginning of the series
        final_change_point_indexes = np.append(final_change_point_indexes, int(len(orig_indexes) - 1))
        final_change_point_indexes = np.append(final_change_point_indexes, int(0))
        final_change_point_indexes = [int(i) for i in final_change_point_indexes]
        final_change_point_indexes = np.sort(final_change_point_indexes)

        # Step 8 : Get the original indexes
        final_change_point_indexes = orig_indexes[final_change_point_indexes]

        final_change_point_indexes = np.array(final_change_point_indexes).tolist()
        idx = 0
        while True:
            length = len(final_change_point_indexes) - 1
            if idx < length:
                ts_duration = pd.to_timedelta(final_change_point_indexes[idx + 1] - final_change_point_indexes[idx])
                if ts_duration.total_seconds() < self.min_separation_window.total_seconds():
                    del final_change_point_indexes[idx + 1]
                    idx = idx
                else:
                    idx = idx + 1
            else:
                break

        final_change_point_indexes = [pd.to_datetime(x) for x in final_change_point_indexes]
        final_change_point_indexes.append(x.index[-1])
        # Step 9 : Return the change point indexes
        return final_change_point_indexes

    def detect_trend(self, ts: pd.Series):
        current_value = ts.tail(1).values[0]
        self.prediction_trend = None

        ts_duration = ts.index[-1] - ts.index[0]
        if ts_duration < self.min_window:
            return None

        window = ts
        r2, slope, reg_line = self.do_linear_regression(window)
        log.debug('Checking linear fit: (r2: %0.3f, slope: %0.3f)', r2, slope)

        if r2 > self.req_trend_r2:
            self.prediction_trend = pd.Series(reg_line, index=window.index)
            exit_time = self.calc_exit_time(current_value, slope)
            return {'start': window.index[0], 'stop': window.index[-1],
                    'duration': window.index[-1] - window.index[0],
                    'exit_time': exit_time, 'slope': slope, 'r2': r2,
                    'prediction_trend': pd.Series(reg_line, index=window.index)}

        return None

    def fit(self, X: pd.Series):
        self.fit_trends = []

        if len(X) == 0:
            log.warning('Empty timeseries provided for training. Skipping.')
            return self
        self.check_input(X)
        X = self.pre_process(X)
        X_copy = X.copy()

        # Check for use_max_values_as_filter if enabled
        if self.use_max_values_as_filter:
            log.debug('Using maximum values as filter is Enabled')
            self.max_n_values =  X_copy.values[np.argsort(X_copy.values)[-self.top_max_n_values:]]
            self.max_n_values = [np.median(self.max_n_values)]


        ts_duration = X.index[-1] - X.index[0]
        log.info('Training with data for %s (%i points)', ts_duration, len(X))

        # Check for seasonality if enabled
        if self.seasonality_detection:
            log.debug('Seasonality detection is Enabled')
            if ts_duration > pd.Timedelta(self.min_time_seasonality_detection):
                seasonality, series_without_seasonality = self.check_seasonality(X_copy, probable_periods=[24])
                if seasonality:
                    X_copy.values[0:len(series_without_seasonality.index)] = series_without_seasonality.value.values
                    log.info('Daily seasonality detected in training data, so removing seasonality and copying rest')
                    self.seasonality_detected = True
                else:
                    log.debug('No daily seasonality detected in training data')
            else:
                log.debug('No seasonality is checked in training data due to less data')
        else:
            log.debug('Seasonality detection is turned off')

        # get change points indexes
        change_point_indexes = self.get_change_point_indices(X_copy)
        self.change_point_indexes = change_point_indexes

        itr_idx = 0

        # Step 1 : Check for the best line fit starting from the back and hopping by the change point indexes
        # The last point is the end point of series so instead we take the second last point to get an initial window
        # from second last to last

        while itr_idx <= len(self.change_point_indexes) - 2:
            current_idx = self.change_point_indexes[itr_idx]
            next_itr_idx = itr_idx + 1
            best_local_trend = None
            log.debug('Searching for the local best trend if it exist from change point')


            # Search for every trend starting from the current point to all the next change points
            while next_itr_idx <= len(self.change_point_indexes) - 1:
                next_idx = self.change_point_indexes[next_itr_idx]
                window_series = X_copy[current_idx:next_idx]

                log.debug('Searching for the local best trend in series of (%i points) ', len(window_series))
                if len(window_series.index) > 1:
                   # print(window_series)
                    trend = self.detect_trend(window_series)
                    if trend is not None and best_local_trend is None and trend['slope'] > 0:
                        best_local_trend = trend
                        log.debug('Found local best trend with error score (r2: %0.3f)', best_local_trend['r2'])
                    elif trend is not None and best_local_trend is not None and \
                            trend['r2'] >= best_local_trend['r2'] and \
                            trend['duration'] >= best_local_trend['duration'] and \
                            trend['slope'] >= best_local_trend['slope'] :
                        best_local_trend = trend
                        log.debug('Found local best trend with error score (r2: %0.3f)', best_local_trend['r2'])
                next_itr_idx = next_itr_idx + 1

            if best_local_trend is not None:
                # We only care for critical trends which are not ongoing at the end of the training data
                if best_local_trend['exit_time'] <= self.critical_time and best_local_trend['stop'] < X_copy.index[-1]:
                    log.debug('Found critical trend with duration %s and slope %0.3f in training data.',
                              best_local_trend['duration'], best_local_trend['slope'])

                    if best_local_trend['duration'] > self.max_training_duration:
                        self.max_training_duration_str = str(best_local_trend['duration'])

                    if best_local_trend['slope'] > self.max_training_slope:
                        self.max_training_slope = best_local_trend['slope']

                    self.fit_trends.append(best_local_trend['prediction_trend'] )
                    # Keep all the found trend information
                    self.found_trends.append(best_local_trend)
                    log.info('Added New Trend')
                    log.info('New acceptable duration and slope for trend are %s and %0.3f',
                             self.max_training_duration, self.max_training_slope)
            itr_idx = itr_idx + 1
        return self

    def predict(self, X: pd.Series):
        self.check_input(X)

        if X.tail(1).values[0] < self.min_value:
            log.info('Not predicting anomalies because the metric is too low')
            return []

        timeseries_preprocessed = self.pre_process(X)
        current_timestamp = timeseries_preprocessed.index[-1]
        ts_duration = current_timestamp - timeseries_preprocessed.index[0]

        self.change_point_indexes_predict = self.get_change_point_indices(timeseries_preprocessed.copy())

        idx = len(self.change_point_indexes_predict) - 2
        final_index = timeseries_preprocessed.index[len(timeseries_preprocessed.index) - 1]

        # Step 1 : Check for the best line fit starting from the back and hopping by the change point indexes

        best_trend = {'start': 0, 'stop': 0, 'duration': 0,
                      'exit_time': 0, 'slope': 0, 'r2': 0, 'prediction_trend': pd.Series([])}

        best_series = pd.Series([])
        found = False
        while idx >= 0:
            val = self.change_point_indexes_predict[idx]
            series = timeseries_preprocessed[val:final_index]

            if len(series.index) > 1:
                log.info('Predicting anomalies with data for %s (%i points)', ts_duration, len(series))
                trend = self.detect_trend(series)

                if trend is None:
                    log.info('No linear window found')
                    idx = idx - 1
                    continue

                if trend['exit_time'] > self.critical_time:
                    log.info('Found trend, but the exit time is too large (%s > %s, slope: %0.3f)',
                             trend['exit_time'], self.critical_time, trend['slope'])
                    idx = idx - 1
                    continue

                if trend['slope'] >= self.max_training_slope and trend['duration'] >= self.max_training_duration:
                    log.info('Found trend, and it is more severe than the maximum in the training data '
                          '(duration %s, slope %0.3f)', trend['duration'], trend['slope'])
                    best_trend = trend
                    best_series = series
                    r2, slope, reg_line = self.do_linear_regression(best_series)
                    self.prediction_trend = pd.Series(reg_line, index=best_series.index)
                    found = True
                    idx = idx - 1
                    continue

                for training_trend in self.found_trends:
                    if trend['slope'] >= training_trend['slope'] and trend['duration'] >= training_trend['duration']:
                        log.info('Found trend, and it is more severe than the one of the trend in the training data '
                                 '(duration %s, slope %0.3f)', trend['duration'], trend['slope'])

                        if trend['slope'] >= best_trend['slope'] and trend['duration'] >= self._conv_to_timedelta(
                                best_trend['duration']):
                            log.info('Found trend, and it is more severe than the ones found in the test data '
                                     '(duration %s, slope %0.3f)', trend['duration'], trend['slope'])
                            best_trend = trend
                            best_series = series
                            r2, slope, reg_line = self.do_linear_regression(best_series)
                            self.prediction_trend = pd.Series(reg_line, index=best_series.index)
                            found = True
                    else:
                        log.info('Found trend, but not greater than the training trend so ignoring it')

            idx = idx - 1
        if not found:
            return []

        # Check for use_max_values_as_filter if enabled
        if self.use_max_values_as_filter:
            log.debug('Using maximum values as filter is Enabled')
            idx = len(self.change_point_indexes_predict) - 2
            val = self.change_point_indexes_predict[idx]
            last_window_series = timeseries_preprocessed[val:final_index]

            last_window__max_n_values = last_window_series.values[np.argsort(last_window_series.values)[-self.top_max_n_values:]]


            for max_value in last_window__max_n_values:
                if max_value < 95 and max_value < self.max_n_values[0] : #- 0.05*self.max_n_values[0]:
                    log.debug('Last window maximum value is less than the one found in the training data')
                    log.info('( Maximum Value in Training %0.3f, Maximum Value in the test data %0.3f)', self.max_n_values[0], max_value)
                    return []


        log.info('Found critical trend with exit time in %s (duration %s, slope %0.3f)',
                 best_trend['exit_time'], best_trend['duration'], best_trend['slope'])
        labels = np.zeros(len(best_series))
        labels[(best_trend['start'] <= best_series.index) & (best_series.index <= best_trend['stop'])] = 1

        indexed_anomalies = make_uniform_result_group(
            best_series, labels, 'may exceed limit soon (about ' + str(best_trend['exit_time']) + ' left)',
            'may exceed limit soon', 10)

        # map anomalies back to original timestamps

        anomalies = [AnomalousGroup([AnomalousPoint(best_series.index[ap.timestamp].value // 10 ** 6, ap.value, ap.label)
                                     for ap in ag.points], ag.label, ag.type)
                     for ag in indexed_anomalies]
        return anomalies

    def get_regression_line(self) -> Optional[pd.Series]:
        return self.prediction_trend

    def get_model(self):
        model = super().get_model()
        model['req_trend_duration'] = self.req_trend_duration_str
        model['req_trend_r2'] = self.req_trend_r2
        model['limit'] = self.limit
        model['critical_time'] = self.critical_time_str
        model['smoothing_window'] = self.smoothing_window
        model['seasonality_detected'] = self.seasonality_detected
        model['min_value'] = self.min_value
        model['max_training_duration'] = self.max_training_duration_str
        model['max_training_slope'] = self.max_training_slope
        return model

    def update_from_model(self, model):
        super().update_from_model(model)
        self.req_trend_duration_str = model['req_trend_duration']
        self.req_trend_r2 = model['req_trend_r2']
        self.limit = model['limit']
        self.critical_time_str = model['critical_time']
        self.smoothing_window = model['smoothing_window']
        self.seasonality_detected = model['seasonality_detected']
        self.min_value = model['min_value']
        self.max_training_duration_str = model['max_training_duration']
        self.max_training_slope = model['max_training_slope']

    @classmethod
    def add_synthetic_anomaly(cls, ts):
        raise NotImplementedError()

    @classmethod
    def add_normal_modification(cls, ts):
        raise NotImplementedError()
