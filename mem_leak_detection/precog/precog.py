import functools
import logging
import typing
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import r2_score

from iforesight2.algorithms.daily_seasonality_detect.seasonality_detection import check_daily_seasonality
from iforesight2.algorithms.profiles import TimeSeriesProfiles
from iforesight2.algorithms.base import TimeseriesAlgorithm, make_uniform_result_group, require_single_timeseries, \
    AnomalousGroup, AnomalousPoint
from iforesight2.algorithms.exceptions import InvalidDataProperties

log = logging.getLogger(__name__)


class Precog(TimeseriesAlgorithm):
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
        'req_trend_r2': 0.88,
        'limit': 100.0,
        'min_value': 40.0,
        'critical_time': '7d',
        'smoothing_window': '1h'
    }

    requires_timestamps = True

    def __init__(self,
                 req_trend_duration=default_config['req_trend_duration'],
                 req_trend_r2=default_config['req_trend_r2'],
                 limit=default_config['limit'],
                 min_value=default_config['min_value'],
                 critical_time=default_config['critical_time'],
                 smoothing_window=default_config['smoothing_window']):

        self.req_trend_duration_str = req_trend_duration
        self.critical_time_str = critical_time

        self.n_windows = 100
        self.req_trend_r2 = req_trend_r2
        self.limit = limit
        self.min_value = min_value

        self.smoothing_window = smoothing_window
        self.seasonality_detected = False

        # Values learned turing training
        self.max_training_duration_str = req_trend_duration
        self.max_training_slope = 0

        # For visualization
        self.fit_trends = []
        self.prediction_trend = None

    @classmethod
    def is_applicable(cls, profiles: typing.Set[TimeSeriesProfiles]) -> bool:
        if TimeSeriesProfiles.is_seasonal in profiles:
            return False

        if TimeSeriesProfiles.has_negatives in profiles:
            return False

        return True

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
        ts = ts.resample('5min').median().interpolate()

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
        return r2, slope, train_prediction.reshape(-1)

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

    @require_single_timeseries
    def fit(self, X: pd.Series):
        self.fit_trends = []
        if len(X) == 0:
            log.warning('Empty timeseries provided for training. Skipping.')
            return self
        self.check_input(X)
        X = self.pre_process(X)

        ts_duration = X.index[-1] - X.index[0]
        log.info('Training with data for %s (%i points)', ts_duration, len(X))

        # Check for seasonality
        if ts_duration > pd.Timedelta('3d') and check_daily_seasonality(X, output='bool', resample='5min') is True:
            log.info('Daily seasonality detected in training data')
            self.seasonality_detected = True
            return self
        else:
            log.debug('No seasonality detected in training data')

        # Detect critical trends in the remaining training data. If some are found, increase min_window to the size
        # of the trend.
        max_trend_duration = ts_duration
        max_steps = max((ts_duration - self.min_window) // pd.Timedelta('2h'), 1)
        for window_size_sec in np.linspace(start=max_trend_duration.total_seconds(),
                                           stop=self.min_window.total_seconds(),
                                           num=min(self.n_windows, max_steps)):
            if window_size_sec > max_trend_duration.total_seconds():
                continue
            window_duration = pd.Timedelta(int(window_size_sec), 's')
            log.debug('Cutting training timeseries after %s', window_duration)

            window = X.copy()[X.index <= X.index[0] + window_duration]

            trend = self.detect_trend(window)

            if trend is not None:
                # We only care for critical trends which are not ongoing at the end of the training data
                if trend['exit_time'] <= self.critical_time and trend['stop'] < X.index[-1]:
                    log.debug('Found critical trend with duration %s and slope %0.3f in training data.',
                              trend['duration'], trend['slope'])

                    if trend['duration'] > self.max_training_duration:
                        self.max_training_duration_str = str(trend['duration'])

                    if trend['slope'] > self.max_training_slope:
                        self.max_training_slope = trend['slope']

                    self.fit_trends.append(self.prediction_trend)

                max_trend_duration = trend['start'] - X.index[0]

        log.info('New acceptable duration and slope for trends are %s and %0.3f',
                 self.max_training_duration, self.max_training_slope)
        return self

    def predict(self, X: pd.Series):
        self.check_input(X)

        if self.seasonality_detected is True:
            log.info('Not predicting anomalies because the metric has seasonality')
            return []

        if X.tail(1).values[0] < self.min_value:
            log.info('Not predicting anomalies because the metric is too low')
            return []

        timeseries_preprocessed = self.pre_process(X)
        current_timestamp = timeseries_preprocessed.index[-1]
        ts_duration = current_timestamp - timeseries_preprocessed.index[0]

        log.info('Predicting anomalies with data for %s (%i points)', ts_duration, len(X))
        trend = self.detect_trend(X)

        if trend is None:
            log.info('No linear window found')
            return []

        if trend['exit_time'] > self.critical_time:
            log.info('Found trend, but the exit time is too large (%s > %s, slope: %0.3f)',
                     trend['exit_time'], self.critical_time, trend['slope'])
            return []

        if trend['slope'] <= self.max_training_slope and trend['duration'] <= self.max_training_duration:
            log.info('Found trend, but it is less severe than the ones in the training data (duration %s, slope %0.3f)',
                     trend['duration'], trend['slope'])
            return []

        log.info('Found critical trend with exit time in %s (duration %s, slope %0.3f)',
                 trend['exit_time'], trend['duration'], trend['slope'])
        labels = np.zeros(len(X))
        labels[(trend['start'] <= X.index) & (X.index <= trend['stop'])] = 1
        indexed_anomalies = make_uniform_result_group(
            X, labels, 'may exceed limit soon (about ' + str(trend['exit_time']) + ' left)',
            'may exceed limit soon', 10)
        # map anomalies back to original timestamps
        anomalies = [AnomalousGroup([AnomalousPoint(X.index[ap.timestamp].value // 10 ** 6, ap.value, ap.label)
                                     for ap in ag.points], ag.label, ag.type)
                     for ag in indexed_anomalies]
        return anomalies

    def detect_trend(self, ts: pd.Series):
        self.prediction_trend = None
        current_value = ts.tail(1).values[0]

        ts_duration = ts.index[-1] - ts.index[0]
        if ts_duration < self.min_window:
            return None

        # Normally evaluate `self.n_windows` windows, unless if windows would be less than 15min apart from each other
        max_steps = max((ts_duration - self.min_window) // pd.Timedelta('15m'), 1)
        for window_size_sec in np.linspace(start=ts_duration.total_seconds(),
                                           stop=self.min_window.total_seconds(),
                                           num=min(self.n_windows, max_steps)):
            window_duration = pd.Timedelta(int(window_size_sec), 's')
            min_timestamp = ts.index[-1] - window_duration
            window = ts.copy()[ts.index >= min_timestamp]
            r2, slope, reg_line = self.do_linear_regression(window)
            log.debug('Checking linear fit with window of size `%s`: (r2: %0.3f, slope: %0.3f)',
                      window_duration, r2, slope)

            if r2 > self.req_trend_r2:
                self.prediction_trend = pd.Series(reg_line, index=window.index)
                exit_time = self.calc_exit_time(current_value, slope)
                return {'start': window.index[0], 'stop': window.index[-1],
                        'duration': window.index[-1] - window.index[0],
                        'exit_time': exit_time, 'slope': slope, 'r2': r2}

        return None

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
