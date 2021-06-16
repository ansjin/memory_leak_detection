import collections
import copy
from functools import wraps
import itertools
import numpy as np


AnomalousPoint = collections.namedtuple('AnomalousPoint', ['timestamp', 'value', 'label'])

ANOMALOUS_SET = 'set'
ANOMALOUS_INTERVAL = 'interval'


class AnomalousGroup:
    """Group of anomalous points (can be a set or continuous interval)"""

    def __init__(self, _points, _label='anomalies', _type=ANOMALOUS_SET):
        self.points = _points
        self.label = _label
        self.type = _type

    def __repr__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.points == other.points and self.label == other.label and self.type == other.type

    def __len__(self):
        return len(self.points)

    def get_timestamps(self):
        point_timestamps = [x.timestamp for x in self.points]
        if self.type is ANOMALOUS_SET:
            return point_timestamps
        elif self.type is ANOMALOUS_INTERVAL:
            return list(range(min(point_timestamps), max(point_timestamps) + 1))


def make_uniform_result(X, anomalies_bitmap, group_label, point_label):
    """Make uniform result of anomaly detection

    :param X: test data, 1-d array
    :param anomalies_bitmap: 1-d array, where True marks an anomalous point
    :param group_label: label for all anomalous groups
    :param point_label: label for all anomalous points
    :return: list of anomalous groups, consisting a list of labelled points
    """
    anomalous_points = itertools.compress(enumerate(X), anomalies_bitmap)
    anomalous_points = [AnomalousPoint(int(t), v, point_label) for t, v in anomalous_points]
    anomalies = [AnomalousGroup(anomalous_points, group_label, 'set')] if anomalous_points else []
    return anomalies


def make_uniform_result_group(X, anomalies_bitmap, group_label, point_label, group_interval):
    """Make uniform result of anomaly detection

    :param X: test data, 1-d array
    :param anomalies_bitmap: 1-d array, where True marks an anomalous point
    :param group_label: label for all anomalous groups
    :param point_label: label for all anomalous points
    :param group_interval: interval between different groups
    :return: list of anomalous groups, consisting a list of labelled points
    """
    idx_lst = np.where(np.array(anomalies_bitmap) == True)[0]
    anomalies = list()
    idx = 0
    if len(idx_lst) <= 0:
        return anomalies

    one_outliers = [idx_lst[0]]
    while idx < len(idx_lst) - 1:
        if idx_lst[idx + 1] - idx_lst[idx] < group_interval:
            one_outliers.append(idx_lst[idx + 1])
        else:
            anomalies.append(AnomalousGroup([AnomalousPoint(int(index), X[index], point_label)
                                             for index in one_outliers], group_label))
            one_outliers = [idx_lst[idx + 1]]
        idx = idx + 1
    # add the last group
    anomalies.append(AnomalousGroup([AnomalousPoint(int(index), X[index], point_label)
                                     for index in one_outliers], group_label))

    return anomalies
