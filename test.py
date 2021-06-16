
import pandas as pd
from pylab import rcParams
import glob


import time
from sklearn.metrics import f1_score
import xlrd

from tests import TestBackwardProp
from tests import TestPolyFit
from tests import TestChangePoints
from tests import TestPrecogChangePointsTrainTestSame
from tests import TestPrecogChangePointsTrainTestSameMaxFiltration
from tests import TestPrecogChangePoints
from tests import TestPrecogChangePointsMaxFiltration
from tests import TestPrecogOnline

ignore_files = ["/mnt/datasets/iforesight/metrics/memleak_negative/memleak04/guangsheng_email_20190314.xlsx",
               "/mnt/datasets/iforesight/metrics/memleak_negative/memleak67/cmc_false_positive.xlsx"]

default_config = {
        'base_path': '/mnt/datasets/iforesight/metrics/',
        #'base_path': 'example/',
        'memleak_negative_name': 'memleak_negative',
        'memleak_positive_name': 'memleak_positive',
    }


class TestAlgorithms:

    def __init__(self,
                 base_path=default_config['base_path'],
                 memleak_negative_name=default_config['memleak_negative_name'],
                 memleak_positive_name=default_config['memleak_positive_name'],
                 ):
        self.mem_leak_negative_path = base_path + memleak_negative_name
        self.mem_leak_positive_path = "/home/anshuljindal/tmp" #base_path + memleak_positive_name
        self.mem_leak_positive_files = [f for f in glob.glob(self.mem_leak_positive_path + "**/**/*.xlsx", recursive=True)]
        self.mem_leak_negative_files = [f for f in glob.glob(self.mem_leak_negative_path + "**/**/*.xlsx", recursive=True)]
        self.total_mem_positive_files = len(self.mem_leak_positive_files)
        self.total_mem_negative_files = len(self.mem_leak_negative_files)

        self.total_true_positive = 0
        self.total_true_negative = 0

        self.false_negative = 0
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0

        self.true_labels = []
        self.pred_labels = []
        self.total_predict_time = 0

    def get_f1_score(self):
        score = f1_score(self.true_labels, self.pred_labels)
        return score

    def apply_algorithm_positive(self, file, algorithm_object, train_test=False):
        if train_test:
            if len(pd.ExcelFile(file).sheet_names) > 1:
                xls = xlrd.open_workbook(file, on_demand=True)
                if "Train Data" in xls.sheet_names() or "Sheet1" in xls.sheet_names() or "Data Train" in xls.sheet_names():
                    self.total_true_positive = self.total_true_positive + 1
                    self.true_labels.append(1)
                    points, predict_time  = algorithm_object.execute(file)
                    print(predict_time)
                    self.total_predict_time += predict_time
                else:
                    return
            else:
                return
        else:
            self.total_true_positive = self.total_true_positive + 1
            self.true_labels.append(1)
            points, predict_time = algorithm_object.execute(file)
            print(predict_time)
            self.total_predict_time += predict_time

        if len(points) > 0:
            self.true_positive = self.true_positive + 1
            self.pred_labels.append(1)
        else:
            self.false_negative = self.false_negative + 1
            self.pred_labels.append(0)

    def apply_algorithm_negative(self, file, algorithm_object, train_test=False):

        if train_test:
            if len(pd.ExcelFile(file).sheet_names) > 1:
                xls = xlrd.open_workbook(file, on_demand=True)
                if "Train Data" in xls.sheet_names() or "Sheet1" in xls.sheet_names() or "Data Train" in xls.sheet_names() :
                    self.total_true_negative = self.total_true_negative + 1
                    self.true_labels.append(0)
                    points, predict_time = algorithm_object.execute(file)
                    print(predict_time)
                    self.total_predict_time += predict_time
                else:
                    return
            else:
                return
        else:
            self.total_true_negative = self.total_true_negative + 1
            self.true_labels.append(0)
            points, predict_time = algorithm_object.execute(file)
            self.total_predict_time += predict_time

        if len(points) > 0:
            self.false_positive = self.false_positive + 1
            self.pred_labels.append(1)
        else:
            self.true_negative = self.true_negative + 1
            self.pred_labels.append(0)

    def test(self, algorithm_object, train_test):
        for file in self.mem_leak_positive_files:
            self.apply_algorithm_positive(file, algorithm_object, train_test)

        for file in self.mem_leak_negative_files:
            if file in ignore_files:
                continue
            self.apply_algorithm_negative(file, algorithm_object, train_test)



if __name__ == "__main__":
    algorithms = [#TestBackwardProp, TestPolyFit, TestChangePoints,
                  #TestPrecogChangePointsTrainTestSame,
                  #TestPrecogChangePointsTrainTestSameMaxFiltration,
                  #TestPrecogChangePoints,
                  TestPrecogChangePointsMaxFiltration,
                  TestPrecogOnline]
    idx = 0
    train_test = False
    for algorithm in algorithms:
        if idx > len(algorithms) - 4: # last three uses training and test data
            train_test = True
        start = time.process_time()
        algorithm_obj = algorithm()
        test_algorithm_object = TestAlgorithms()
        test_algorithm_object.test(algorithm_obj, train_test)
        f1_score_calc = test_algorithm_object.get_f1_score()
        print("Algorithm: ", algorithm_obj.name,
              "F1-Score: ", f1_score_calc,
              "Time Taken: ", time.process_time() - start,
              "Total Predict Time: ", test_algorithm_object.total_predict_time,
              "TP : ", test_algorithm_object.true_positive,
              "FP : ", test_algorithm_object.false_positive,
              "TN : ", test_algorithm_object.true_negative,
              "FN : ", test_algorithm_object.false_negative,
              "TTP : ", test_algorithm_object.total_true_positive,
              "TTN : ", test_algorithm_object.total_true_negative,)
        idx = idx + 1


