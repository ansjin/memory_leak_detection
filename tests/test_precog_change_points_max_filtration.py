import sys
import pandas as pd
import xlrd
from mem_leak_detection import Precog
sys.path.append('../')
from datetime import timedelta
from sklearn.metrics import f1_score
import time
class TestPrecogChangePointsMaxFiltration:
    def __init__(self):
        self.name = 'Precog_Change_Points_Max Filtration'
        self.simulate_online = True
        self.threshold = 0.7
        self.negative = False

    @staticmethod
    def convert_timestamp(df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        return df

    def execute(self, file):
        xls = xlrd.open_workbook(file, on_demand=True)
        if "Train Data" in xls.sheet_names():
            dataset_train = pd.read_excel(file, "Train Data")
        elif "Sheet1" in xls.sheet_names():
            dataset_train = pd.read_excel(file, "Sheet1")
        elif "Data Train" in xls.sheet_names():
            dataset_train = pd.read_excel(file, "Data Train")
        else:
            print(file)
            return []

        dataset_true_labels = pd.DataFrame()
        if "Anomalies" in xls.sheet_names():
            dataset_true_labels = pd.read_excel(file, "Anomalies", header=0)
            dataset_true_labels['Start'] = pd.to_datetime(dataset_true_labels['Start'], unit='s')
            dataset_true_labels['End'] = pd.to_datetime(dataset_true_labels['End'], unit='s')

        dataset_test = pd.read_excel(file, "Test Data")
        dataset_train = self.convert_timestamp(dataset_train)
        dataset_test = self.convert_timestamp(dataset_test)

        dataset_test['True_Labels'] = 0

        for row in range(0, len(dataset_true_labels.index)):
            dataset_test['True_Labels'][
            dataset_true_labels['Start'].values[row]:dataset_true_labels['End'].values[row]] = 1

        p1 = Precog(use_max_values_as_filter=True)
        p1 = p1.fit(dataset_train.Value)
        start = time.process_time()
        if self.simulate_online:
            points = []
            actual_labels = dataset_test['True_Labels'].values
            predicted_labels = []
            for i in range(1, len(dataset_test.Value.values) + 1):
                point = p1.predict(dataset_test.Value[0: i])
                if len(point) > 0:
                    points.append(point)
                    predicted_labels.append(1)
                else:
                    predicted_labels.append(0)

            dataset_test['Updated_Labels'] = 0
            if all(elem == 0 for elem in actual_labels):
                self.negative = True

            if p1.prediction_trend is not None:
                dataset_test['Updated_Labels'][p1.prediction_trend.index[0]:p1.prediction_trend.index[-1]] = 1
            f1score = f1_score(actual_labels, predicted_labels)
            if all(elem == 0 for elem in actual_labels) and all(elem == 0 for elem in predicted_labels):
                # print(file, 1)
                score = 1
            elif all(elem == 1 for elem in actual_labels) and all(elem == 1 for elem in predicted_labels):
                # print(file, 1)
                score = 1
            elif p1.prediction_trend is not None:
                #print(file, "score:", f1score, "updated_score:",
                #      f1_score(actual_labels, dataset_test['Updated_Labels'].values))
                score = f1_score(actual_labels, dataset_test['Updated_Labels'].values)
                # print(predicted_labels)
            elif p1.prediction_trend is None and self.negative is True and all(
                    elem == 0 for elem in predicted_labels) is not True and all(elem == 0 for elem in actual_labels):
                score = 1
            else:
                #print(file, "score:", f1score)
                score = f1score

            if score >= self.threshold and self.negative == False:
                predict_time = time.process_time() - start
                return [1], predict_time
            elif score >= self.threshold and self.negative == True:
                predict_time = time.process_time() - start
                return [], predict_time
            elif score < self.threshold and self.negative == True: # negative cases but there were some prediction
                predict_time = time.process_time() - start
                return [1], predict_time
            else:
                predict_time = time.process_time() - start
                return [], predict_time

            if len(points) > 0:
                point = points[-1]
                # print(point)
                timestamp = [pd.to_datetime(x.timestamp, unit="ms") for x in point[0].points][-1]
                # print(dataset_test.index[-1] - dataset_test.index[len(dataset_test.index) - 2])
                if (dataset_test.index[-1] - timestamp) > timedelta(minutes=30):
                    #print("Not equal", dataset_test.index[-1] - timestamp)
                    return []
                else:
                    return points
            else:
                return points
        else:
            points = p1.predict(dataset_test.Value)
            predict_time = time.process_time() - start
            return points, predict_time
