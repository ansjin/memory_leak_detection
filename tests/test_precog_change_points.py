import sys
import pandas as pd
import xlrd
from mem_leak_detection import Precog
sys.path.append('../')
from datetime import timedelta

class TestPrecogChangePoints:
    def __init__(self):
        self.name = 'Precog_Change_Points'
        self.simulate_online = True

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
        else:
            dataset_train = pd.read_excel(file, "Data Train")
        dataset_test = pd.read_excel(file, "Test Data")
        dataset_train = self.convert_timestamp(dataset_train)
        dataset_test = self.convert_timestamp(dataset_test)
        p1 = Precog(use_max_values_as_filter=False)
        p1 = p1.fit(dataset_train.Value)

        if self.simulate_online:
            points = []
            for i in range(1, len(dataset_test.Value.values)):
                point = p1.predict(dataset_test.Value[0: i])
                if len(point) > 0:
                    points.append(point)

            if len(points) > 0:
                point = points[-1]
                #print(point)
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
            return points

