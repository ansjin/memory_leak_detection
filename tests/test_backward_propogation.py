import sys
import pandas as pd
import xlrd
from mem_leak_detection import MemLeakDetectionAlgorithm
sys.path.append('../')


class TestBackwardProp:
    def __init__(self):
        self.name = 'Backward_Propagation'

    @staticmethod
    def convert_timestamp(df):
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df.set_index('Timestamp', inplace=True)
        return df

    def execute(self, file):
        df = pd.DataFrame()
        if len(pd.ExcelFile(file).sheet_names) > 1:
            xls = xlrd.open_workbook(file, on_demand=True)
            if "Train Data" in xls.sheet_names():
                dataset_train = pd.read_excel(file, "Train Data")
            elif "Sheet1" in xls.sheet_names():
                dataset_train = pd.read_excel(file, "Sheet1")
            else:
                dataset_train = pd.read_excel(file, "Data Train")
            dataset_test = pd.read_excel(file, "Test Data")
            df = dataset_train.append(dataset_test)
        else:
            df = pd.read_excel(file)

        df = self.convert_timestamp(df)
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        p1 = MemLeakDetectionAlgorithm()
        p1 = p1.fit(df)
        points = p1.predict(df)
        return points

