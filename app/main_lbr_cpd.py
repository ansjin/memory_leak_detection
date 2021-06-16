import warnings  # `do not disturbe` mode
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
from .operations import Operations
import sys
sys.path.append('../')
from mem_leak_detection  import MemLeakDetectionAlgorithm
from mem_leak_detection import MemLeakDetectionAlgorithmPolyFit
from mem_leak_detection import MemLeakDetectionAlgorithmChangePoints
from mem_leak_detection import Precog


class MainLBRCPD:

    def __init__(self, req_trend_r2=0.6, min_value=0, critical_time="5000d"):
        self.host = "10.195.1.185"
        self.port = 8086
        self.user = "root"
        self.password = "root"
        self.dbname = "lrz_ccs_main"
        self.anomalous_dbname = "lrz_ccs_main_anomaly"
        self.measurement_names_orig = ['host_mem', 'ccs_ccs_1', 'prometheus']
        self.op_obj = Operations()
        self.client = InfluxDBClient(self.host, self.port, self.user, self.password, self.dbname)

        self.client_anomaly = InfluxDBClient(self.host, self.port, self.user, self.password, self.anomalous_dbname)
        self.df_client_anomaly = DataFrameClient(self.host, self.port, self.user, self.password, self.anomalous_dbname)
        self.train_data_set_percent = 0.25

        self.client.create_database(self.anomalous_dbname)
        self.protocol = 'line'
        self.measurements = ["host_mem_lbr_cpd", 'ccs_ccs_1_lbr_cpd', 'prometheus_lbr_cpd']

        self.req_trend_r2 = req_trend_r2

    def train(self):
        iter = 0
        for orig_measurement in self.measurement_names_orig:
            if orig_measurement == "host_mem":
                mem_used_df = self.op_obj.get_host_memory_usage(self.client)
                dataset = pd.DataFrame(mem_used_df['used'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']
            else:
                mem_used_df = self.op_obj.get_container_memory_usage(self.client,orig_measurement)
                dataset = pd.DataFrame(mem_used_df['usage'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']

            dataset.columns = ['Value']
            p1 = MemLeakDetectionAlgorithmChangePoints(min_r2=self.req_trend_r2)
            p1 = p1.fit(dataset)
            dataset_n = p1.predict(dataset)
            dataset['anomalous'] = 0
            dataset['trend'] = 0

            if len(dataset_n) > 0:
                dataset['anomalous'].values[len(dataset['anomalous'].values) - len(dataset_n.index):] = 1
                dataset['trend'].values[len(dataset['anomalous'].values) - len(dataset_n.index):] = dataset_n[
                    'trend']

            else:
                None
            self.client_anomaly.drop_measurement(self.measurements[iter])

            self.df_client_anomaly.write_points(dataset, measurement=self.measurements[iter], protocol=self.protocol)
            iter = iter + 1

    def predict(self):
        iter = 0
        for orig_measurement in self.measurement_names_orig:
            if orig_measurement == "host_mem":
                mem_used_df = self.op_obj.get_host_memory_usage(self.client)
                dataset = pd.DataFrame(mem_used_df['used'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']
            else:
                mem_used_df = self.op_obj.get_container_memory_usage(self.client, orig_measurement)
                dataset = pd.DataFrame(mem_used_df['usage'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']

            dataset.columns = ['Value']
            p1 = MemLeakDetectionAlgorithmChangePoints(min_r2=self.req_trend_r2)
            p1 = p1.fit(dataset)
            dataset_n = p1.predict(dataset)
            dataset['anomalous'] = 0
            dataset['trend'] = 0

            if len(dataset_n) > 0:
                dataset['anomalous'].values[len(dataset['anomalous'].values) - len(dataset_n.index):] = 1
                dataset['trend'].values[len(dataset['anomalous'].values) - len(dataset_n.index):] = dataset_n[
                    'trend']

            else:
                None
            self.client_anomaly.drop_measurement(self.measurements[iter])

            self.df_client_anomaly.write_points(dataset, measurement=self.measurements[iter], protocol=self.protocol)
            iter = iter + 1







