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


class Main:

    def __init__(self, req_trend_r2=0.6, min_value=0, critical_time="5000d"):
        self.host = "10.195.1.185"
        self.port = 8086
        self.user = "root"
        self.password = "root"
        self.dbname = "lrz_ccs_main"
        self.anomalous_dbname = "lrz_ccs_main_anomaly"
        self.container_names = ['cadvisor', 'ccs_ccs_1', 'nodeexporter', 'prometheus', 'pushgateway']
        self.op_obj = Operations()
        self.client = InfluxDBClient(self.host, self.port, self.user, self.password, self.dbname)

        self.client_anomaly = InfluxDBClient(self.host, self.port, self.user, self.password, self.anomalous_dbname)
        self.df_client_anomaly = DataFrameClient(self.host, self.port, self.user, self.password, self.anomalous_dbname)
        self.train_data_set_percent = 0.25

        self.client.create_database(self.anomalous_dbname)
        self.protocol = 'line'
        self.measurements = ["host_mem", 'ccs_ccs_1', 'prometheus']
        self.iter = int(0)
        self.train_data_set_length = [0] * len(self.measurements)
        self.p1 = [None] * len(self.measurements)
        for measurement in self.measurements:
            self.p1[self.iter] = Precog(req_trend_r2=req_trend_r2, min_value=min_value, critical_time=critical_time)
            self.iter = self.iter + 1

    def train(self):
        iter = 0
        for measurement in self.measurements:
            if measurement == "host_mem":
                mem_used_df = self.op_obj.get_host_memory_usage(self.client)
                dataset = pd.DataFrame(mem_used_df['used'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']
            else:
                mem_used_df = self.op_obj.get_container_memory_usage(self.client,measurement)
                dataset = pd.DataFrame(mem_used_df['usage'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']

            self.train_data_set_length[iter] = int(self.train_data_set_percent * len(dataset.mem_util_percent.values))

            self.p1[iter] = self.p1[iter].fit(
                dataset.mem_util_percent[:self.train_data_set_length[iter]])
            points = self.p1[iter].predict(
                dataset.mem_util_percent[self.train_data_set_length[iter]:])

            self.client_anomaly.drop_measurement(measurement)
            dataset['anomalous'] = 0

            if len(points) > 0:
                timestamps = [pd.to_datetime(x.timestamp, unit="ms") for x in points[0].points]
                anomalies = [x.value for x in points[0].points]
                df_anomalies = pd.DataFrame({'timestamp': timestamps, 'values': anomalies})
                df_anomalies.set_index("timestamp", inplace=True)

                dataset['anomalous'][df_anomalies.index[0]:df_anomalies.index[-1]] = 1
            else:
                None

            self.df_client_anomaly.write_points(dataset, measurement=measurement, protocol=self.protocol)
            iter = iter + 1

    def predict(self):
        iter = 0
        for measurement in self.measurements:
            if measurement == "host_mem":
                mem_used_df = self.op_obj.get_host_memory_usage(self.client)
                dataset = pd.DataFrame(mem_used_df['used'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']
            else:
                mem_used_df = self.op_obj.get_container_memory_usage(self.client, measurement)
                dataset = pd.DataFrame(mem_used_df['usage'].values)
                dataset.index = mem_used_df['time'].values
                dataset.columns = ['mem_util_percent']

            points = self.p1[iter].predict(
                dataset.mem_util_percent[self.train_data_set_length[iter]:])

            dataset['anomalous'] = 0
            self.client_anomaly.drop_measurement(measurement)

            if len(points) > 0:
                timestamps = [pd.to_datetime(x.timestamp, unit="ms") for x in points[0].points]
                anomalies = [x.value for x in points[0].points]
                df_anomalies = pd.DataFrame({'timestamp': timestamps, 'values': anomalies})
                df_anomalies.set_index("timestamp", inplace=True)

                dataset['anomalous'][df_anomalies.index[0]:df_anomalies.index[-1]] = 1
            else:
                None

            self.df_client_anomaly.write_points(dataset, measurement=measurement, protocol=self.protocol)
            iter = iter + 1







