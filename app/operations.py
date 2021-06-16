import warnings  # `do not disturbe` mode
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
from .sql_queries import SQLQueries
from datetime import datetime


class Operations:

    def __init__(self):

        self.sql_query_obj = SQLQueries()

    def get_dataframe(self, query_resultset, valuename):
        qrs_json = query_resultset.raw
        np_list = qrs_json.get('series')[0].get('values')
        df = pd.DataFrame(np_list, columns=["time", valuename])
        return df

    def newCalculateDifferenceBetweenDatapoints(self, containerSec):
        firstRound = True
        last5thValue = 0
        last5thTimeStamp = 0
        every5thItem = 0
        results = {}
        for item in containerSec:
            # For getting every the rate of 30 seconds
            if firstRound:
                last5thValue = containerSec[item]
                last5thTimeStamp = item
                firstRound = False
            else:
                if every5thItem == 2:
                    # Calculation
                    # Cut the last 9 numbers away
                    divider = item - last5thTimeStamp
                    value = (containerSec[item] - last5thValue)
                    # Save to the new map
                    results[item] = (value / divider.total_seconds()) * 100
                    # Reset with the existing values
                    last5thValue = containerSec[item]
                    last5thTimeStamp = item
                    every5thItem = 0
            every5thItem = every5thItem + 1
        return results

    def transformDicToArrays(self, containerSec):
        xresult = []
        yresult = []
        for item in containerSec:
            xresult.append(item)
            yresult.append(containerSec.get(item, 1) / 2)
        return xresult, yresult

    def get_container_memory_usage(self, client, cname):
        """
        prometheus query is as follows -
        sum by (name)(container_memory_usage_bytes{image!=“",container_label_org_label_schema_group=""})
        :return:
        """

        # total machine memory
        memTotalQ = self.sql_query_obj.get_total_host_memory()

        memTotal_rs = client.query(memTotalQ)
        memTotal = self.get_dataframe(memTotal_rs, "total")

        # Container memory usage

        containerMemUsageQ = self.sql_query_obj.get_container_memory(cname)

        containerMemUsage_rs = client.query(containerMemUsageQ)
        containerMemUsage = self.get_dataframe(containerMemUsage_rs, "usage")

        containerMemUsage['usage'] = (containerMemUsage['usage'] / memTotal['total'].values[0]) * 100
        containerMemUsage['time'] = pd.to_datetime(containerMemUsage['time'])

        return containerMemUsage

    def get_host_memory_usage(self, client):
        """
        prometheus query:
        node_memory_MemTotal_bytes - (node_memory_MemFree_bytes+node_memory_Buffers_bytes+node_memory_Cached_bytes)
        :return:
        """
        # total memory
        memTotalQ = self.sql_query_obj.get_total_host_memory()

        memTotal_rs = client.query(memTotalQ)
        memTotal = self.get_dataframe(memTotal_rs, "total")

        # free memory
        memFreeQ = self.sql_query_obj.get_host_free_memory()

        memFree_rs = client.query(memFreeQ)
        memFree = self.get_dataframe(memFree_rs, "free")

        # buffered
        memBufferQ = self.sql_query_obj.get_host_buffer_memory()
        memBuffer_rs = client.query(memBufferQ)
        memBuffer = self.get_dataframe(memBuffer_rs, "buffer")

        # cached
        memCachedQ = self.sql_query_obj.get_host_cache_memory()

        memCached_rs = client.query(memCachedQ)
        memCache = self.get_dataframe(memCached_rs, "cache")

        # join dataframes for easier plotting
        tf_merge = pd.merge(memTotal, memFree, on="time")
        tfc_merge = pd.merge(tf_merge, memCache, on="time")
        tfcb_merge = pd.merge(tfc_merge, memBuffer, on="time")

        # add a column based on operation (used = total - {free+cache+buffer})
        tfcb_merge.apply(lambda row: row.total - (row.free + row.buffer + row.cache), axis=1)
        tfcb_merge['used'] = tfcb_merge.apply(
            lambda row: row.total - (row.free + row.buffer + row.cache),
            axis=1)
        tfcb_merge['time'] = pd.to_datetime(tfcb_merge['time'])
        tfcb_merge['used'] = (tfcb_merge['used'] / memTotal['total'].values[0]) * 100
        tfcb_merge['free'] = (tfcb_merge['free'] / memTotal['total'].values[0]) * 100
        tfcb_merge['buffer'] = (tfcb_merge['buffer'] / memTotal['total'].values[0]) * 100
        tfcb_merge['cache'] = (tfcb_merge['cache'] / memTotal['total'].values[0]) * 100
        # return final dataframe to plot

        return tfcb_merge

    def get_container_cpu_usage(self, client, cname):
        """
        sum by (name) (rate(container_cpu_usage_seconds_total{image!="",container_label_org_label_schema_group=""}[1m])) / scalar(count(node_cpu_seconds_total{mode="user"})) * 100

        :return:
        """

        q = self.sql_query_obj.get_container_cpu(cname)

        qrs = client.query(q)

        containerCPUUserSecondsTotal = {}

        containerResult = qrs
        containerPoints = containerResult.get_points()

        for item in containerPoints:
            datetime_object = datetime.strptime(item['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
            containerCPUUserSecondsTotal[datetime_object] = item['value']

        results = self.newCalculateDifferenceBetweenDatapoints(containerCPUUserSecondsTotal)
        xresult, yresult = self.transformDicToArrays(results)
        df = pd.DataFrame({'time': xresult, 'percentage': yresult})

        df['time'] = pd.to_datetime(df['time'])
        df_cut_xtreme = df
        return df_cut_xtreme

    def get_host_cpu_usage(self, client):
        """
        sum(rate(container_cpu_user_seconds_total{image!=""}[1m])) / count(node_cpu_seconds_total{mode="user"}) * 100
    sum(rate(node_cpu_seconds_total[1m])) by (mode) * 100 / scalar(count(node_cpu_seconds_total{mode="user"}))
        :return:
        """
        q = self.sql_query_obj.get_host_cpu_usage()
        qrs = client.query(q)

        containerCPUUserSecondsTotal = {}

        containerResult = qrs
        containerPoints = containerResult.get_points()

        for item in containerPoints:
            datetime_object = datetime.strptime(item['time'], '%Y-%m-%dT%H:%M:%S.%fZ')
            containerCPUUserSecondsTotal[datetime_object] = item['value']

        results = self.newCalculateDifferenceBetweenDatapoints(containerCPUUserSecondsTotal)
        xresult, yresult = self.transformDicToArrays(results)
        df = pd.DataFrame({'time': xresult, 'percentage': yresult})

        df['percentage'] = 100 - df['percentage']

        df['time'] = pd.to_datetime(df['time'])
        df_cut_xtreme = df
        return df_cut_xtreme

"""

    def get_container_names(self, client, host):
        q = "select value, \"name\" " \
            "from container_cpu_user_seconds_total where 'name'!=None limit 10"

        qrs = client.query(q)
        qrs_json = qrs.raw
        np_list = qrs_json.get('series')[0].get('values')
        df = pd.DataFrame(np_list, columns=["time", 'value', 'name'])
        return df
"""

