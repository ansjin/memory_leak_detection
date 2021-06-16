class SQLQueries:
    def __init__(self):
        None

    def get_host_cpu_usage(self):
        host_cpu = "SELECT median(value) " \
            "FROM node_cpu_seconds_total " \
            "WHERE mode='user' GROUP BY time(1m)"
        return host_cpu


    def get_total_host_memory(self):
        host_total_memory = "select value from node_memory_MemTotal_bytes "
        return host_total_memory

    def get_host_free_memory(self):
        free_memory = "select value from node_memory_MemFree_bytes "
        return free_memory

    def get_host_cache_memory(self):
        cache_memory = "select value from node_memory_Cached_bytes "
        return cache_memory

    def get_host_buffer_memory(self):
        buffer_memory = "select value from node_memory_Buffers_bytes "
        return buffer_memory



    def get_container_memory(self, container_name):
        container_memory = "SELECT median(value) as usage " \
                             "FROM container_memory_usage_bytes " \
                             "where \"name\"='" + container_name + "' GROUP BY time(1m)"
        return container_memory

    def get_container_cpu(self, container_name):
        container_cpu = "SELECT median(value) as usage " \
                             "FROM container_cpu_user_seconds_total " \
                             "where \"name\"='" + container_name + "' GROUP BY time(1m)"
        return container_cpu

