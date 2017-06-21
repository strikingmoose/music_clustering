import psutil

print psutil.cpu_percent()
print psutil.cpu_count()

print psutil.virtual_memory()[2]
print psutil.disk_partitions()