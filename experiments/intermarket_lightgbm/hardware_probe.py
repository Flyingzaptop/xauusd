import os,psutil,shutil,json
print(json.dumps({'cpus':os.cpu_count(),'ram_GB':psutil.virtual_memory().total/1e9,'disk_GB':shutil.disk_usage('/content').free/1e9}))
