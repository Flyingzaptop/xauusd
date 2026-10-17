import torch,os,shutil
print({'gpu':torch.cuda.get_device_name(0),'vram_GB':torch.cuda.get_device_properties(0).total_memory/1e9,'torch':torch.__version__,'disk':str(shutil.disk_usage('/content'))})
print(open('/proc/meminfo').read().splitlines()[:3])
