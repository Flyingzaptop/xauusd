import subprocess,json
p=subprocess.Popen(['python','-u','/content/meta_reward/bench_patch10.py'],stdout=open('/content/meta_reward/bench_patch10.log','w'),stderr=subprocess.STDOUT,start_new_session=True);print(json.dumps({'benchmark_pid':p.pid}))
