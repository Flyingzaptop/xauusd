import json,glob,sys
root=sys.argv[1] if len(sys.argv)>1 else '.'
for f in sorted(glob.glob(root+'/*/result.json')):
    r=json.load(open(f));m=r['test_means'];lm=r['learned_minus']
    print(f.split('/')[-2],'lr',r['lr'],'oracle %.2f'%r['oracle_mean'],' '.join('%s=%.2f'%(k,v) for k,v in m.items()))
    for c in ['money','initial','constant','shuffled','learned_zero_future']:
        x=lm[c];print('   learned-%-20s %+.3f CI[%+.3f,%+.3f] wins %d/%d'%(c,x['mean_diff'],*x['ci95'],x['wins'],x['n']))
