"""Teacher input/score diagnostics on user-supplied real tensors or synthetic fixtures."""
import numpy as np,torch

def diagnose(teacher,x,future,outcome=None,seed=87):
    teacher.eval();rng=np.random.default_rng(seed);order=torch.tensor(rng.permutation(len(x)))
    with torch.no_grad():
        actual=teacher(x,future);v=actual.mean.numpy();noise=actual.variance.numpy()
        sf=teacher(x,future[order]).mean.numpy();sp=teacher(x[order],future).mean.numpy();zero=teacher(x,torch.zeros_like(future)).mean.numpy()
    result={'rows':len(x),'score_mean':v.mean(0).tolist(),'score_std':v.std(0).tolist(),'mean_sampling_std':np.sqrt(noise).mean(0).tolist(),'future_shuffle_mean_abs_change':np.abs(v-sf).mean(0).tolist(),'past_shuffle_mean_abs_change':np.abs(v-sp).mean(0).tolist(),'zero_future_mean_abs_change':np.abs(v-zero).mean(0).tolist(),'score_semantics':'Uncalibrated reward scores, not win probabilities'}
    if outcome is not None:
        result['pearson_to_fixed_exit_net']=[float(np.corrcoef(v[:,j],outcome[:,j])[0,1]) if np.std(outcome[:,j])>0 and np.std(v[:,j])>0 else None for j in range(2)]
    return result,{'actual':v,'future_shuffled':sf,'past_shuffled':sp,'future_zero':zero}
