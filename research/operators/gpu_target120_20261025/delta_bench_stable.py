"""Full-matrix retest after observed simultaneous GPU timing regime changes.

Keep the kernels and event/median estimator unchanged. Use six seconds of
additional warmup before each boundary and require at least 30 paired rounds.
Retain every sample; no outlier deletion, retries, or best-run selection.
"""
import statistics
import time
import run as runner

original_timing=runner.timing

def stabilized_timing(graphs,rounds):
    assert rounds>=30
    start=time.monotonic()
    while time.monotonic()-start<6:
        for graph in graphs.values():graph.replay()
        runner.ACC.synchronize()
    result=original_timing(graphs,rounds)
    names=list(graphs)
    # Diagnostic only: the headline remains ratio of marginal medians.
    base=result['samples_us'][names[0]]
    result['paired_ratio_median']={name:statistics.median(a/b for a,b in zip(base,result['samples_us'][name])) for name in names[1:]}
    result['additional_warmup_seconds']=6
    result['retained_all_samples']=True
    return result

runner.timing=stabilized_timing
if __name__=='__main__':
    with runner.ACC.stream(runner.ACC.Stream()):runner.main()
