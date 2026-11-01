import hashlib, json, os, statistics, sys, time, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
sys.path.append(str(ROOT/'family_completion_20261024'))
import run_cuda as e
EMIT=e.emit
import torch, triton
DEV=e.DEVICE
ACC=e.ACCEL
CASES=[dict(B=b,T=t,H=h,D=128) for b,t,h in [(1,512,4),(1,2048,4),(1,8192,4),(1,2048,16),(2,2048,16),(1,8192,16)]]
NAMES=['dq','dk','dv','dg','dbeta','dh0']
def metrics(ref,out,names=NAMES):
    return {n:e.error(r,x,.02 if n in ('dg','dbeta') else .008) for n,r,x in zip(names,ref,out)}
def check(m): return all(v['pass'] for v in m.values())
def emit(x,path): EMIT(x,path)
def hashes():
    files=sorted(HERE.glob('*.py'))
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    (HERE/'results').mkdir(exist_ok=True)
    archive=HERE/'results'/('sources_'+digest+'.zip')
    if not archive.exists():
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.name)
    return manifest
def capture(fn):
    for _ in range(3): last=fn()
    ACC.synchronize()
    graph=ACC.CUDAGraph() if DEV=='cuda' else ACC.NPUGraph()
    with ACC.graph(graph,stream=ACC.current_stream()): held=[fn(),fn()]
    graph.replay();ACC.synchronize()
    return graph,held
def timing(graphs,rounds):
    names=list(graphs)
    start=time.monotonic()
    while time.monotonic()-start<2:
        for n in names: graphs[n].replay()
        ACC.synchronize()
    a,b=ACC.Event(enable_timing=True),ACC.Event(enable_timing=True)
    a.record()
    for _ in range(4):
        for n in names: graphs[n].replay()
    b.record();ACC.synchronize()
    one=a.elapsed_time(b)/(4*len(names))
    repeat=max(1,min(1024,int(30/max(one,.001))))
    samples={n:[] for n in names};orders=[]
    for r in range(rounds):
        shift=(r//2)%len(names)
        order=names[shift:]+names[:shift]
        if r%2:order.reverse()
        order=order+order[::-1];orders.append(order);events=[]
        for n in order:
            a,b=ACC.Event(enable_timing=True),ACC.Event(enable_timing=True)
            a.record()
            for _ in range(repeat): graphs[n].replay()
            b.record();events.append((n,a,b))
        ACC.synchronize()
        for n,a,b in events:samples[n].append(a.elapsed_time(b)*1000/(repeat*2))
    return dict(samples_us=samples,median_us={n:statistics.median(v) for n,v in samples.items()},orders=orders,replays=repeat,unroll=2)
