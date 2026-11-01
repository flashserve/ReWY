"""Run a frozen release suite, retaining each raw result and environment."""
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from .cli import doctor

def reproduce(device,output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    suite=json.loads(Path(__file__).with_name('suites').joinpath('release.json').read_text())
    (output/'suite.json').write_text(json.dumps(suite,indent=2)+'\n')
    (output/'environment.json').write_text(json.dumps(doctor(True),indent=2)+'\n')
    subprocess.run([sys.executable,'-m','fast_delta_rule','verify','--device',device,'--output',str(output/'verification.json')],check=True)
    rows=[]
    for op in suite['ops']:
        ratios=[]
        for b,t,h in suite['shapes_B_T_H']:
            dest=output/f'{op}_B{b}_T{t}_H{h}.json'
            subprocess.run([sys.executable,'-m','fast_delta_rule','benchmark','--device',device,
                '--op',op,'--batch',str(b),'--length',str(t),'--heads',str(h),'--rounds',str(suite['rounds']),
                '--boundary',suite['boundary'],'--policy',suite['policy'],'--output',str(dest)],check=True)
            ratios.append(json.loads(dest.read_text())['speedup'])
        rows.append(dict(op=op,device=device,boundary=suite['boundary'],policy=suite['policy'],
                         geometric_mean_speedup=math.exp(sum(map(math.log,ratios))/len(ratios))))
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))
