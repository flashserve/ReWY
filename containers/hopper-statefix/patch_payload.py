"""Apply a narrow state schedule fix to the immutable syncmma1 payload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

BASE='zhangyi2026/residual-wy-h100:20261030-cu126-triton371-syncmma1'
TAG='zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1'
parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/opt/h100-experiments'))
args=parser.parse_args();root=args.root.resolve()
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
for name,digest in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,('Base integrity',name)

changed=[]
def edit(name,old,new):
    path=root/name;text=path.read_text(encoding='utf-8')
    assert text.count(old)==1,(name,old)
    path.write_text(text.replace(old,new),encoding='utf-8',newline='\n')
    if name not in changed:changed.append(name)

edit('single/kernels/scalar_state_reference_gpu.py','num_warps=4,num_stages=stages)',
     "arch='sm80',num_warps=4,num_stages=stages)")
edit('single/kernels/scalar_state_reference_gpu.py','"""Fixed multi-stage control using the unchanged frozen reference JIT body.\n\nUsed only to isolate the full-K rewrite at identical BV/warps/stages.\nThe headline \'original\' baseline keeps its original autotuner unchanged.\n"""',
     '"""Synchronous-MMA optimized state launch; unchanged frozen JIT body.\n\nNative SM90 D256/BV16/three-stage lowering needs 315392 shared bytes.\nPer-launch SM80 lowering uses 215040 bytes, within the 232448-byte limit;\nfinal PTX/cubin still target the actual device. All dimensions use this\nschedule. Arithmetic, casts, warps, stages and the original autotuner\nstay unchanged. No global compiler override is used.\n"""')
edit('single/protocol.py','native K64 BV16 s3 state','SM80-lowered K64 BV16 s3 state')

for name in ('README.zh-CN.md','RUN_AND_RESULTS.zh-CN.md'):
    path=root/name
    text=path.read_text(encoding='utf-8').replace(BASE,TAG)
    text=('# D256 状态反向共享内存修复（2026-10-31）\n\n'
          '本版基于 syncmma1，仅将 DeltaNet 优化状态反向统一使用同步 MMA 降低路径；'
          '保留三阶段和四 warps，最终代码仍针对实际设备。SM90 D256 共享内存由 315,392 降至 215,040 字节，上限 232,448 字节。'
          'FP32 累加、运算正文、保存策略及冻结原始对照保持。\n\n'
          '离线检查通过真实 JIT warmup 编译实际调用参数，保留旧原生 SM90 超限负对照。'
          'A800 验证与 Hopper 实机验收分开记录，发布证据见 GitHub containers/hopper-statefix。\n\n'+text)
    path.write_text(text,encoding='utf-8',newline='\n');changed.append(name)

check=root/'check_state_codegen.py'
shutil.copyfile(Path(__file__).with_name('check_state_codegen.py'),check)
changed.append(check.name)
edit('test_image.py','class PackagingTests(unittest.TestCase):',
'''class PackagingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('H100_IMAGE_IMPORT_TEST')=='1','container-only offline compiler')
    def test_actual_state_sm90_resources(self):
        result=subprocess.run([sys.executable,str(Path(__file__).resolve().parent/'check_state_codegen.py')],
                              capture_output=True,text=True,timeout=900)
        self.assertEqual(result.returncode,0,result.stdout+'\\n'+result.stderr)
''')

single=json.loads((root/'single/manifest.json').read_text(encoding='utf-8'))
for name in changed:
    if name.startswith('single/'):
        rel=name.removeprefix('single/')
        single['files'][rel]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        single.setdefault('origins',{}).setdefault(rel,{})['statefix_adaptation']='Per-launch SM80 lowering, native-device assembly; frozen JIT arithmetic, warps and stages unchanged'
single['state_fix']=dict(date='2026-10-31',base=BASE,stages=3,lowering_arch='sm80')
(root/'single/manifest.json').write_text(json.dumps(single,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
changed.append('single/manifest.json')
for name in changed:
    manifest['files'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
manifest['state_fix']=dict(date='2026-10-31',base=BASE,tag=TAG,changed_files=changed,stages=3,lowering_arch='sm80',
                          precision='unchanged',original_baseline='unchanged')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(manifest['state_fix']))
