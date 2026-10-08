"""Run/replay unchanged row-wise Metal models in smaller GPU dispatches.

Segmentation changes scheduling only. Full original batch hashes must still
match on replay. The archived model/controller hashes stay intact; this
executor and its dispatch size get an additional provenance record.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse,json,hashlib,shutil,gc
from pathlib import Path
import mlx.core as mx
import torch
import numpy as np
import compact_certified_search_v4_gpu as engine

LIMIT=4096
original_evaluate=engine.evaluate
original_minimum=engine.evaluate_minimum
original_batch_step=engine.batch_step
original_load=mx.load

def ram_staged_load(path,*args,**kwargs):
    if Path(path).suffix=='.npy' and not args and not kwargs:
        # Disk decoding/byte copying only. The numerical operations stay on GPU.
        payload=np.load(path,mmap_mode=None,allow_pickle=False).copy(order='C')
        return mx.array(payload)
    return original_load(path,*args,**kwargs)

def release_gpu_caches():
    mx.synchronize();torch.mps.synchronize()
    gc.collect();torch.mps.empty_cache();mx.clear_cache()

def checked_batch_step(*args,**kwargs):
    result=original_batch_step(*args,**kwargs)
    if result[0] is not None:mx.eval(result[0])
    if result[1] is not None:mx.eval(result[1])
    release_gpu_caches()
    return result

def segmented_evaluate(states,*args,**kwargs):
    parts=[]
    for first in range(0,states.shape[0],LIMIT):
        result=original_evaluate(states[first:first+LIMIT],*args,**kwargs)
        mx.eval(*result);mx.synchronize();parts.append(result)
    if len(parts)==1:return parts[0]
    return tuple(mx.concatenate([part[j] for part in parts],axis=0) for j in range(3))

def segmented_minimum(states,*args,**kwargs):
    parts=[]
    for first in range(0,states.shape[0],LIMIT):
        result=original_minimum(states[first:first+LIMIT],*args,**kwargs)
        mx.eval(result);mx.synchronize();parts.append(result)
    return parts[0] if len(parts)==1 else mx.concatenate(parts,axis=0)

def provenance(directory):
    config=json.loads((directory/'config.json').read_text())
    path=Path(__file__).resolve();h=hashlib.sha256(path.read_bytes()).hexdigest()
    config['code_file_sha256'][str(path)]=h
    config['execution_dispatch_rows']=LIMIT
    config['array_load_policy']='Eager file read and RAM-owned byte copy before GPU use'
    (directory/'config.json').write_text(json.dumps(config,indent=2))
    shutil.copyfile(path,directory/'executor_snapshot.py')

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--output-base');p.add_argument('--replay')
    p.add_argument('--seconds',type=float,default=120);p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--max-frontier',type=int,default=12000000);p.add_argument('--min-width',type=int,default=1)
    p.add_argument('--shell-numerator',type=int);p.add_argument('--shell-denominator',type=int,default=10000000)
    a=p.parse_args();engine.evaluate=segmented_evaluate;engine.evaluate_minimum=segmented_minimum
    engine.batch_step=checked_batch_step
    mx.load=ram_staged_load
    if a.replay:
        directory=Path(a.replay).resolve();cp=json.loads((directory/'checkpoint.json').read_text())
        if not cp.get('offline_replay_passed'):provenance(directory)
        result=engine.replay(directory)
        receipt={**result,'dispatch_segment_rows':LIMIT,'executor_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        receipts=directory.parent/'execution_receipts';receipts.mkdir(exist_ok=True)
        (receipts/f'{directory.name}_segmented.json').write_text(json.dumps(receipt,indent=2))
    else:
        shell=(a.shell_numerator,a.shell_denominator) if a.shell_numerator else None
        directory=engine.run(a.parent,a.seconds,a.batch,a.max_frontier,a.min_width,shell,a.output_base)
        provenance(directory)

if __name__=='__main__':main()
