"""Retain every active/precision-limited region and restart at Q=2^30."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse,json,uuid,hashlib,shutil
from pathlib import Path
import mlx.core as mx
import torch
import compact_certified_search_v4_gpu as engine
import wd_proof_executor as scheduling
import certified_precision30_gpu as precision
import precision30_control_gpu as control
from check_compact_splits_mps import transfer,D
from n3_branches_gpu import ROOT

def configure():
    mx.load=scheduling.ram_staged_load
    engine.HEADER=precision.HEADER;engine.MODEL_HEADER=precision.MODEL_HEADER
    engine.reference_header=lambda:precision.REFERENCE_HEADER
    engine.model_inputs=precision.model_inputs;engine.build_kernel=precision.build_kernel
    engine.split_rows=control.split_rows;engine.check_split_union=control.check_split_union
    scheduling.original_minimum=precision.evaluate_minimum
    engine.evaluate=scheduling.segmented_evaluate;engine.evaluate_minimum=scheduling.segmented_minimum
    engine.batch_step=scheduling.checked_batch_step

def provenance(directory):
    config=json.loads((directory/'config.json').read_text())
    config['fixed_point_bits']=30;config['depth_limit']=360;config['execution_dispatch_rows']=4096
    config['array_load_policy']='Eager RAM-owned file payload before GPU use'
    for name in ['precision30_runner.py','certified_precision30_gpu.py','precision30_control_gpu.py','wd_proof_executor.py']:
        p=ROOT/name;config['code_file_sha256'][str(p)]=engine.sha(p)
        shutil.copyfile(p,directory/('snapshot_'+name))
    (directory/'config.json').write_text(json.dumps(config,indent=2))

def upgrade(parent,output_base):
    parent=Path(parent).resolve();old=json.loads((parent/'checkpoint.json').read_text())
    assert old['fixed_point_bits']==24 and old['offline_replay_passed']
    tests=json.loads((ROOT/'certified_spatial/precision30_checks.json').read_text());assert tests['passed']
    files=old['frontier_files']+old['terminal_files']
    raw=engine.load_rows(files);co=precision.constants(old['target_numerator'],old['target_denominator'])
    chunks=[];outside=0
    for first in range(0,raw.shape[0],32768):
        block=raw[first:first+32768]
        bounds=block[:,3:].astype(mx.int64)<<6
        assert mx.all((bounds>=-2147483648)&(bounds<=2147483647)).item()
        bounds=bounds.at[:,19].add(mx.minimum(bounds[:,19],co[6])-bounds[:,19])
        lifted=mx.concatenate([block[:,:3],bounds.astype(mx.int32)],axis=1)
        keep=lifted[:,21]<=lifted[:,22]
        # Independent MPS verification of scaling, target clipping and metadata.
        tb=transfer(block);tl=transfer(lifted)
        assert torch.equal(tl[:,:3],tb[:,:3])
        assert torch.equal(tl[:,3:22].to(torch.int64),tb[:,3:22].to(torch.int64)<<6)
        target=torch.tensor(co[6].item(),device=D,dtype=torch.int64)
        assert torch.equal(tl[:,22].to(torch.int64),torch.minimum(tb[:,22].to(torch.int64)<<6,target))
        assert torch.equal(torch.tensor(keep.tolist(),device=D),tl[:,21]<=tl[:,22])
        selected=engine.filter_rows(lifted,keep);outside+=block.shape[0]-engine.rows_count(selected)
        if selected is not None:chunks.append(selected)
    rows=engine.concatenate(chunks);directory=Path(output_base)/f'precision_seed_{uuid.uuid4().hex[:8]}';directory.mkdir()
    if rows.shape[0]:mx.save(str(directory/'frontier.npy'),rows)
    record={'passed':True,'parent_checkpoint':str(parent/'checkpoint.json'),'parent_sha256':engine.sha(parent/'checkpoint.json'),
      'input_files':[{'path':p,'sha256':engine.sha(p)} for p in files],
      'old_active_rows':old['unresolved_active_boxes'],'old_terminal_rows':old['unresolved_precision_boxes'],
      'input_rows':raw.shape[0],'retained_rows':rows.shape[0],'outside_target_rows':outside,
      'retained_sha256':engine.digest(rows),'shift_bits':6,'fixed_point_bits':30,
      'independent_MPS_lift_check_passed':True,'all_old_terminal_regions_reconsidered':True,
      'cpu_numerical_fallback':False,'global_optimality_proved':False}
    (directory/'precision_upgrade.json').write_text(json.dumps(record,indent=2))
    cp=dict(old);cp.update({'format':'precision-upgrade-v1','parent_directory':str(parent),
      'fixed_point_bits':30,'unresolved_active_boxes':rows.shape[0],'unresolved_precision_boxes':0,
      'frontier_files':[str(directory/'frontier.npy')] if rows.shape[0] else [],'terminal_files':[],
      'offline_replay_passed':True,'complete_below_target_exclusion':False,'all_minimum_candidate_regions_excluded':False,
      'certified_lower_bound':None,'new_boxes_evaluated':0,'new_interval_exclusions':0,'new_splits':0,
      'phase':'precision_upgrade','global_optimality_proved':False})
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    (directory/'checkpoint_audit.json').write_text(json.dumps({**record,'saved_splits_checked':0,
      'frontier_boxes':rows.shape[0],'frontier_profiles':old['frontier_profiles']},indent=2))
    print(json.dumps(record,indent=2),flush=True);print('PRECISION_SEED',directory,flush=True)
    return directory

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--output-base');p.add_argument('--replay');p.add_argument('--upgrade',action='store_true')
    p.add_argument('--seconds',type=float,default=120);a=p.parse_args();configure()
    if a.replay:
        directory=Path(a.replay).resolve();cp=json.loads((directory/'checkpoint.json').read_text())
        assert cp['fixed_point_bits']==30
        if not cp.get('offline_replay_passed'):provenance(directory)
        engine.replay(directory)
    else:
        parent=Path(a.parent).resolve()
        if a.upgrade:parent=upgrade(parent,a.output_base)
        assert json.loads((parent/'checkpoint.json').read_text())['fixed_point_bits']==30
        directory=engine.run(parent,a.seconds,8192,30000000,1,None,a.output_base)
        provenance(directory)

if __name__=='__main__':main()
