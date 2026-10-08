"""Replayable Q30 shell through a conservative enclosure of the exact optimum.

An algebraic lower-bound family is recorded as covered, never as physically
infeasible. The shell's dyadic upper endpoint exceeds the exact optimum, so
no rational exclusion claim is made for that endpoint. Geometry runs on GPU;
CPU only controls the run, records metadata, and copies/hashes file bytes.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import argparse,json,uuid,shutil,time
from pathlib import Path
import mlx.core as mx
import torch
import compact_boundary_search_gpu as engine
import precision30_runner as scheduling_setup
import certified_precision30_gpu as precision
import precision30_control_gpu as control
import wd_proof_executor as scheduling
from n3_branches_gpu import ROOT
from contact_catalog_gpu import OUT
from check_compact_splits_mps import transfer,D

def configure():
    scheduling_setup.configure()
    engine.HEADER=precision.HEADER;engine.MODEL_HEADER=precision.MODEL_HEADER
    engine.reference_header=lambda:precision.REFERENCE_HEADER
    engine.model_inputs=precision.model_inputs;engine.build_kernel=precision.build_kernel
    engine.split_rows=control.split_rows;engine.check_split_union=control.check_split_union
    engine.evaluate=scheduling.segmented_evaluate
    engine.evaluate_minimum=scheduling.segmented_minimum
    original=engine.batch_step
    def step(*args,**kwargs):
        result=original(*args,**kwargs)
        for item in result[:2]:
            if item is not None:mx.eval(item)
        scheduling.release_gpu_caches()
        return result
    engine.batch_step=step

def analytic_wall_families(masks,triples):
    wm=masks[triples]
    vertex=(wm[:,:,None]>>mx.array([0,4,8],dtype=mx.uint32))&15
    count=mx.stack([mx.sum(((vertex>>w)&1).astype(mx.int32),axis=2) for w in range(4)],axis=2)
    covered=mx.zeros((triples.shape[0],),dtype=mx.bool_)
    for corner,horizontal,vertical in [(5,1,3),(6,0,3),(9,1,2),(10,0,2)]:
        has_corner=mx.any((vertex&corner)==corner,axis=2)
        for i,j,k in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            covered|=has_corner[:,i]&(count[:,j,horizontal]>=2)&(count[:,k,vertical]>=2)
    return covered

def reference_wall_families(masks,triples):
    wm=masks[triples.to(torch.int64)]
    vertex=(wm[:,:,None]>>torch.tensor([0,4,8],device=D,dtype=torch.int32))&15
    bases=torch.stack([((vertex&(1<<w))!=0).sum(2)>=2 for w in range(4)],2)
    covered=torch.zeros(triples.shape[0],device=D,dtype=torch.bool)
    for hw in [0,1]:
        for vw in [2,3]:
            corner=(1<<hw)|(1<<vw)
            cp=((vertex&corner)==corner).any(2)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if len({i,j,k})==3:
                            covered|=cp[:,i]&bases[:,j,1-hw]&bases[:,k,5-vw]
    return covered

def roots(ids,co,low):
    ordinal=mx.arange(ids.size*8,dtype=mx.uint32)
    half=(co[6]+1)>>1
    domain=mx.concatenate([mx.broadcast_to(mx.stack([-half,half])[None,:],(6,2)),
        mx.broadcast_to(mx.array([-precision.Q,precision.Q],dtype=mx.int64)[None,:],(3,2)),
        mx.stack([low,co[6]])[None,:]],0).astype(mx.int32).reshape(20)
    meta=mx.stack([ids[ordinal>>3].astype(mx.int32),(ordinal&7).astype(mx.int32),mx.zeros_like(ordinal).astype(mx.int32)],1)
    rows=mx.concatenate([meta,mx.broadcast_to(domain[None,:],(ordinal.size,20))],1)
    mx.eval(rows);return rows

def provenance(directory):
    config=json.loads((directory/'config.json').read_text())
    config.update(fixed_point_bits=30,depth_limit=360,execution_dispatch_rows=4096,
        exact_boundary_goal=True,array_load_policy='Eager RAM-owned file payload before GPU use')
    names=['exact_boundary_runner_gpu.py','precision30_runner.py','certified_precision30_gpu.py',
        'precision30_control_gpu.py','wd_proof_executor.py','compact_certified_search_v4_gpu.py']
    for name in names:
        p=ROOT/name;config['code_file_sha256'][str(p)]=engine.sha(p)
        shutil.copyfile(p,directory/('snapshot_'+name))
    (directory/'config.json').write_text(json.dumps(config,indent=2))

def build_seed(parent,output_base):
    parent=Path(parent).resolve();old=json.loads((parent/'checkpoint.json').read_text())
    assert old['complete_below_target_exclusion'] and old['offline_replay_passed'] and old['fixed_point_bits']==30
    certificate=ROOT/'certified_spatial/corner_two_bases_certificate.json'
    assert json.loads(certificate.read_text())['passed']
    old_co=precision.constants(old['target_numerator'],old['target_denominator'])
    upper=old_co[8].item();inputs=precision.model_inputs(upper,precision.Q);co=inputs[3]
    low=old_co[5]
    assert (co[6]==old_co[8]).item() and (co[6]>=old_co[8]).item()
    assert (2*low>co[1]+precision.Q).item() and (co[6]<co[0]).item()
    families=analytic_wall_families(inputs[0],inputs[1]);mx.eval(families)
    rf=reference_wall_families(transfer(inputs[0]),transfer(inputs[1]))
    assert torch.equal(rf,torch.tensor(families.tolist(),device=D,dtype=torch.bool))
    ids=mx.load(str(ROOT/'certified_spatial/spatial_candidates.npy')).astype(mx.int32)
    lookup=mx.load(str(ROOT/'certified_spatial/rotation_critical_lookup.npy'))
    eligible=lookup[ids]>0
    active=eligible&~families[ids//50653]
    selected=engine.filter_rows(ids[:,None],active)[:,0]
    covered_count=mx.sum((eligible&families[ids//50653]).astype(mx.int32)).item()
    eligible_count=mx.sum(eligible.astype(mx.int32)).item()
    assert families[11823248//50653].item()
    rows=roots(selected,co,low)
    ts=transfer(selected);low_value=low.item();half=(co[6]+1)>>1
    td=torch.tensor([-half.item(),half.item()]*6+[-precision.Q,precision.Q]*3+[low_value,co[6].item()],device=D,dtype=torch.int32)
    for first in range(0,rows.shape[0],32768):
        block=transfer(rows[first:first+32768]);ix=torch.arange(first,first+block.shape[0],device=D,dtype=torch.int32)
        assert torch.equal(block[:,0],ts[(ix>>3).to(torch.int64)])
        assert torch.equal(block[:,1],ix&7) and (block[:,2]==0).all().item()
        assert torch.equal(block[:,3:],td[None,:].expand(block.shape[0],-1))
    directory=Path(output_base)/f'boundary_seed_{uuid.uuid4().hex[:8]}';directory.mkdir()
    if rows.shape[0]:mx.save(str(directory/'frontier.npy'),rows)
    mx.save(str(directory/'covered_wall_families.npy'),families)
    record={'passed':True,'parent_checkpoint':str(parent/'checkpoint.json'),'parent_sha256':engine.sha(parent/'checkpoint.json'),
        'side_lower_fixed':low_value,'side_upper_fixed':co[6].item(),'fixed_point_bits':30,
        'side_window_verified_on_GPU':True,'upper_contains_exact_reference':True,
        'independent_MPS_wall_classification':True,'independent_MPS_root_coverage':True,
        'critical_profiles':eligible_count,'analytic_covered_profiles':covered_count,'remaining_profiles':selected.size,
        'full_circle_charts_per_profile':8,'frontier_rows':rows.shape[0],'frontier_sha256':engine.digest(rows),
        'families_sha256':engine.digest(families),'analytic_certificate':str(certificate),
        'analytic_certificate_sha256':engine.sha(certificate),'cpu_numerical_fallback':False,
        'scope':'Analytic covered families have lower bound s0; all other full minimum profiles enter continuous interval roots.',
        'global_optimality_proved':False}
    paths=[Path(__file__).resolve(),ROOT/'corner_two_bases_certificate_gpu.py',ROOT/'certified_precision30_gpu.py']
    record['code_file_sha256']={str(p):engine.sha(p) for p in paths}
    record['input_file_sha256']={str(p):engine.sha(p) for p in [ROOT/'certified_spatial/spatial_candidates.npy',ROOT/'certified_spatial/rotation_critical_lookup.npy',OUT/'wall_masks.npy',OUT/'wall_triplets.npy']}
    for p in paths:shutil.copyfile(p,directory/('snapshot_'+p.name))
    (directory/'boundary_seed_certificate.json').write_text(json.dumps(record,indent=2))
    cp=dict(old);cp.update(format='exact-boundary-seed-v1',target_numerator=upper,target_denominator=precision.Q,
        parent_directory=str(parent),initial_jobs_total=eligible_count*8,initial_jobs_processed=eligible_count*8,
        initial_jobs_unprocessed=0,unresolved_active_boxes=rows.shape[0],unresolved_precision_boxes=0,
        frontier_profiles=selected.size,frontier_files=[str(directory/'frontier.npy')] if rows.shape[0] else [],terminal_files=[],
        complete_below_target_exclusion=False,certified_lower_bound=None,all_minimum_candidate_regions_excluded=False,
        offline_replay_passed=True,phase='exact_boundary_seed',exact_boundary_goal=True,
        analytic_seed_directory=str(directory),analytic_covered_profiles=covered_count,global_optimality_proved=False)
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    (directory/'checkpoint_audit.json').write_text(json.dumps({**record,'frontier_boxes':rows.shape[0],'frontier_profiles':selected.size,'saved_splits_checked':0},indent=2))
    print(json.dumps(record,indent=2),flush=True);print('BOUNDARY_SEED',directory,flush=True)
    return directory

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--output-base');p.add_argument('--replay')
    p.add_argument('--seed',action='store_true');p.add_argument('--seconds',type=float,default=120)
    a=p.parse_args();configure()
    if a.replay:
        directory=Path(a.replay).resolve()
        cp=json.loads((directory/'checkpoint.json').read_text());assert cp['exact_boundary_goal']
        if not cp.get('offline_replay_passed'):provenance(directory)
        engine.replay(directory)
    else:
        parent=build_seed(a.parent,a.output_base) if a.seed else Path(a.parent).resolve()
        assert json.loads((parent/'checkpoint.json').read_text())['exact_boundary_goal']
        directory=engine.run(parent,a.seconds,8192,30000000,1,None,a.output_base);provenance(directory)

if __name__=='__main__':main()
