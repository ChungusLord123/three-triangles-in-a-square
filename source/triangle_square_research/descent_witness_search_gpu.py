"""Find rational velocity witnesses on GPU; accept only exact interval proofs.

The floating solver proposes candidates. Replay uses saved dyadic witnesses
and exact integer inequalities, not floating solver convergence or failures.
"""
import json,uuid,hashlib,shutil
from pathlib import Path
import mlx.core as mx
from n3_branches_gpu import ROOT
from descent_coefficients_gpu import coefficients,strict_certificate,M
from certified_precision30_gpu import Q,model_inputs
from compact_certified_search_gpu import filter_rows,load_rows,digest,sha
from wd_proof_executor import ram_staged_load
mx.set_default_device(mx.gpu);mx.load=ram_staged_load

def candidates(states,inputs,iterations=600):
    c,b,count,axes,available=coefficients(states,inputs);mx.eval(c,b,count,available)
    a=(c[:,:,:,0].astype(mx.float32)+c[:,:,:,1].astype(mx.float32))/(2*Q)
    bias=b.astype(mx.float32)/Q+0.03
    enabled=(mx.arange(M,dtype=mx.uint32)[None,:]<count[:,None])&(available[:,None]>0)
    codes=mx.arange(27,dtype=mx.int32)
    omega=(mx.stack([codes//9,(codes//3)%3,codes%3],axis=1)-1).astype(mx.float32)*4
    rhs=bias[:,None,:]-mx.sum(a[:,None,:,6:]*omega[None,:,None,:],axis=3)
    translation=mx.zeros((states.shape[0],27,6),dtype=mx.float32)
    linear=mx.broadcast_to(a[:,None,:,:6],(states.shape[0],27,M,6))
    norm=mx.sum(linear*linear,axis=3)
    @mx.compile
    def update(v):
        gap=rhs-mx.sum(linear*v[:,:,None,:],axis=3)
        gap=mx.where(enabled[:,None,:],gap,-1000000)
        j=mx.argmax(gap,axis=2)
        row=mx.take_along_axis(linear,j[:,:,None,None],axis=2)[:,:,0,:]
        largest=mx.take_along_axis(gap,j[:,:,None],axis=2)[:,:,0]
        denom=mx.take_along_axis(norm,j[:,:,None],axis=2)[:,:,0]
        delta=0.9*mx.maximum(largest,0)/(denom+1e-6)
        return mx.clip(v+delta[:,:,None]*row,-8,8)
    for step in range(iterations):
        translation=update(translation)
        if step%10==0:mx.eval(translation)
    v=mx.concatenate([translation,mx.broadcast_to(omega[None,:,:],(states.shape[0],27,3))],axis=2)
    quantized=mx.round(v*1024).astype(mx.int32)
    scores=[]
    for j in range(27):
        vel=quantized[:,j]
        selected=mx.where(vel[:,None,:]>=0,c[:,:,:,0],c[:,:,:,1])
        slack=mx.sum(selected*vel[:,None,:].astype(mx.int64),axis=2)-b*1024
        valid=(available>0)&mx.all((slack>0)|~enabled,axis=1)
        scores.append(mx.where(valid,mx.min(mx.where(enabled,slack,9223372036854775807),axis=1),-9223372036854775807))
    score=mx.stack(scores,axis=1);order=mx.argsort(-score,axis=1)[:,:4]
    selected=mx.take_along_axis(quantized,order[:,:,None],axis=1)
    good=mx.take_along_axis(score,order,axis=1)>0
    mx.eval(selected,good)
    return selected,good

def reduce(parent,output_base):
    parent=Path(parent).resolve();old=json.loads((parent/'checkpoint.json').read_text())
    assert old['fixed_point_bits']==30 and old['offline_replay_passed']
    source=load_rows(old['frontier_files']+old['terminal_files'])
    inputs=model_inputs(old['target_numerator'],old['target_denominator'])
    key=source[:,0]*8+source[:,1];order=mx.argsort(key);sorted_key=key[order]
    unique_mask=mx.concatenate([mx.array([True]),sorted_key[1:]!=sorted_key[:-1]])
    n=mx.sum(unique_mask.astype(mx.int32));mx.eval(n)
    pos=filter_rows(mx.arange(source.shape[0])[:,None],unique_mask)[:,0]
    reps=source[order[pos]];unique_keys=sorted_key[pos]
    velocity,good=candidates(reps,inputs)
    print('Velocity prototypes found for',mx.sum(mx.any(good,axis=1).astype(mx.int32)).item(),'of',reps.shape[0],'profile/chart groups',flush=True)
    kept=[];removed=0;accepted_groups=0
    for first in range(0,source.shape[0],4096):
        rows=source[first:first+4096];match=(rows[:,0]*8+rows[:,1])[:,None]==unique_keys[None,:]
        assert mx.all(mx.any(match,axis=1)).item();idx=mx.argmax(match,axis=1)
        c,b,count,axes,available=coefficients(rows,inputs)
        accepted=mx.zeros((rows.shape[0],),dtype=mx.bool_)
        for j in range(4):
            vel=velocity[idx,j];ok=good[idx,j]&strict_certificate(c,b,count,available,vel)&~accepted
            selected=filter_rows(rows,ok)
            if selected is not None:
                ix=filter_rows(mx.arange(rows.shape[0])[:,None],ok)[:,0]
                rc,rb,rn,ra,rv=coefficients(selected,inputs,axes=axes[ix],reference=True)
                assert mx.all(strict_certificate(rc,rb,rn,rv,vel[ix])).item()
                accepted_groups+=1
            accepted|=ok
        surviving=filter_rows(rows,~accepted);removed+=rows.shape[0]-(0 if surviving is None else surviving.shape[0])
        if surviving is not None:kept.append(surviving)
    result=mx.concatenate(kept,axis=0) if kept else mx.zeros((0,23),dtype=mx.int32)
    directory=Path(output_base)/f'descent_{uuid.uuid4().hex[:8]}';directory.mkdir()
    if result.shape[0]:mx.save(str(directory/'frontier.npy'),result)
    mx.save(str(directory/'prototype_keys.npy'),unique_keys);mx.save(str(directory/'velocities.npy'),velocity);mx.save(str(directory/'prototype_valid.npy'),good)
    certificate={'passed':True,'parent_checkpoint':str(parent/'checkpoint.json'),'parent_sha256':sha(parent/'checkpoint.json'),
      'source_sha256':digest(source),'result_sha256':digest(result),'regions_checked':source.shape[0],
      'regions_with_strict_descent':removed,'remaining_regions':result.shape[0],'reference_rejection_checks':accepted_groups,
      'velocity_denominator':1024,'solver_role':'Candidate generation only; exact integer certificates determine acceptance.',
      'justification':__import__('descent_coefficients_gpu').__doc__,'cpu_numerical_fallback':False,'global_optimality_proved':False}
    paths=[ROOT/'descent_coefficients_gpu.py',ROOT/'descent_witness_search_gpu.py',ROOT/'certified_precision30_gpu.py']
    certificate['code_file_sha256']={str(p):sha(p) for p in paths}
    certificate['witness_file_sha256']={str(directory/name):sha(directory/name) for name in ['prototype_keys.npy','velocities.npy','prototype_valid.npy']}
    for name in ['descent_coefficients_gpu.py','descent_witness_search_gpu.py','certified_precision30_gpu.py']:
        shutil.copyfile(ROOT/name,directory/('snapshot_'+name))
    (directory/'descent_certificate.json').write_text(json.dumps(certificate,indent=2))
    cp=dict(old);cp.update({'format':'strict-descent-reduction-v1','parent_directory':str(parent),
      'unresolved_active_boxes':result.shape[0],'unresolved_precision_boxes':0,
      'frontier_files':[str(directory/'frontier.npy')] if result.shape[0] else [],'terminal_files':[],
      'new_strict_descent_regions_removed':removed,'phase':'strict_descent_reduction',
      'offline_replay_passed':False,'complete_below_target_exclusion':False,
      'all_minimum_candidate_regions_excluded':result.shape[0]==0,'certified_lower_bound':None,
      'new_boxes_evaluated':0,'new_interval_exclusions':0,'new_splits':0})
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    (directory/'online_witness_check.json').write_text(json.dumps({'passed':True,'independent_reference_witness_checks_passed':True,
      'regions_removed':removed,'frontier_boxes':result.shape[0],'frontier_profiles':old['frontier_profiles'],'saved_splits_checked':0,
      'cpu_numerical_fallback':False,'global_optimality_proved':False},indent=2))
    print(json.dumps({k:v for k,v in certificate.items() if k!='justification'},indent=2),flush=True)
    print('DESCENT_DIRECTORY',directory,flush=True)
    return directory

def replay(directory,publish=True):
    directory=Path(directory).resolve();cp=json.loads((directory/'checkpoint.json').read_text())
    cert=json.loads((directory/'descent_certificate.json').read_text())
    assert sha(cert['parent_checkpoint'])==cert['parent_sha256']
    for p,h in {**cert['code_file_sha256'],**cert['witness_file_sha256']}.items():assert sha(p)==h
    parent=json.loads(Path(cert['parent_checkpoint']).read_text())
    source=load_rows(parent['frontier_files']+parent['terminal_files']);assert digest(source)==cert['source_sha256']
    keys=mx.load(str(directory/'prototype_keys.npy'));velocity=mx.load(str(directory/'velocities.npy'));good=mx.load(str(directory/'prototype_valid.npy'))
    inputs=model_inputs(cp['target_numerator'],cp['target_denominator']);kept=[];removed=0
    for first in range(0,source.shape[0],4096):
        rows=source[first:first+4096];match=(rows[:,0]*8+rows[:,1])[:,None]==keys[None,:]
        assert mx.all(mx.any(match,axis=1)).item();idx=mx.argmax(match,axis=1)
        c,b,n,axes,available=coefficients(rows,inputs);accepted=mx.zeros((rows.shape[0],),dtype=mx.bool_)
        for j in range(4):
            v=velocity[idx,j];ok=good[idx,j]&strict_certificate(c,b,n,available,v)&~accepted
            chosen=filter_rows(rows,ok)
            if chosen is not None:
                ix=filter_rows(mx.arange(rows.shape[0])[:,None],ok)[:,0]
                rc,rb,rn,ra,rv=coefficients(chosen,inputs,axes=axes[ix],reference=True)
                assert mx.all(strict_certificate(rc,rb,rn,rv,v[ix])).item()
            accepted|=ok
        alive=filter_rows(rows,~accepted);removed+=rows.shape[0]-(0 if alive is None else alive.shape[0])
        if alive is not None:kept.append(alive)
    result=mx.concatenate(kept,axis=0) if kept else mx.zeros((0,23),dtype=mx.int32)
    assert digest(result)==cert['result_sha256'] and removed==cert['regions_with_strict_descent']
    stored=load_rows(cp['frontier_files']);assert stored.shape==result.shape and mx.all(stored==result).item()
    audit={'passed':True,'rational_witness_replay_passed':True,'regions_removed':removed,
      'frontier_boxes':result.shape[0],'frontier_profiles':cp['frontier_profiles'],'saved_splits_checked':0,
      'cpu_numerical_fallback':False,'global_optimality_proved':False}
    (directory/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
    cp['offline_replay_passed']=True;cp['complete_below_target_exclusion']=cp['all_minimum_candidate_regions_excluded']
    if cp['complete_below_target_exclusion']:cp['certified_lower_bound']=f"s_min > {cp['target_numerator']}/{cp['target_denominator']} under documented coverage premises"
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    if publish:(ROOT/'certified_spatial/latest_run.json').write_text(json.dumps({'directory':str(directory)},indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--output-base');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    else:reduce(a.parent,a.output_base)
