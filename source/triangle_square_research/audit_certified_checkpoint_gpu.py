"""Audit saved bisections and frontier statistics using independent MPS integers.

This checks every saved split, file availability, root range continuity and
row conservation. It does not certify the shared geometric model or replace
the online interval-exclusion replays. A nonempty frontier is never a proof.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import json, time, hashlib
from pathlib import Path
import torch
from check_contact_catalog_mps import read_u32, D
from n3_branches_gpu import ROOT

def audit():
    start=time.monotonic()
    directory=Path(json.loads((ROOT/'certified_spatial/latest_run.json').read_text())['directory'])
    cp=json.loads((directory/'checkpoint.json').read_text())
    records=cp['records']
    next_root=0; processed=0; initial_kept=0; children_total=0; terminals_total=0
    checked_splits=0; last=time.monotonic()
    checks=torch.ones((),device=D,dtype=torch.bool)
    parents=[];dimensions=[];child_parts=[];terminal_parts=[]
    def flush():
        nonlocal checks, checked_splits, children_total, terminals_total
        if not parents:return
        parent=torch.cat(parents,dim=0);dims=torch.cat(dimensions,dim=0)
        safe_dim=dims.clamp_min(0)
        # MPS gather rounds some int32 values through float32 on this host.
        # Exact selection via pointwise where avoids that backend defect.
        low=torch.zeros(parent.shape[0],device=D,dtype=torch.int32)
        high=torch.zeros_like(low)
        for j in range(10):
            low=torch.where(safe_dim==j,parent[:,3+2*j],low)
            high=torch.where(safe_dim==j,parent[:,4+2*j],high)
        # The recorded width policy is constant within this checkpoint.
        branch=(dims>=0)&(high-low>16)&(parent[:,2]<180)
        ix=torch.nonzero(branch,as_tuple=True)[0]
        terminal_ix=torch.nonzero(~branch,as_tuple=True)[0]
        if terminal_parts:
            terminal=torch.cat(terminal_parts,dim=0);terminals_total+=terminal.shape[0]
            assert terminal.shape==parent[terminal_ix].shape
            checks &= (terminal==parent[terminal_ix]).all()
        else:checks &= terminal_ix.numel()==0
        if child_parts:
            children=torch.cat(child_parts,dim=0).reshape(-1,2,23)
            p=parent[ix];dim=dims[ix];lower=low[ix];upper=high[ix]
            assert children.shape[0]==p.shape[0]
            midpoint=lower+((upper-lower)>>1)
            index=torch.arange(p.shape[0],device=D)
            columns=torch.arange(23,device=D)[None,:]
            a=torch.where(columns==(4+2*dim)[:,None],midpoint[:,None],p)
            b=torch.where(columns==(3+2*dim)[:,None],midpoint[:,None],p)
            a=torch.where(columns==2,a+1,a);b=torch.where(columns==2,b+1,b)
            checks &= (children[:,0]==a).all() & (children[:,1]==b).all()
            checks &= ((lower<midpoint)&(midpoint<upper)).all()
            if not checks.item():
                mismatch=((children[:,0]!=a).any(1)|(children[:,1]!=b).any(1))
                first=torch.nonzero(mismatch,as_tuple=True)[0][:3]
                print('split mismatch',{ 'parent':p[first].tolist(),'dimension':dim[first].tolist(),
                      'actual_children':children[first].tolist(),'expected_left':a[first].tolist(),
                      'expected_right':b[first].tolist()},flush=True)
                raise AssertionError('Saved split does not match integer child union')
            checked_splits+=p.shape[0];children_total+=2*p.shape[0]
        else:checks &= ix.numel()==0
        parents.clear();dimensions.clear();child_parts.clear();terminal_parts.clear()
    for record in records:
        for key in ['input_file','reason_file','contracted_file','dimensions_file','children_file','terminal_file']:
            if record.get(key): assert Path(record[key]).is_file(), record[key]
        processed+=record['boxes']
        if record['kind']=='initial':
            first,count=record['initial_range']
            assert first==next_root and count==record['boxes']
            next_root+=count
            if record['contracted_file']:
                initial_kept+=read_u32(record['contracted_file']).shape[0]
            continue
        if not record['contracted_file']:
            assert not record.get('children_file') and not record.get('terminal_file')
            continue
        assert record['minimum_split_width']==16
        parents.append(read_u32(record['contracted_file']))
        dimensions.append(read_u32(record['dimensions_file']))
        if record.get('terminal_file'):
            terminal_parts.append(read_u32(record['terminal_file']))
        if record.get('children_file'):
            child_parts.append(read_u32(record['children_file']))
        if len(parents)>=64:flush()
        if time.monotonic()-last>10:
            print('Saved splits checked',checked_splits,flush=True);last=time.monotonic()
    flush();assert checks.item()
    assert processed==cp['boxes_evaluated'] and next_root==cp['initial_jobs_processed']
    assert terminals_total==cp['unresolved_precision_boxes']
    frontier=[read_u32(f) for f in cp['frontier_files']]
    pending=sum(f.shape[0] for f in frontier)
    assert pending==cp['unresolved_active_boxes']
    assert next_root+initial_kept+children_total==processed+pending+terminals_total
    state=torch.cat(frontier,dim=0)
    valid=(state[:,3::2]<=state[:,4::2]).all()
    profiles=torch.unique(state[:,0]);depth=state[:,2]
    assert valid.item()
    source_hash=hashlib.sha256((directory/'primary_model.metal').read_bytes()).hexdigest()
    assert source_hash==cp['primary_model_sha256']
    # Hash only data bytes; no geometric or arithmetic evaluation on the host.
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        ROOT/'certified_spatial/spatial_candidates.npy',ROOT/'contact_catalog/wall_masks.npy',
        ROOT/'contact_catalog/wall_triplets.npy',ROOT/'contact_catalog/wall_angles.npy']}
    result={'passed':True,'saved_splits_checked':checked_splits,'all_split_children_cover_parent':True,
      'root_ordinal_ranges_contiguous':True,'row_conservation_passed':True,
      'frontier_files_exist':True,'frontier_boxes':pending,'frontier_profiles':profiles.numel(),
      'frontier_min_depth':depth.min().item(),'frontier_max_depth':depth.max().item(),
      'shared_geometric_model_independently_verified':False,
      'check_scope':'File and bisection audit; exclusions were replayed online with independent arithmetic but the same geometric constraints.',
      'input_data_sha256':inputs,'primary_model_sha256':source_hash,
      'cpu_numerical_fallback':False,'numerical_engine':'PyTorch MPS int32',
      'checker_backend_note':'Avoids MPS gather, which rounded large int32 endpoints on this host; exact pointwise where and bit shifts are used instead.',
      'elapsed_seconds':time.monotonic()-start,'global_optimality_proved':False}
    (directory/'checkpoint_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':audit()
