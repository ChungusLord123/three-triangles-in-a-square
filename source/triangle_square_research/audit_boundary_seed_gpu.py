"""Fresh independent verification of the exact-boundary root partition."""
import json,sys,time
from pathlib import Path
import mlx.core as mx
import torch
from exact_boundary_runner_gpu import configure,analytic_wall_families,reference_wall_families,roots
from certified_precision30_gpu import model_inputs,constants,Q
from check_compact_splits_mps import transfer,D
from compact_boundary_search_gpu import sha,digest,filter_rows
from n3_branches_gpu import ROOT

def exact_quotient(t,divisor):
    # Binary long division avoids any integer-to-float lowering on MPS.
    rem=torch.zeros_like(t);out=torch.zeros_like(t)
    for bit in range(24,-1,-1):
        rem=(rem<<1)|((t>>bit)&1);hit=rem>=divisor
        rem=torch.where(hit,rem-divisor,rem);out|=hit.to(torch.int32)<<bit
    return out

def audit(directory):
    configure();start=time.monotonic();directory=Path(directory)
    cert=json.loads((directory/'boundary_seed_certificate.json').read_text())
    assert sha(cert['parent_checkpoint'])==cert['parent_sha256']
    assert sha(cert['analytic_certificate'])==cert['analytic_certificate_sha256']
    for p,h in {**cert['code_file_sha256'],**cert['input_file_sha256']}.items():assert sha(p)==h
    parent=json.loads(Path(cert['parent_checkpoint']).read_text())
    co_old=constants(parent['target_numerator'],parent['target_denominator'])
    masks,triples,angles,co=model_inputs(cert['side_upper_fixed'],Q)
    assert co_old[5].item()==cert['side_lower_fixed'] and co_old[8].item()==cert['side_upper_fixed']
    assert (2*co_old[5]>co[1]+Q).item() and (co[6]<co[0]).item()
    families=analytic_wall_families(masks,triples)
    rf=reference_wall_families(transfer(masks),transfer(triples))
    assert torch.equal(rf,torch.tensor(families.tolist(),device=D,dtype=torch.bool))
    ids=mx.load(str(ROOT/'certified_spatial/spatial_candidates.npy')).astype(mx.int32)
    lookup=mx.load(str(ROOT/'certified_spatial/rotation_critical_lookup.npy'))
    keep=(lookup[ids]>0)&~families[ids//50653]
    ti=transfer(ids);tl=transfer(lookup.astype(mx.int32))
    tk=(tl[ti.to(torch.int64)]>0)&~rf[exact_quotient(ti,50653).to(torch.int64)]
    assert torch.equal(tk,torch.tensor(keep.tolist(),device=D,dtype=torch.bool))
    selected=filter_rows(ids[:,None],keep)[:,0]
    assert selected.size==cert['remaining_profiles']
    rows=roots(selected,co,co_old[5])
    assert rows.shape[0]==cert['frontier_rows'] and digest(rows)==cert['frontier_sha256']
    assert digest(families)==cert['families_sha256']
    stored=mx.load(str(directory/'frontier.npy'))
    assert stored.shape==rows.shape and mx.all(stored==rows).item()
    tselected=ti[tk]
    high=torch.tensor(cert['side_upper_fixed'],device=D,dtype=torch.int64)
    low=torch.tensor(cert['side_lower_fixed'],device=D,dtype=torch.int64)
    half=(high+1)>>1
    domain=torch.cat([torch.stack([-half,half]).repeat(6),
        torch.tensor([-Q,Q],device=D,dtype=torch.int64).repeat(3),torch.stack([low,high])]).to(torch.int32)
    for first in range(0,rows.shape[0],32768):
        b=transfer(rows[first:first+32768]);ix=torch.arange(first,first+b.shape[0],device=D,dtype=torch.int32)
        assert torch.equal(b[:,0],tselected[(ix>>3).to(torch.int64)])
        assert torch.equal(b[:,1],ix&7) and (b[:,2]==0).all().item()
        assert torch.equal(b[:,3:],domain[None,:].expand(b.shape[0],-1))
    result={'passed':True,'parent_and_all_code_input_hashes_matched':True,
        'entire_root_partition_independently_checked_on_MPS':True,
        'MPS_profile_quotients_used_exact_binary_division':True,'all_full_circle_charts_reconstructed':True,
        'all_remaining_roots_reconstructed':True,'analytic_covered_profiles':cert['analytic_covered_profiles'],
        'remaining_profiles':selected.size,'frontier_boxes':rows.shape[0],
        'cpu_numerical_fallback':False,'global_optimality_proved':False,'elapsed_seconds':time.monotonic()-start}
    dest=ROOT/'certified_spatial/replay_receipts';dest.mkdir(exist_ok=True)
    (dest/(directory.name+'_independent_seed.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':audit(sys.argv[1])
