"""Independent checks of initial domains, shell overlap and precision lifts.

Original routines are used only as root-data producers for comparison. The
admissibility predicates use the new standalone arithmetic/model, and all
range/coordinate calculations execute on GPU.
"""
import json,hashlib,time
from pathlib import Path
import mlx.core as mx
from independent_geometry_gpu import ROOT,Verifier,load_file
from certified_intervals_gpu import constants as original_constant_producer
from certified_spatial_search import root_states as original_root_producer

CHECK=r'''
bool radicals=claimed[0]==roots[0]&&claimed[1]==roots[1]&&claimed[2]==roots[2]&&claimed[3]==roots[3];
long n=fraction[0]*UNIT,d=fraction[1];
out[0]=radicals;
out[1]=claimed[5]==floor_div(n,d)&&claimed[6]==ceil_div(n,d);
out[2]=claimed[7]==roots[6]&&claimed[8]==roots[7]&&claimed[9]==0;
bool range=claimed[4]>0&&claimed[4]<2*UNIT;
ulong a=ulong(claimed[4]);Wide fourth=full_product(a*a,a*a);
Wide lhs={(fourth.upper<<4)|(fourth.lower>>60),fourth.lower<<4};
Wide rhs={27UL<<(4*BITS-64),0};
out[3]=range&&at_most(lhs,rhs);
out[4]=claimed[6]>claimed[4];
'''
FRACTION=r'''
out[0]=floor_div(fraction[0]*UNIT,fraction[1]);out[1]=ceil_div(fraction[0]*UNIT,fraction[1]);
'''

def choose(a,mask):
    count=mx.sum(mask.astype(mx.int32)).item()
    if count==0:return a[:0]
    positions=mx.argsort(mx.where(mask,mx.arange(mask.size),2147483647))[:count]
    return a[positions]
def digest(a):
    import numpy as np
    mx.eval(a);return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
def target_bounds(v,num,den):
    k=mx.fast.metal_kernel(name=f'independent_seed_fraction_{v.bits}',input_names=['fraction'],output_names=['out'],header=v.header,source=FRACTION)
    return k(inputs=[mx.array([num,den],dtype=mx.int64)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(2,)],output_dtypes=[mx.int64])[0]
def independent_roots(ids,ordinals,lower,upper,bits):
    unit=mx.array(1,dtype=mx.int64)<<mx.array(bits,dtype=mx.int64);half_value=(upper+1)>>1
    coordinates=mx.broadcast_to(mx.stack([-half_value,half_value])[None,:],(6,2))
    turns=mx.broadcast_to(mx.stack([-unit,unit])[None,:],(3,2))
    sides=mx.stack([lower,upper])[None,:]
    bounds=mx.concatenate([coordinates,turns,sides]).astype(mx.int32).reshape(20)
    metadata=mx.stack([ids[ordinals>>3].astype(mx.int32),(ordinals&7).astype(mx.int32),mx.zeros_like(ordinals).astype(mx.int32)],1)
    return mx.concatenate([metadata,mx.broadcast_to(bounds[None,:],(ordinals.size,20))],1)

def main():
    start=time.monotonic();dest=ROOT/'proof_n3/independent_verification'
    tree=ROOT/'certified_spatial/run_14783_10000_a8bd59da';cp=json.loads((tree/'checkpoint.json').read_text())
    v24,v30=Verifier(24),Verifier(30)
    declared=mx.array(json.loads((tree/'constants.json').read_text()),dtype=mx.int64)
    produced=original_constant_producer(cp['target_numerator'],cp['target_denominator']).astype(mx.int64)
    assert mx.all(declared==produced).item()
    kernel=mx.fast.metal_kernel(name='independent_legacy_domain_constants',input_names=['claimed','roots','fraction'],output_names=['out'],header=v24.header,source=CHECK)
    out=kernel(inputs=[declared,v24.roots,mx.array([cp['target_numerator'],cp['target_denominator']],dtype=mx.int64)],
        grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(5,)],output_dtypes=[mx.int32])[0]
    assert mx.all(out==1).item()
    ids=load_file(ROOT/'certified_spatial/spatial_candidates.npy').astype(mx.int32)
    for first in range(0,ids.size*8,65536):
        ordinals=mx.arange(first,min(first+65536,ids.size*8),dtype=mx.uint32)
        expected=independent_roots(ids,ordinals,declared[4],declared[6],24)
        actual=original_root_producer(ids,ordinals,produced.astype(mx.int32))
        assert actual.shape==expected.shape and mx.all(actual==expected).item()
    chain=json.loads((ROOT/'certified_spatial/replay_receipts/proof_chain_integrity.json').read_text())['nodes']
    shells=0;precision_rows=precision_retained=0;boundary_rows=0
    for node in chain:
        directory=Path(node['directory']);c=json.loads((directory/'checkpoint.json').read_text())
        if (directory/'basis.json').exists():
            basis=json.loads((directory/'basis.json').read_text())
            if basis.get('kind')=='side_shell':
                parent=json.loads(Path(basis['parent_checkpoint']).read_text());v=v30 if c['fixed_point_bits']==30 else v24
                old=target_bounds(v,parent['target_numerator'],parent['target_denominator'])
                new=target_bounds(v,c['target_numerator'],c['target_denominator'])
                assert old[0].item()==basis['side_lower_fixed'] and (new[1]>old[0]).item()
                shells+=1
        if c.get('format')=='precision-upgrade-v1':
            certificate=json.loads((directory/'precision_upgrade.json').read_text())
            parent=json.loads(Path(certificate['parent_checkpoint']).read_text())
            raw=mx.concatenate([load_file(p) for p in parent['frontier_files']+parent['terminal_files']])
            ceiling=target_bounds(v30,c['target_numerator'],c['target_denominator'])[1]
            lifted=[];retained=0
            for first in range(0,raw.shape[0],32768):
                old=raw[first:first+32768];bounds=old[:,3:].astype(mx.int64)<<6
                assert mx.all((bounds>=-2147483648)&(bounds<=2147483647)).item()
                bounds=bounds.at[:,19].add(mx.minimum(bounds[:,19],ceiling)-bounds[:,19])
                new=mx.concatenate([old[:,:3],bounds.astype(mx.int32)],axis=1)
                keep=new[:,21]<=new[:,22]
                valid=choose(new,keep);retained+=valid.shape[0]
                if valid.shape[0]:lifted.append(valid)
            result=mx.concatenate(lifted);stored=mx.concatenate([load_file(p) for p in c['frontier_files']])
            assert result.shape==stored.shape and mx.all(result==stored).item()
            assert digest(result)==certificate['retained_sha256']
            precision_rows=raw.shape[0];precision_retained=retained
        if c.get('format')=='exact-boundary-seed-v1':
            seed=json.loads((directory/'boundary_seed_certificate.json').read_text());parent=json.loads(Path(seed['parent_checkpoint']).read_text())
            parent_target=target_bounds(v30,parent['target_numerator'],parent['target_denominator'])
            assert parent_target[0].item()==seed['side_lower_fixed'] and v30.roots[7].item()==seed['side_upper_fixed']
            lookup=load_file(ROOT/'certified_spatial/rotation_critical_lookup.npy')
            families=load_file(directory/'covered_wall_families.npy')
            remaining=choose(ids,(lookup[ids]>0)&~families[ids//50653])
            stored=load_file(directory/'frontier.npy')
            assert stored.shape[0]==remaining.size*8
            for first in range(0,stored.shape[0],65536):
                ordinals=mx.arange(first,min(first+65536,stored.shape[0]),dtype=mx.uint32)
                expected=independent_roots(remaining,ordinals,parent_target[0],v30.roots[7],30)
                assert mx.all(expected==stored[first:first+expected.shape[0]]).item()
            boundary_rows=stored.shape[0]
    result={'passed':True,'initial_continuous_roots_checked':ids.size*8,
        'all_initial_root_bounds_and_chart_labels_match_independent_generator':True,'area_lower_bound_checked_by_exact_128bit_quartic_inequality':True,
        'overlapping_rational_shells_checked':shells,'precision_upgrade_input_regions':precision_rows,
        'precision_upgrade_retained_regions':precision_retained,'all_old_terminals_reconsidered_and_exact_scaling_checked':True,
        'boundary_continuous_roots_checked':boundary_rows,'exact_boundary_upper_contains_s0':True,
        'cpu_numerical_fallback':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-start}
    (dest/'continuous_domain_and_precision.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
