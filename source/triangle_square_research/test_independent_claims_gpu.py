"""Evidence adapter test. Only this adapter reads the original producer code.

The verifier remains a standalone module and consumes rows/claimed bounds as
certificate data. The producer's archived shader reconstructs those claims.
"""
import json,time
from pathlib import Path
import mlx.core as mx
from independent_geometry_gpu import Verifier,ROOT,load_file
from certified_model_gpu import model_inputs,evaluate
from certified_precision30_gpu import model_inputs as inputs30
from certified_spatial_search import root_states

def archived_kernel(path,name):
    text=Path(path).read_text();p=text.index('uint row=thread_position_in_grid.x')
    return mx.fast.metal_kernel(name=name,header=text[:p],source=text[p:],
        input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
        output_names=['updated','reasons','split_variable'])
def main():
    start=time.monotonic();dest=ROOT/'proof_n3/independent_verification'
    directory=ROOT/'certified_spatial/run_14783_10000_a8bd59da';cp=json.loads((directory/'checkpoint.json').read_text())
    records=cp['records'];branches=[r for r in records if r['kind']=='branch']
    selected=records[:12]+branches[:20]+branches[-12:]
    inputs=model_inputs(cp['target_numerator'],cp['target_denominator']);v=Verifier(24)
    kernel=archived_kernel(directory/'primary_model.metal','independent_claim_producer_24')
    profiles=load_file(ROOT/'certified_spatial/spatial_candidates.npy')
    checked=failed=physical=contractions=0
    for index,r in enumerate(selected):
        if r['kind']=='initial':
            first,count=r['initial_range'];states=root_states(profiles,mx.arange(first,first+count,dtype=mx.uint32),inputs[3])
        else:states=load_file(r['input_file'])
        claimed,reasons,_=evaluate(states,inputs,kernel=kernel,rounds=4);mx.eval(claimed,reasons)
        accepted,refined,verdict=v.verify_transition(states,claimed,reasons,12)
        n=mx.sum((~accepted).astype(mx.int32)).item();checked+=states.shape[0];failed+=n
        physical+=mx.sum(((reasons!=0)&(reasons!=9)).astype(mx.int32)).item()
        contractions+=mx.sum((reasons==0).astype(mx.int32)).item()
        if n:
            for name,a in [('states',states),('claimed',claimed),('reasons',reasons),('refined',refined),('verdict',verdict),('accepted',accepted)]:
                mx.save(str(dest/f'claim_failure_{index}_{name}.npy'),a)
            print('Unverified claim batch',index,n,flush=True);break
    last=Path('/Volumes/SquarePackingProof/SquarePacking/runs/compact_a4156676');lastcp=json.loads((last/'checkpoint.json').read_text())
    parent=json.loads((Path(lastcp['parent_directory'])/'checkpoint.json').read_text());states=load_file(parent['frontier_files'][0])
    i30=inputs30(lastcp['target_numerator'],lastcp['target_denominator']);k30=archived_kernel(last/'primary_model.metal','independent_claim_producer_30')
    claimed,reasons,_=evaluate(states,i30,kernel=k30,rounds=4);accepted,refined,verdict=Verifier(30).verify_transition(states,claimed,reasons,16)
    f30=mx.sum((~accepted).astype(mx.int32)).item()
    if f30:
        for name,a in [('states',states),('claimed',claimed),('reasons',reasons),('refined',refined),('verdict',verdict)]:mx.save(str(dest/f'claim_failure_30_{name}.npy'),a)
    result={'passed':failed==0 and f30==0,'legacy_claims_checked':checked,'legacy_physical_exclusions':physical,
        'legacy_successful_contractions':contractions,'legacy_unverified_claims':failed,
        'final_precision30_claims_checked':states.shape[0],'final_unverified_claims':f30,
        'cpu_numerical_fallback':False,'elapsed_seconds':time.monotonic()-start}
    (dest/'claim_adapter_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    assert result['passed']

if __name__=='__main__':main()
