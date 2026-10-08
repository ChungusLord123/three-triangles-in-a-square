"""Reconstruct the frozen legacy producer; verify all claims independently.

Producer code is used only to reconstruct the committed evidence stream. The
standalone verifier receives states, claimed bounds and claimed decisions;
it imports none of that code. Existing receipts are not overwritten.
"""
import json,time,hashlib
from pathlib import Path
import mlx.core as mx
import replay_legacy_tree_gpu as producer
from independent_geometry_gpu import Verifier,ROOT,source_commitment

def main():
    start=time.monotonic();dest=ROOT/'proof_n3/independent_verification';dest.mkdir(exist_ok=True)
    v=Verifier(24);commitment=source_commitment();primary=None
    original_kernel=producer.kernel;original_evaluate=producer.evaluate;original_write=Path.write_text
    counts={'rows':0,'physical_claims':0,'contraction_claims':0,'retained_claims':0};last=time.monotonic()
    def kernel(path,label):
        nonlocal primary
        result=original_kernel(path,label)
        if label=='legacy_fresh_primary':primary=result
        return result
    def observe(states,*args,**kwargs):
        nonlocal last
        result=original_evaluate(states,*args,**kwargs)
        if kwargs.get('kernel') is primary:
            claimed,reasons,_=result;mx.eval(claimed,reasons)
            accepted,refined,verdict=v.verify_transition(states,claimed,reasons,12)
            if not mx.all(accepted).item():
                for name,a in [('states',states),('claimed',claimed),('reasons',reasons),('refined',refined),('verdict',verdict),('accepted',accepted)]:
                    mx.save(str(dest/f'legacy_unverified_{name}.npy'),a)
                raise RuntimeError('Independent verifier did not establish every legacy claim; no verification success is recorded.')
            counts['rows']+=states.shape[0]
            counts['physical_claims']+=mx.sum(((reasons!=0)&(reasons!=9)).astype(mx.int32)).item()
            counts['contraction_claims']+=mx.sum((reasons==0).astype(mx.int32)).item()
            counts['retained_claims']+=mx.sum((reasons==9).astype(mx.int32)).item()
            if time.monotonic()-last>10:
                print('Independent geometry verified legacy rows',counts['rows'],flush=True);last=time.monotonic()
        return result
    old_receipt=ROOT/'certified_spatial/replay_receipts/legacy_complete_reconstruction.json'
    def write(path,text,*args,**kwargs):
        if path==old_receipt:return original_write(dest/'legacy_producer_reconstruction.json',text,*args,**kwargs)
        return original_write(path,text,*args,**kwargs)
    producer.kernel=kernel;producer.evaluate=observe;Path.write_text=write
    try:producer.replay()
    finally:Path.write_text=original_write
    assert source_commitment()==commitment
    previous=json.loads(old_receipt.read_text());assert counts['rows']==previous['boxes_reconstructed']
    assert counts['physical_claims']==previous['physical_exclusions_freshly_rechecked']
    result={'passed':True,**counts,'every_physical_rejection_independently_verified':True,
        'every_successful_contraction_independently_verified':True,'standalone_model_imports_no_search_geometry':True,
        'verifier_sha256':commitment,'producer_reconstruction_sha256':hashlib.sha256((dest/'legacy_producer_reconstruction.json').read_bytes()).hexdigest(),
        'cpu_numerical_fallback':False,'elapsed_seconds':time.monotonic()-start}
    (dest/'legacy_independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
