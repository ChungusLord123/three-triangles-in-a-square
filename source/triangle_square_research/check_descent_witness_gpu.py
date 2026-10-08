"""Check a strict descent witness for an incomplete support profile and retain
the genuine singular full-support minimum. All coordinates/certificates on GPU.
"""
import json
import mlx.core as mx
from n3_branches_gpu import ROOT
from certified_precision30_gpu import Q,HEADER,model_inputs
from certified_model_tests import SOURCE as FIXTURE
from descent_coefficients_gpu import coefficients,strict_certificate
mx.set_default_device(mx.gpu)

def check():
    known=json.loads((ROOT/'contact_catalog/critical_family.json').read_text())['known_pattern']['requirement_id']
    k=mx.fast.metal_kernel(name='descent_test_exact_fixture',input_names=['delta','known_id','amount'],output_names=['output','flags'],source=FIXTURE,header=HEADER)
    root,flags=k(inputs=[mx.array([0],dtype=mx.int32),mx.array([known],dtype=mx.int32),mx.array([1],dtype=mx.uint32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(1,23),(1,)],output_dtypes=[mx.int32,mx.uint32])
    inputs=model_inputs(151,100)
    raw=coefficients(root,inputs);ref=coefficients(root,inputs,axes=raw[3],reference=True)
    original=mx.array([[563,563,1434,-2048,2150,1434,0,-4096,-4096]],dtype=mx.int32)
    full_test=strict_certificate(raw[0],raw[1],raw[2],raw[4],original)
    full_ref=strict_certificate(ref[0],ref[1],ref[2],ref[4],original)
    # Rotate by pi and permute [old1,old2,old0], then omit one base-wall
    # requirement from each base triangle. This is a requirement prefix,
    # not the actual full profile of the optimum; a strict descent is allowed.
    wall_triplets=inputs[0][inputs[1]]
    match=mx.all(wall_triplets==mx.array([1,4,10],dtype=mx.uint32)[None,:],axis=1)
    wi=mx.argmax(match).astype(mx.int32)
    assert mx.sum(match.astype(mx.int32)).item()==1
    fake_id=wi*50653+mx.array(9*1369+24*37+24,dtype=mx.int32)
    old=root[:,3:].reshape(1,10,2)
    order=mx.array([2,3,4,5,0,1],dtype=mx.int32)
    centers=-old[:,order,::-1]
    angles=old[:,mx.array([7,8,6],dtype=mx.int32)]
    newvars=mx.concatenate([centers,angles,old[:,9:10]],axis=1).reshape(1,20)
    fake=mx.concatenate([mx.stack([fake_id,mx.array(3),mx.array(0)])[None,:],newvars],axis=1).astype(mx.int32)
    raw2=coefficients(fake,inputs);ref2=coefficients(fake,inputs,axes=raw2[3],reference=True)
    v=mx.array([[-1434,2048,-2150,-1434,-563,-563,-4096,-4096,0]],dtype=mx.int32)
    ok=strict_certificate(raw2[0],raw2[1],raw2[2],raw2[4],v)
    rok=strict_certificate(ref2[0],ref2[1],ref2[2],ref2[4],v)
    mx.eval(full_test,full_ref,ok,rok,flags)
    assert not full_test[0].item() and not full_ref[0].item()
    assert ok[0].item() and rok[0].item(),{'available':raw2[4].tolist(),'axes':raw2[3].tolist(),'rows':raw2[2].tolist()}
    result={'passed':True,'genuine_singular_full_profile_retained':True,'strict_descent_prefix_witness_checked':True,
      'prefix_requirement_id':fake_id.item(),'velocity_numerators':v.tolist(),'velocity_denominator':1024,
      'independent_reference_certificate_passed':True,'cpu_numerical_fallback':False,'global_optimality_proved':False}
    (ROOT/'certified_spatial/descent_witness_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':check()
