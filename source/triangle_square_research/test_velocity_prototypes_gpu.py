"""Bounded solver regression, using local fixture data while WD is unavailable."""
import json
import mlx.core as mx
from n3_branches_gpu import ROOT
from certified_precision30_gpu import HEADER,model_inputs
from certified_model_tests import SOURCE as FIXTURE
from descent_witness_search_gpu import candidates
from descent_coefficients_gpu import coefficients,strict_certificate
from wd_proof_executor import ram_staged_load
mx.set_default_device(mx.gpu);mx.load=ram_staged_load

def test():
    known=json.loads((ROOT/'contact_catalog/critical_family.json').read_text())['known_pattern']['requirement_id']
    k=mx.fast.metal_kernel(name='velocity_prototype_fixture',input_names=['delta','known_id','amount'],output_names=['output','flags'],source=FIXTURE,header=HEADER)
    root,_=k(inputs=[mx.array([0],dtype=mx.int32),mx.array([known],dtype=mx.int32),mx.array([1],dtype=mx.uint32)],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(1,23),(1,)],output_dtypes=[mx.int32,mx.uint32])
    inputs=model_inputs(151,100);v=root[:,3:].reshape(1,10,2)
    trip=inputs[0][inputs[1]];wi=mx.argmax(mx.all(trip==mx.array([1,4,10],dtype=mx.uint32),axis=1)).astype(mx.int32)
    fakeid=wi*50653+mx.array(9*1369+24*37+24,dtype=mx.int32)
    center=-v[:,mx.array([2,3,4,5,0,1]),::-1]
    newvars=mx.concatenate([center,v[:,mx.array([7,8,6])],v[:,9:10]],axis=1).reshape(1,20)
    fake=mx.concatenate([mx.stack([fakeid,mx.array(3),mx.array(0)])[None,:],newvars],axis=1).astype(mx.int32)
    states=mx.concatenate([root,fake]);vel,good=candidates(states,inputs)
    mx.eval(vel,good)
    assert not mx.any(good[0]).item(),'Singular optimum must remain'
    assert mx.any(good[1]).item(),'The known descent prefix must get a witness'
    c,b,n,axes,available=coefficients(fake,inputs,reference=True)
    checked=0
    for j in range(4):
        if good[1,j].item():
            assert strict_certificate(c,b,n,available,vel[1:2,j])[0].item();checked+=1
    result={'passed':True,'genuine_singular_optimum_retained':True,'generated_rational_witnesses_independently_checked':checked,'cpu_numerical_fallback':False,'global_optimality_proved':False}
    (ROOT/'certified_spatial/velocity_prototype_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':test()
