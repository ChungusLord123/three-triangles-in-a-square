"""GPU exact-integer regression certificates and feasible-root calibration."""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT
from certified_precision30_gpu import Q,HEADER,REFERENCE_HEADER,MODEL_HEADER,constants,model_inputs,build_kernel
from certified_intervals_gpu import TEST_SOURCE
from certified_model_tests import SOURCE as FIXTURE
from certified_model_gpu import evaluate
mx.set_default_device(mx.gpu)
D=torch.device('mps')

def checks(n=12000):
    mx.random.seed(301224)
    a=mx.sort(mx.random.randint(-Q,Q,shape=(n,2),dtype=mx.int32),axis=1)
    b=mx.sort(mx.random.randint(Q//2,Q,shape=(n,2),dtype=mx.int32),axis=1)
    b=mx.where((mx.arange(n)%2)[:,None]==0,b,-b[:,::-1])
    angles=mx.sort(mx.random.randint(-Q,Q+1,shape=(n,2),dtype=mx.int32),axis=1)
    raw=mx.concatenate([a,b],axis=1)
    def primitive(header,name):
        k=mx.fast.metal_kernel(name=name,input_names=['inputs','angles','amount'],output_names=['out','flags'],source=TEST_SOURCE,header=header)
        return k(inputs=[raw,angles,mx.array([n],dtype=mx.uint32)],grid=(n,1,1),threadgroup=(128,1,1),
          output_shapes=[(n,14),(n,)],output_dtypes=[mx.int64,mx.uint32])
    p,f=primitive(HEADER,'precision30_arithmetic_primary');r,rf=primitive(REFERENCE_HEADER,'precision30_arithmetic_reference')
    mx.eval(p,r,f,rf,raw)
    assert not mx.any(f).item() and not mx.any(rf).item()
    assert mx.all(p[:,:10]==r[:,:10]).item()
    assert mx.all((p[:,10]<=r[:,10])&(p[:,11]>=r[:,11])&(p[:,12]<=r[:,12])&(p[:,13]>=r[:,13])).item()
    # Independent MPS multiplication/floor/division inequalities, no gather or
    # MPS integer division; intermediates remain inside signed int64.
    inp=torch.tensor(raw.tolist(),device=D,dtype=torch.int64);out=torch.tensor(p.tolist(),device=D,dtype=torch.int64)
    al,ah,bl,bh=[inp[:,j] for j in range(4)]
    assert torch.equal(out[:,0],al+bl) and torch.equal(out[:,1],ah+bh)
    assert torch.equal(out[:,2],al-bh) and torch.equal(out[:,3],ah-bl)
    prod=torch.stack([al*bl,al*bh,ah*bl,ah*bh],1)
    mn=prod.amin(1);mxp=prod.amax(1)
    assert ((out[:,4]*Q<=mn)&((out[:,4]+1)*Q>mn)).all().item()
    assert ((out[:,5]*Q>=mxp)&((out[:,5]-1)*Q<mxp)).all().item()
    numer=torch.stack([al,al,ah,ah],1)*Q;den=torch.stack([bl,bh,bl,bh],1)
    numer=torch.where(den<0,-numer,numer);den=den.abs()
    lower=out[:,6,None];upper=out[:,7,None]
    assert (lower*den<=numer).all().item() and (upper*den>=numer).all().item()
    assert (numer<(lower+1)*den).any(1).all().item()
    assert (numer>(upper-1)*den).any(1).all().item()
    co=constants(14783977,10000000);c=torch.tensor(co.tolist(),device=D,dtype=torch.int64)
    assert ((c[0]*c[0]<=3*Q*Q)&(c[1]*c[1]>3*Q*Q)).item()
    assert ((c[2]*c[2]<=6*Q*Q)&(c[3]*c[3]>6*Q*Q)).item()
    known=json.loads((ROOT/'contact_catalog/critical_family.json').read_text())['known_pattern']['requirement_id']
    delta=mx.arange(-2,3,dtype=mx.int32)*(Q//32)
    k=mx.fast.metal_kernel(name='precision30_feasible_fixture',input_names=['delta','known_id','amount'],output_names=['output','flags'],source=FIXTURE,header=HEADER)
    roots,flags=k(inputs=[delta,mx.array([known],dtype=mx.int32),mx.array([5],dtype=mx.uint32)],grid=(5,1,1),threadgroup=(1,1,1),output_shapes=[(5,23),(5,)],output_dtypes=[mx.int32,mx.uint32])
    larger=roots.at[:,3::2].add(-1024).at[:,4::2].add(1024)
    cases=mx.concatenate([roots,larger]);inputs=model_inputs(151,100)
    _,reason,_=evaluate(cases,inputs,kernel=build_kernel(),allow_sos=False)
    _,rr,_=evaluate(cases,inputs,kernel=build_kernel(reference=True),allow_sos=False)
    mx.eval(reason,rr,flags)
    assert mx.all(reason==0).item() and mx.all(rr==0).item() and not mx.any(flags).item()
    result={'passed':True,'fixed_point_bits':30,'primitive_cases':n,'independent_MPS_integer_inequalities_passed':True,
      'reference_arithmetic_matched':True,'feasible_singular_and_rotated_root_boxes_preserved':10,
      'radical_bounds_verified':True,'cpu_numerical_fallback':False,'global_optimality_proved':False,
      'constants':co.tolist()}
    (ROOT/'certified_spatial/precision30_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':checks()
