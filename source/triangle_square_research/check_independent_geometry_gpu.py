"""Independent verifier calibration against exact feasible/infeasible fixtures."""
import json,hashlib
from pathlib import Path
import mlx.core as mx
from independent_geometry_gpu import Verifier,ROOT

FIXTURE=r'''
bool unknown=false;
Range h=divide_small(Range{roots[0],roots[1]},2),q=divide_small(Range{roots[2],roots[3]},4),a=divide_small(Range{roots[4],roots[5]},4);
Range side=add(h,q),half_value=divide_small(side,2),center= subtract(divide_small(scale_int(q,2),3),half_value);
Range off=divide_small(scale_int(h,2),3);
Range domains[10]={center,center,subtract(add(q,off),half_value),subtract(q,half_value),subtract(q,half_value),subtract(add(q,off),half_value),
    quotient(add(q,a),add(value(UNIT),subtract(q,a)),unknown),
    negate(quotient(value(UNIT/2),add(value(UNIT),h),unknown)),
    quotient(h,value(3*UNIT/2),unknown),side};
output[0]=11823248;output[1]=1;output[2]=0;
for(uint i=0;i<10;i++){output[3+2*i]=int(domains[i].lower);output[4+2*i]=int(domains[i].upper);}
flag[0]=unknown?1u:0u;
'''

def fixture(verifier):
    k=mx.fast.metal_kernel(name=f'independent_known_fixture_{verifier.bits}',input_names=['roots'],output_names=['output','flag'],source=FIXTURE,header=verifier.header)
    state,flag=k(inputs=[verifier.roots],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(1,23),(1,)],output_dtypes=[mx.int32,mx.uint32])
    mx.eval(state,flag);assert flag[0].item()==0
    return state

def main():
    outcomes=[]
    for bits in [24,30]:
        v=Verifier(bits);root=fixture(v);refined,reject=v.contract(root,16)
        assert reject[0].item()==0,{'bits':bits,'reject':reject.tolist()}
        assert v.motion_check(root,mx.zeros((1,9),dtype=mx.int32))[0].item()==0
        # All coordinates are shifted by the GPU; this packing violates walls.
        shift=mx.array(1,dtype=mx.int32)<<mx.array(bits,dtype=mx.int32)
        wrong=root.at[:,3:5].add(shift)
        assert v.contract(wrong,4)[1][0].item()!=0
        triples=v.masks[v.triples]
        empty=mx.argmax(mx.all(triples==0,axis=1).astype(mx.int32))
        overlap=root.at[:,0].add(empty*50653-root[:,0])
        overlap=overlap.at[:,3:21].add(-overlap[:,3:21])
        assert v.contract(overlap,4)[1][0].item()!=0
        target=mx.all(triples==mx.array([1,4,10],dtype=mx.uint32)[None,:],axis=1)
        assert mx.sum(target.astype(mx.int32)).item()==1
        wi=mx.argmax(target).astype(mx.int32)
        ident=wi*50653+mx.array(9*1369+24*37+24,dtype=mx.int32)
        old=root[:,3:].reshape(1,10,2)
        centers=-old[:,mx.array([2,3,4,5,0,1]),::-1]
        angles=old[:,mx.array([7,8,6])]
        values=mx.concatenate([centers,angles,old[:,9:10]],axis=1).reshape(1,20)
        fake=mx.concatenate([mx.stack([ident,mx.array(3),mx.array(0)])[None,:],values],axis=1).astype(mx.int32)
        witness=mx.array([[-1434,2048,-2150,-1434,-563,-563,-4096,-4096,0]],dtype=mx.int32)
        assert v.motion_check(fake,witness)[0].item()==1
        assert v.motion_check(root,witness)[0].item()==0
        outcomes.append({'bits':bits,'singular_exact_construction_retained':True,
            'invalid_containment_rejected':True,'overlapping_interiors_rejected':True,
            'strict_descent_witness_verified':True,'true_minimum_not_given_strict_descent':True,'radical_enclosures':v.roots.tolist()})
    result={'passed':True,'outcomes':outcomes,'imports_search_geometry':False,
        'cpu_numerical_fallback':False,'source_sha256':hashlib.sha256((ROOT/'independent_geometry_gpu.py').read_bytes()).hexdigest()}
    dest=ROOT/'proof_n3/independent_verification';dest.mkdir(exist_ok=True)
    (dest/'calibration.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
