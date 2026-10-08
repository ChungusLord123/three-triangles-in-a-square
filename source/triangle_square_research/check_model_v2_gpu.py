"""GPU calibration and endpoint-enclosure checks for the stronger model."""
import json
import mlx.core as mx
from n3_branches_gpu import ROOT
from certified_intervals_gpu import Q,HEADER
from certified_reference_gpu import reference_header
from certified_model_gpu import model_inputs,evaluate
from certified_model_v2_gpu import build_kernel,ROTATION_HELPERS
from certified_model_tests import SOURCE as FIXTURE_SOURCE
mx.set_default_device(mx.gpu)

RANGE_SOURCE=r'''
uint i=thread_position_in_grid.x;if(i>=amount[0])return;
bool unsafe=false;CI t={limits[2*i],limits[2*i+1]},c,s,cl,sl,ch,sh;
rational_rotation(t,c,s,unsafe);rational_rotation(point(t.lo),cl,sl,unsafe);rational_rotation(point(t.hi),ch,sh,unsafe);
if(i&1u){c=negative(c);s=negative(s);cl=negative(cl);sl=negative(sl);ch=negative(ch);sh=negative(sh);}
CP a=(i&2u)?CP{point(0),point(CQ)}:CP{point(CQ),point(0)};
CP b=bounded_rotate(c,s,cl,sl,ch,sh,a,CQ,unsafe);
output[4*i]=b.x.lo;output[4*i+1]=b.x.hi;output[4*i+2]=b.y.lo;output[4*i+3]=b.y.hi;
flags[i]=(uint)unsafe;
'''
POINT_SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0]*9u)return;
uint i=row/9u,j=row%9u;bool unsafe=false;
long lo=limits[2*i],hi=limits[2*i+1],t=lo+(hi-lo)*j/8;
CI c,s;rational_rotation(point(t),c,s,unsafe);if(i&1u){c=negative(c);s=negative(s);}
CP a=(i&2u)?CP{point(0),point(CQ)}:CP{point(CQ),point(0)};CP b=rotate(c,s,a,unsafe);
output[4*row]=b.x.lo;output[4*row+1]=b.x.hi;output[4*row+2]=b.y.lo;output[4*row+3]=b.y.hi;
flags[row]=(uint)unsafe;
'''

def checks(count=16000):
    mx.random.seed(11417)
    limits=mx.sort(mx.random.randint(-Q,Q+1,shape=(count,2),dtype=mx.int32),axis=1)
    rk=mx.fast.metal_kernel(name='v2_rotation_range_check',input_names=['limits','amount'],output_names=['output','flags'],source=RANGE_SOURCE,header=HEADER+ROTATION_HELPERS)
    pk=mx.fast.metal_kernel(name='v2_rotation_reference_points',input_names=['limits','amount'],output_names=['output','flags'],source=POINT_SOURCE,header=reference_header())
    bound,flags=rk(inputs=[limits,mx.array([count],dtype=mx.uint32)],grid=(count,1,1),threadgroup=(128,1,1),output_shapes=[(count,4),(count,)],output_dtypes=[mx.int64,mx.uint32])
    points,pflags=pk(inputs=[limits,mx.array([count],dtype=mx.uint32)],grid=(count*9,1,1),threadgroup=(128,1,1),output_shapes=[(count*9,4),(count*9,)],output_dtypes=[mx.int64,mx.uint32])
    points=points.reshape(count,9,4)
    contain=(bound[:,None,0]<=points[:,:,0])&(bound[:,None,1]>=points[:,:,1])&(bound[:,None,2]<=points[:,:,2])&(bound[:,None,3]>=points[:,:,3])
    mx.eval(contain,flags,pflags)
    assert mx.all(contain).item() and not mx.any(flags).item() and not mx.any(pflags).item()
    known=json.loads((ROOT/'contact_catalog/critical_family.json').read_text())['known_pattern']['requirement_id']
    delta=mx.arange(-2,3,dtype=mx.int32)*(Q//32)
    fixture=mx.fast.metal_kernel(name='v2_root_calibration',input_names=['delta','known_id','amount'],output_names=['output','flags'],source=FIXTURE_SOURCE,header=HEADER)
    roots,flags=fixture(inputs=[delta,mx.array([known],dtype=mx.int32),mx.array([5],dtype=mx.uint32)],grid=(5,1,1),threadgroup=(1,1,1),output_shapes=[(5,23),(5,)],output_dtypes=[mx.int32,mx.uint32])
    margin=mx.array(1024,dtype=mx.int32)
    states=mx.concatenate([roots,roots.at[:,3::2].add(-margin).at[:,4::2].add(margin)])
    inputs=model_inputs(151,100)
    _,r,_=evaluate(states,inputs,kernel=build_kernel(),allow_sos=False)
    _,rr,_=evaluate(states,inputs,kernel=build_kernel(reference=True),allow_sos=False)
    mx.eval(r,rr,flags);assert mx.all(r==0).item() and mx.all(rr==0).item() and not mx.any(flags).item()
    result={'passed':True,'random_rotation_intervals':count,'reference_points_per_interval':9,
      'sampled_reference_point_enclosures_preserved':True,'singular_and_rotated_feasible_roots_preserved':10,
      'scope':'Calibration and regression checks; enclosure justification is the documented derivative-sign argument, not sampling.',
      'cpu_numerical_fallback':False,'global_optimality_proved':False}
    (ROOT/'certified_spatial/model_v2_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':checks()
