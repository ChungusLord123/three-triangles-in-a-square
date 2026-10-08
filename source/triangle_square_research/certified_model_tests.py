"""Guaranteed-root calibration boxes, including the singular known minimum.
They are constructed with integer intervals, not rounded floating poses.
"""
import json
import mlx.core as mx
from certified_intervals_gpu import HEADER,Q
from certified_model_gpu import model_inputs,evaluate
from certified_reference_gpu import reference_kernel
from n3_branches_gpu import ROOT

SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
bool unsafe=false;CI one=point(CQ),d=point(delta[row]);
long r2=isqrt_exact(2*CQ*CQ),r3=isqrt_exact(3*CQ*CQ),r6=isqrt_exact(6*CQ*CQ);
CI k2=CI{r2,r2+1},k3=CI{r3,r3+1},k6=CI{r6,r6+1};
CI h=quotient(k3,point(2*CQ),unsafe),q=quotient(k6,point(4*CQ),unsafe);
CI d2=square(d,unsafe),u=times(q,quotient(ci_add(one,d2),ci_sub(one,d2),unsafe),unsafe);
CI side=ci_add(h,u),halfside=quotient(side,point(2*CQ),unsafe);
CI center=quotient(times_int(q,2),point(3*CQ),unsafe);
CI co,si;rational_rotation(d,co,si,unsafe);
CP corner=rotate(co,si,CP{center,center},unsafe);
CI alpha=quotient(ci_sub(k6,k2),point(4*CQ),unsafe),beta=quotient(ci_add(k6,k2),point(4*CQ),unsafe);
CI original=quotient(beta,ci_add(one,alpha),unsafe);
CI t0=quotient(ci_add(original,d),ci_sub(one,times(original,d,unsafe)),unsafe);
CI t1=quotient(point(-CQ/2),ci_add(one,h),unsafe);
CI t2=quotient(h,point(3*CQ/2),unsafe);
CI twoh=quotient(times_int(h,2),point(3*CQ),unsafe);
CI z[10]={ci_sub(corner.x,halfside),ci_sub(corner.y,halfside),
 ci_sub(ci_add(u,twoh),halfside),ci_sub(u,halfside),
 ci_sub(u,halfside),ci_sub(ci_add(u,twoh),halfside),t0,t1,t2,side};
output[23*row]=known_id[0];output[23*row+1]=1;output[23*row+2]=0;
for(uint j=0;j<10;j++){output[23*row+3+2*j]=(int)z[j].lo;output[23*row+4+2*j]=(int)z[j].hi;}
flags[row]=(uint)unsafe;
'''

def calibration_tests():
 known=json.loads((ROOT/'contact_catalog'/'critical_family.json').read_text())['known_pattern']['requirement_id']
 delta=mx.arange(-2,3,dtype=mx.int32)*(Q//32)
 k=mx.fast.metal_kernel(name='certified_root_calibration',input_names=['delta','known_id','amount'],output_names=['output','flags'],source=SOURCE,header=HEADER)
 roots,flags=k(inputs=[delta,mx.array([known],dtype=mx.int32),mx.array([5],dtype=mx.uint32)],grid=(5,1,1),threadgroup=(1,1,1),output_shapes=[(5,23),(5,)],output_dtypes=[mx.int32,mx.uint32])
 # Also calibrate boxes enlarged around those exact feasible roots.
 margin=mx.array(1024,dtype=mx.int32)
 enlarged=roots.at[:,3::2].add(-margin).at[:,4::2].add(margin)
 states=mx.concatenate([roots,enlarged],axis=0)
 mx.eval(states,flags);assert not mx.any(flags).item()
 inputs=model_inputs(151,100)
 primary,reason,split=evaluate(states,inputs,allow_sos=False)
 ref,reference_reason,_=evaluate(states,inputs,kernel=reference_kernel(),allow_sos=False)
 mx.eval(primary,reason,reference_reason)
 assert mx.all(reason==0).item(),reason.tolist()
 assert mx.all(reference_reason==0).item(),reference_reason.tolist()
 low_inputs=model_inputs(14783,10000);co=low_inputs[3]
 half=(co[6].astype(mx.int64)+1)//2
 intervals=mx.concatenate([mx.broadcast_to(mx.stack([-half,half])[None,:],(6,2)),
              mx.broadcast_to(mx.array([-Q,Q])[None,:],(3,2)),mx.stack([co[4],co[6]])[None,:]],axis=0).astype(mx.int32).reshape(1,20)
 low=mx.concatenate([mx.array([[known,1,0]],dtype=mx.int32),intervals],axis=1)
 _,lr,_=evaluate(low,low_inputs,allow_sos=False)
 _,rr,_=evaluate(low,low_inputs,kernel=reference_kernel(),allow_sos=False)
 mx.eval(lr,rr)
 assert lr[0].item() not in [0,9] and rr[0].item() not in [0,9]
 result={'passed':True,'feasible_root_boxes_preserved':states.shape[0],
         'includes_singular_optimum':True,'includes_nonzero_corner_rotations':True,
         'primary_and_reference_preserve_all_roots':True,
         'known_family_below_1_4783_excluded_without_SOS_shortcut':True,
         'primary_exclusion_reason':lr[0].item(),'reference_exclusion_reason':rr[0].item(),
         'root_construction':'Exact dyadic interval enclosures using radical bounds and rational half-angle formulas',
         'cpu_numerical_fallback':False,'global_optimality_proved':False}
 (ROOT/'certified_spatial'/'model_calibration_checks.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':calibration_tests()
