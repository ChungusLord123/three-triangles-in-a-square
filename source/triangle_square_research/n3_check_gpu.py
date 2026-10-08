"""Independent MPS feasibility check and GPU sanity checks of the algebra.
Float32 sanity checks are not used as proof certificates.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import json
from pathlib import Path
import torch
import mlx.core as mx
from n3_branches_gpu import ROOT,LOCAL,PI

mx.set_default_device(mx.gpu)
if not torch.backends.mps.is_available():raise RuntimeError('MPS GPU required')
device=torch.device('mps')

def check_branches():
 result=json.loads((ROOT/'results'/'n3_branches.json').read_text())
 accepted=[r for r in result['branches'] if r['feasible_found']]
 if not accepted:raise RuntimeError('No candidate layouts to check')
 pp=torch.tensor([r['poses'] for r in accepted],dtype=torch.float32,device=device)
 side=torch.tensor([r['square_side'] for r in accepted],dtype=torch.float32,device=device)
 h=torch.sqrt(torch.tensor(3.0,device=device))/2
 r=h/3
 local=torch.stack([torch.stack([torch.tensor(0.5,device=device),-r]),
                    torch.stack([torch.tensor(0.0,device=device),2*r]),
                    torch.stack([torch.tensor(-0.5,device=device),-r])])
 co=torch.cos(pp[:,:,2])[:,:,None];si=torch.sin(pp[:,:,2])[:,:,None]
 vx=co*local[None,None,:,0]-si*local[None,None,:,1]+pp[:,:,None,0]
 vy=si*local[None,None,:,0]+co*local[None,None,:,1]+pp[:,:,None,1]
 verts=torch.stack([vx,vy],dim=-1)
 left=torch.tensor([0,0,1],device=device,dtype=torch.int32)
 right=torch.tensor([1,2,2],device=device,dtype=torch.int32)
 va=verts[:,left];vb=verts[:,right]
 edges=torch.cat([torch.roll(va,-1,dims=2)-va,torch.roll(vb,-1,dims=2)-vb],dim=2)
 axes=torch.stack([-edges[:,:,:,1],edges[:,:,:,0]],dim=-1)
 axnorm=axes.square().sum(-1).sqrt()
 pa=torch.einsum('bpvd,bpad->bpav',va,axes)
 pb=torch.einsum('bpvd,bpad->bpav',vb,axes)
 gaps=torch.maximum(pb.amin(-1)-pa.amax(-1),pa.amin(-1)-pb.amax(-1))/axnorm
 minsep=gaps.amax(-1).amin(-1)
 boxclear=(side/2)[:,None,None,None]-verts.abs()
 minimum_box=boxclear.amin(dim=(1,2,3))
 valid=(minsep>0.00002)&(minimum_box>0.00002)&torch.isfinite(pp).all(dim=(1,2))
 # Actual edge lengths, independent of the SAT implementation.
 edge_lengths=(torch.roll(verts,-1,dims=2)-verts).square().sum(-1).sqrt()
 edge_error=(edge_lengths-1).abs().amax(dim=(1,2))
 valid=valid&(edge_error<0.00001)
 values=list(zip(valid.tolist(),minsep.tolist(),minimum_box.tolist(),edge_error.tolist()))
 for row,(passed,separation,clearance,error) in zip(accepted,values):
  row['independent_check']={'passed':passed,'engine':'PyTorch MPS; CPU fallback disabled',
    'min_pair_separation':separation,'min_square_clearance':clearance,
    'max_unit_edge_error':error,'precision':'float32; positive margins required'}
 result['independent_check']={'layouts_tested':len(accepted),'all_passed':bool(valid.all().item()),
                              'engine':'PyTorch MPS','cpu_numerical_fallback':False}
 (ROOT/'results'/'n3_branches.json').write_text(json.dumps(result,indent=2))
 print(result['independent_check'])

def algebra_sanity():
 h=mx.sqrt(mx.array(3.0))/2;q=mx.sqrt(mx.array(6.0))/4
 aa=(mx.sqrt(mx.array(6.0))-mx.sqrt(mx.array(2.0)))/4
 bb=(mx.sqrt(mx.array(6.0))+mx.sqrt(mx.array(2.0)))/4
 side=h+q
 vs=mx.stack([
     mx.stack([mx.array([0.0,0.0]),mx.stack([aa,bb]),mx.stack([bb,aa])]),
     mx.stack([mx.stack([q,q]),mx.stack([q-mx.array(0.5),side]),mx.stack([q+mx.array(0.5),side])]),
     mx.stack([mx.stack([q,q]),mx.stack([side,q-mx.array(0.5)]),mx.stack([side,q+mx.array(0.5)])])])
 edge=mx.sqrt(mx.sum((mx.roll(vs,-1,axis=1)-vs)**2,axis=-1))
 edge_error=mx.max(mx.abs(edge-1))
 polynomial=64*side**4-144*side**2+9
 # Exhaustive sampling is NOT used for proof: this checks a hand derivation.
 angles=mx.linspace(0,2*PI/3,12001)
 alpha=angles[:,None]+mx.arange(3)[None,:]*2*PI/3
 xy=mx.stack([mx.cos(alpha),mx.sin(alpha)],axis=-1)/mx.sqrt(mx.array(3.0))
 support=mx.max(xy[:,:,0]+xy[:,:,1],axis=1)-mx.min(xy[:,:,0],axis=1)-mx.min(xy[:,:,1],axis=1)
 endpoints=mx.array([0.0,90.0,105.0,120.0])*PI/180
 ae=endpoints[:,None]+mx.arange(3)[None,:]*2*PI/3
 ve=mx.stack([mx.cos(ae),mx.sin(ae)],axis=-1)/mx.sqrt(mx.array(3.0))
 vals=mx.max(ve[:,:,0]+ve[:,:,1],axis=1)-mx.min(ve[:,:,0],axis=1)-mx.min(ve[:,:,1],axis=1)
 area_bound=mx.sqrt(3*mx.sqrt(mx.array(3.0))/4)
 mx.eval(vs,edge_error,polynomial,support,vals,side,area_bound)
 result={'construction_side':side.item(),'construction_side_exact':'sqrt(3)/2+sqrt(6)/4',
         'construction_vertices':vs.tolist(),'construction_max_edge_error_gpu':edge_error.item(),
         'quartic':'64 s^4 - 144 s^2 + 9 = 0','quartic_residual_gpu':polynomial.item(),
         'corner_support_switch_degrees':[0,90,105,120],
         'corner_endpoint_values_gpu':vals.tolist(),'corner_minimum_exact':'sqrt(6)/2',
         'sampled_corner_minimum_gpu':mx.min(support).item(),
         'unconditional_area_lower_bound_gpu':area_bound.item(),
         'global_optimality_proved':False,'restricted_layout_optimality_proved_by_hand_algebra':True,
         'restriction':'Top and right triangles have wall-aligned bases and share an inward vertex; third triangle is contained in x+y<=2(s-sqrt(3)/2).',
         'missing_global_step':'No proof that every packing below the listed side can be reduced to this restricted layout.'}
 (ROOT/'results'/'n3_algebra.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='construction_vertices'},indent=2))

if __name__=='__main__':
 check_branches();algebra_sanity()
