"""Independent MPS enumeration of all 4096 wall masks and D4 representatives.
Also checks a local shrinking direction for contact strata with no pair contact.
"""
import os,json,time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
from contact_catalog_gpu import OUT
from check_contact_catalog_mps import read_u32
D=torch.device('mps')

def run():
 start=time.monotonic()
 masks=torch.arange(4096,device=D,dtype=torch.int32)
 bit=((masks[:,None]>>torch.arange(12,device=D,dtype=torch.int32)[None,:])&1).reshape(4096,3,4)
 walls=bit.any(dim=1)
 structure=(~((walls[:,0]&walls[:,1])|(walls[:,2]&walls[:,3])))&(walls.to(torch.int32).sum(1)<=2)
 # Fifteen-degree endpoints/midpoints, independent of the enumerator's 30-degree grid.
 phi=torch.tensor([12,0,18,6],device=D,dtype=torch.int32)
 beta=torch.tensor([22,6,14],device=D,dtype=torch.int32)
 angle=torch.arange(24,device=D,dtype=torch.int32)
 delta=torch.remainder(phi[None,:,None]-beta[:,None,None]-angle[None,None,:]+12,24)-12
 support=delta.abs()<=4
 permitted=((bit[:,:,:,None]==0)|support[None,:,:,:]).all(dim=(1,2)).any(1)
 rot1=((masks<<4)&4095)|(masks>>8);rot2=((masks<<8)&4095)|(masks>>4)
 canonical=masks==torch.minimum(masks,torch.minimum(rot1,rot2))
 table=masks[structure&permitted&canonical]
 original=read_u32(OUT/'wall_masks.npy')
 assert torch.equal(table,original)
 w=table.numel()
 matrices=torch.tensor([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]],
           [[-1,0],[0,1]],[[0,-1],[-1,0]],[[1,0],[0,-1]],[[0,1],[1,0]]],device=D,dtype=torch.int32)
 norm=torch.tensor([[-1,0],[1,0],[0,-1],[0,1]],device=D,dtype=torch.int32)
 changed=(matrices[:,None,:,:]*norm[None,:,None,:]).sum(-1)
 mapwall=(changed[:,:,None,:]==norm[None,None,:,:]).all(-1).to(torch.int32).argmax(-1)
 determinant=matrices[:,0,0]*matrices[:,1,1]-matrices[:,0,1]*matrices[:,1,0]
 active=((table[:,None]>>torch.arange(12,device=D,dtype=torch.int32)[None,:])&1)
 transformed=[]
 for d in range(8):
  vertex=torch.arange(3,device=D,dtype=torch.int32)
  vv=torch.where(determinant[d]<0,torch.remainder(-vertex,3),vertex)
  positions=(4*vv[:,None]+mapwall[d][None,:]).reshape(12)
  weights=torch.bitwise_left_shift(torch.ones(12,device=D,dtype=torch.int32),positions)
  m=(active*weights[None,:]).sum(1).to(torch.int32)
  a=((m<<4)&4095)|(m>>8);b=((m<<8)&4095)|(m>>4)
  m=torch.minimum(m,torch.minimum(a,b))
  ids=(table[None,:]<m[:,None]).sum(1).to(torch.int32)
  assert torch.equal(table[ids],m)
  transformed.append(ids)
 transform=torch.stack(transformed)
 i=torch.arange(w*w*w,device=D,dtype=torch.int32)
 triple=torch.stack([i//(w*w),(i//w)%w,i%w],dim=1)
 representative=[]
 for d in range(8):
  t=transform[d][triple].sort(dim=1).values
  representative.append(t[:,0]*w*w+t[:,1]*w+t[:,2])
 minimum=torch.stack(representative).amin(0)
 keep=i==minimum
 flags=(table[:,None]>>(4*torch.arange(3,device=D,dtype=torch.int32)[None,:]))&15
 corner=((flags&3)!=0)&((flags&12)!=0)
 c=torch.where(corner,((flags>>1)&1)+2*((flags>>3)&1),99).amin(1)
 cc=c[triple]
 duplicate=((cc[:,0]==cc[:,1])&(cc[:,0]<99))|((cc[:,0]==cc[:,2])&(cc[:,0]<99))|((cc[:,1]==cc[:,2])&(cc[:,1]<99))
 reps=triple[keep&~duplicate]
 assert torch.equal(reps,read_u32(OUT/'wall_triplets.npy'))
 # Exact int32 coefficient check of an infinitesimal shrinking direction.
 touched=flags[:,:,None].bitwise_and(torch.tensor([1,2,4,8],device=D,dtype=torch.int32)[None,None,:])!=0
 touch=touched.any(1)
 dx=torch.where(touch[:,0],1,torch.where(touch[:,1],-1,0))
 dy=torch.where(touch[:,2],1,torch.where(touch[:,3],-1,0))
 velocity=torch.stack([dx,dy],dim=1).to(torch.int32) # twice the centroid velocity
 derivative=(velocity[:,None,:]*norm[None,:,:]).sum(-1)+1
 shrink_ok=bool(((~touch)|(derivative==0)).all().item())
 assert shrink_ok
 result={'passed':True,'wall_subsets_checked':4096,'independent_angle_grid_degrees':15,
    'wall_type_count':w,'symmetry_representative_count':reps.shape[0],
    'all_wall_masks_and_representatives_match':True,'cpu_numerical_fallback':False,
    'elapsed_seconds':time.monotonic()-start,
    'no_pair_contact_stratum_certificate':{
      'integer_wall_derivatives_checked':shrink_ok,'centroid_velocity_times_two':velocity.tolist(),
      'square_side_velocity':-1,'domain':'s>1 and all three triangle pairs have strict positive separation',
      'consequence':'Such a packing is not locally optimal: shrink the square slightly while moving individual triangles to preserve their active walls; strict pair gaps remain positive by continuity.',
      'scope':'Applies to actual no-contact strata, not to closures or contact-requirement prefixes.'},
    'global_optimality_proved':False}
 (OUT/'wall_independent_check.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k!='no_pair_contact_stratum_certificate'},indent=2))

if __name__=='__main__':run()
