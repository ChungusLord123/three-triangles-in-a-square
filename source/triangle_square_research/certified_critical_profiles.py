"""Exact critical-profile reduction, preserving flexible/singular candidates.
Some contact-connected component of an optimum for s>1 spans opposite walls;
otherwise components admit an inward translating/shrinking motion. This is
an optimality-candidate reduction, not an infeasibility claim for the prefixes.
"""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import mlx.core as mx
import torch
from contact_catalog_gpu import OUT
from n3_branches_gpu import ROOT
from check_contact_catalog_mps import read_u32
mx.set_default_device(mx.gpu)
DEST=ROOT/'certified_spatial';DEST.mkdir(exist_ok=True)

def prepare():
 summary=json.loads((OUT/'screening_summary.json').read_text())
 ids=mx.concatenate([mx.load(p['ids_file']) for p in summary['parts'] if p['ids_file']])
 masks=mx.load(str(OUT/'wall_masks.npy'));triples=mx.load(str(OUT/'wall_triplets.npy'))
 m=ids%50653;codes=mx.stack([m//1369,(m//37)%37,m%37],axis=1)
 wm=masks[triples[ids//50653]]
 wall=(wm|(wm>>4)|(wm>>8))&15
 con=mx.broadcast_to(mx.array([1,2,4],dtype=mx.uint32)[None,:],(ids.size,3))
 for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
  additions=mx.array([1<<j if v==i else 1<<i if v==j else 0 for v in range(3)],dtype=mx.uint32)
  con=con|mx.where((codes[:,k]>0)[:,None],additions[None,:],0)
 for k in range(3):con=con|mx.where(((con>>k)&1)>0,con[:,k,None],0)
 flags=mx.zeros_like(con)
 for k in range(3):flags=flags|mx.where(((con>>k)&1)>0,wall[:,k,None],0)
 keep=mx.any(((flags&3)==3)|((flags&12)==12),axis=1)
 n=mx.sum(keep.astype(mx.int32));mx.eval(n,keep,ids)
 index=mx.argsort(mx.where(keep,mx.arange(ids.size),999999999))[:n.item()]
 critical=ids[index]
 mx.save(str(DEST/'critical_profiles.npy'),critical)
 # Independent GPU Boolean component closure.
 d=torch.device('mps')
 tm=torch.tensor(masks.tolist(),device=d,dtype=torch.int32)
 tt=torch.tensor(triples.tolist(),device=d,dtype=torch.int32)
 ti=torch.tensor(ids.tolist(),device=d,dtype=torch.int32)
 mode=ti%50653;tc=torch.stack([mode//1369,(mode//37)%37,mode%37],dim=1)
 tw=tm[tt[ti//50653]];tw=(tw|(tw>>4)|(tw>>8))&15
 adj=torch.eye(3,device=d,dtype=torch.bool)[None,:,:].expand(ti.numel(),-1,-1).clone()
 for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):adj[:,i,j]=tc[:,k]>0;adj[:,j,i]=tc[:,k]>0
 for _ in range(2):adj=torch.bmm(adj.to(torch.float32),adj.to(torch.float32))>0
 tf=torch.zeros_like(tw)
 for k in range(3):tf|=torch.where(adj[:,:,k],tw[:,k,None],0)
 tk=(((tf&3)==3)|((tf&12)==12)).any(1)
 assert torch.equal(tk,torch.tensor(keep.tolist(),device=d,dtype=torch.bool))
 result={'input_angular_survivors':ids.size,'critical_candidates':critical.size,
   'excluded_noncritical_proper_profiles':ids.size-critical.size,'independent_MPS_check_passed':True,
   'criterion':'At least one contact-connected component touches opposite square walls.',
   'domain':'Proper full active contact profiles of a global optimum with s>1.',
   'justification':'If every component touches at most two adjacent walls, translate each component at velocity +/-1/2 in the touched coordinate directions while s decreases at velocity -1. Internal contacts are preserved; strict intercomponent gaps and untapped-wall clearances remain positive for a sufficiently small step.',
   'scope_warning':'A skipped contact-requirement prefix may still contain a feasible packing with extra contacts. Coverage is justified by the full active profile of an attained optimum, which is represented by the catalogue.',
   'cpu_numerical_fallback':False,'global_optimality_proved':False}
 (DEST/'critical_profile_check.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 return result

def geometry_filter(numerator=14783,denominator=10000):
 ids=mx.load(str(DEST/'critical_profiles.npy'))
 masks=mx.load(str(OUT/'wall_masks.npy'));triples=mx.load(str(OUT/'wall_triplets.npy'))
 mode=ids%50653;code=mx.stack([mode//1369,(mode//37)%37,mode%37],axis=1).astype(mx.int32)
 wm=masks[triples[ids//50653]]
 bits=((wm[:,:,None,None]>>mx.arange(12,dtype=mx.uint32).reshape(1,1,3,4))&1)>0
 full=bits&mx.roll(bits,-1,axis=2)
 edge_on_wall=mx.any(full,axis=-1)
 wall_full=mx.any(full,axis=2)
 twin=mx.any(wall_full[:,0]&wall_full[:,1],axis=-1)|mx.any(wall_full[:,0]&wall_full[:,2],axis=-1)|mx.any(wall_full[:,1]&wall_full[:,2],axis=-1)
 target_under_two=mx.array(numerator,dtype=mx.int64)<2*mx.array(denominator,dtype=mx.int64)
 invalid=twin&target_under_two
 for p,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
  c=code[:,p];ve=(c>=10)&(c<28);ee=c>=28
  z=mx.clip(c-10,0,17);owner=z//9;e=(z%9)//3
  ei=edge_on_wall[:,i];ej=edge_on_wall[:,j]
  edge_i=mx.take_along_axis(ei,e[:,None],axis=1)[:,0]
  edge_j=mx.take_along_axis(ej,e[:,None],axis=1)[:,0]
  invalid|=ve&mx.where(owner==0,edge_i,edge_j)
  z=mx.clip(c-28,0,8)
  invalid|=ee&(mx.take_along_axis(ei,(z//3)[:,None],axis=1)[:,0]|mx.take_along_axis(ej,(z%3)[:,None],axis=1)[:,0])
 keep=~invalid;n=mx.sum(keep.astype(mx.int32));mx.eval(n,keep)
 ind=mx.argsort(mx.where(keep,mx.arange(ids.size),999999999))[:n.item()]
 selected=ids[ind];mx.save(str(DEST/'spatial_candidates.npy'),selected)
 # Independently reproduce every rejection on MPS.
 d=torch.device('mps');ti=torch.tensor(ids.tolist(),device=d,dtype=torch.int32)
 tm=torch.tensor(masks.tolist(),device=d,dtype=torch.int32);tt=torch.tensor(triples.tolist(),device=d,dtype=torch.int32)
 tw=tm[tt[ti//50653]];m=ti%50653;tc=torch.stack([m//1369,(m//37)%37,m%37],dim=1)
 tb=((tw[:,:,None,None]>>torch.arange(12,device=d,dtype=torch.int32).reshape(1,1,3,4))&1)>0
 tf=tb&torch.roll(tb,-1,dims=2);te=tf.any(-1);wf=tf.any(2)
 bad=(wf[:,0]&wf[:,1]).any(-1)|(wf[:,0]&wf[:,2]).any(-1)|(wf[:,1]&wf[:,2]).any(-1)
 bad&=(torch.tensor(numerator,device=d,dtype=torch.int64)<2*torch.tensor(denominator,device=d,dtype=torch.int64))
 for p,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
  c=tc[:,p];z=(c-10).clamp(0,17);e=(z%9)//3
  a=te[:,i].gather(1,e[:,None])[:,0];b=te[:,j].gather(1,e[:,None])[:,0]
  bad|=((c>=10)&(c<28))&torch.where(z//9==0,a,b)
  z=(c-28).clamp(0,8)
  bad|=(c>=28)&(te[:,i].gather(1,(z//3)[:,None])[:,0]|te[:,j].gather(1,(z%3)[:,None])[:,0])
 assert torch.equal(~bad,torch.tensor(keep.tolist(),device=d,dtype=torch.bool))
 result={'input_critical_candidates':ids.size,'spatial_candidates':selected.size,'geometric_exclusions':ids.size-selected.size,
    'target_numerator':numerator,'target_denominator':denominator,'independent_MPS_check_passed':True,
    'rules':['A proper vertex-edge interior contact or positive edge-edge overlap cannot use an edge lying on the container wall: both polygons would require interior on the same side.',
             'For s<2, two unit triangle edges cannot both lie on the same square wall: their unit-length base intervals must overlap in positive length, causing interior overlap.'],
    'boundary_cases':'Endpoint contacts are covered by vertex-vertex modes; zero-length edge overlaps are also covered there.',
    'cpu_numerical_fallback':False,'global_optimality_proved':False}
 (DEST/'geometric_profile_check.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 return result

if __name__=='__main__':prepare();geometry_filter()
