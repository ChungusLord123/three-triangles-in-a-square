"""Necessary rotation-criticality condition for full optimum contact profiles.

If every component that determines square size spans just one opposite-wall
pair, with one unique vertex on each of those walls, rotate each such rigid
component a little to reduce its determining span. That span is A cos(a)+
B sin(a)>0 and has second derivative equal to its negative. A nonzero first
derivative permits descent; a zero derivative is a strict local maximum.
The other span has strict slack. Unconnected components have strict gaps, so
small rotations and translations preserve disjoint interiors. Thus an attained
optimum needs a component spanning both wall pairs, or at least two labelled
support vertices on a wall of a determining pair. Counting labels overcounts
coincident supports and only retains extra cases.

This rejects noncritical FULL active profiles, not feasible requirement
prefixes. Extra-contact boundary solutions are represented by their full
profiles and must not be discarded globally. Flexible and singular pieces
are preserved; the motion rotates each entire connected component rigidly.
"""
import os,json,hashlib
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT
from contact_catalog_gpu import OUT
from check_contact_catalog_mps import read_u32,D
mx.set_default_device(mx.gpu)

def decisions(ids,masks,triples):
    mode=ids%50653
    code=mx.stack([mode//1369,(mode//37)%37,mode%37],axis=1)
    wm=masks[triples[ids//50653]]
    bits=((wm[:,:,None,None]>>mx.arange(12,dtype=mx.uint32).reshape(1,1,3,4))&1).astype(mx.int32)
    supports=mx.sum(bits,axis=2)
    connection=mx.broadcast_to(mx.array([1,2,4],dtype=mx.uint32)[None,:],(ids.size,3))
    for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        add=mx.array([1<<j if v==i else 1<<i if v==j else 0 for v in range(3)],dtype=mx.uint32)
        connection|=mx.where((code[:,k]>0)[:,None],add[None,:],0)
    for k in range(3):connection|=mx.where(((connection>>k)&1)>0,connection[:,k,None],0)
    count=mx.zeros((ids.size,3,4),dtype=mx.int32)
    for k in range(3):count+=mx.where((((connection>>k)&1)>0)[:,:,None],supports[:,k,None,:],0)
    horizontal=(count[:,:,0]>0)&(count[:,:,1]>0)
    vertical=(count[:,:,2]>0)&(count[:,:,3]>0)
    must_keep=(horizontal&vertical)|(horizontal&((count[:,:,0]>1)|(count[:,:,1]>1)))|(vertical&((count[:,:,2]>1)|(count[:,:,3]>1)))
    return mx.any(must_keep,axis=1)

def reference_decisions(ids,masks,triples):
    mode=ids%50653;code=torch.stack([mode//1369,(mode//37)%37,mode%37],dim=1)
    wm=masks[triples[ids//50653]]
    bits=((wm[:,:,None,None]>>torch.arange(12,device=D,dtype=torch.int32).reshape(1,1,3,4))&1)
    supports=bits.sum(2)
    adj=torch.eye(3,device=D,dtype=torch.bool)[None,:,:].expand(ids.numel(),-1,-1).clone()
    for k,(i,j) in enumerate([(0,1),(0,2),(1,2)]):adj[:,i,j]=code[:,k]>0;adj[:,j,i]=code[:,k]>0
    for _ in range(2):adj=torch.bmm(adj.to(torch.float32),adj.to(torch.float32))>0
    count=torch.zeros((ids.numel(),3,4),device=D,dtype=torch.int64)
    for k in range(3):count+=torch.where(adj[:,:,k,None],supports[:,k,None,:],0)
    horizontal=(count[:,:,0]>0)&(count[:,:,1]>0);vertical=(count[:,:,2]>0)&(count[:,:,3]>0)
    return ((horizontal&vertical)|(horizontal&((count[:,:,0]>1)|(count[:,:,1]>1)))|
            (vertical&((count[:,:,2]>1)|(count[:,:,3]>1)))).any(1)

def prepare():
    dest=ROOT/'certified_spatial'
    ids=mx.load(str(dest/'spatial_candidates.npy')).astype(mx.uint32)
    masks=mx.load(str(OUT/'wall_masks.npy')).astype(mx.uint32)
    triples=mx.load(str(OUT/'wall_triplets.npy')).astype(mx.uint32)
    keep=decisions(ids,masks,triples);n=mx.sum(keep.astype(mx.int32));mx.eval(ids,keep,n)
    ti=read_u32(dest/'spatial_candidates.npy');tm=read_u32(OUT/'wall_masks.npy');tt=read_u32(OUT/'wall_triplets.npy')
    rk=reference_decisions(ti,tm,tt)
    assert torch.equal(rk,torch.tensor(keep.tolist(),device=D,dtype=torch.bool))
    dense=mx.zeros((16918102,),dtype=mx.uint8).at[ids].add(keep.astype(mx.uint8))
    mx.save(str(dest/'rotation_critical_lookup.npy'),dense)
    known=json.loads((OUT/'critical_family.json').read_text())['known_pattern']['requirement_id']
    assert dense[known].item()==1
    result={'passed':True,'input_profiles':ids.size,'rotation_critical_profiles':n.item(),
      'noncritical_full_profiles':ids.size-n.item(),'independent_MPS_all_decisions_passed':True,
      'known_singular_optimum_retained':True,'cpu_numerical_fallback':False,
      'scope':'Necessary condition on the full active profile of an attained optimum; not a claim that skipped prefixes have no feasible packings.',
      'criterion':'Some contact component spans all four walls, or spans one opposite pair with at least two labelled supports on one wall of that pair.',
      'justification':__doc__,'global_optimality_proved':False,
      'lookup_sha256':hashlib.sha256((dest/'rotation_critical_lookup.npy').read_bytes()).hexdigest()}
    (dest/'rotation_critical_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='justification'},indent=2),flush=True)
    return result

if __name__=='__main__':prepare()
