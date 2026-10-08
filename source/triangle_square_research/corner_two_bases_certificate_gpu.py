"""Exact lower bound for a corner triangle opposite two wall-base triangles.

After square symmetry, corner vertex A=(0,0), right apex P=(u,z1), top
apex T=(z2,u), u=s-h. Base containment gives z1,z2 in [1/2,s-1/2].
For (sqrt(3)+1)/2<s<sqrt(3), these apices lie strictly in the common
30..60 degree cone of every unit triangle supported at the corner. Therefore
nonoverlap with that triangle forces A1=nx*u+ny*z1-h>=0 and
A2=nx*z2+ny*u-h>=0. Its opposite-edge unit normal has nx,ny>0.
The right/top triangles can separate only on a facing sloped edge: their
wall edges and the other sloped edges cannot separate their contained bases.
Thus C=(1+k)*u-z1-k*z2>=0, for k=sqrt(3) or 1/sqrt(3).
The verified weighted polynomial identity yields u*(nx+ny)>=h. Unit-normal
Cauchy (also the existing verified SOS identity) yields u>=h/sqrt(2),
hence s>=sqrt(3)/2+sqrt(6)/4. No pair-contact equality is assumed.
"""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT
mx.set_default_device(mx.gpu)
sizes=mx.array([2,2,3,3,2,2,2],dtype=mx.int32)
weights=mx.array([144,72,24,8,4,2,1],dtype=mx.int32)
indices=mx.arange(288,dtype=mx.int32)
exponents=(indices[:,None]//weights[None,:])%sizes[None,:]
summed=exponents[:,None,:]+exponents[None,:,:];valid=mx.all(summed<sizes,axis=2)
dest=mx.sum(mx.minimum(summed,sizes-1)*weights,axis=2).reshape(-1)
def basis(e):return mx.all(exponents==mx.array(e,dtype=mx.int32),axis=1).astype(mx.int32)
def product(a,b):
    values=a[:,None]*b[None,:]
    assert not mx.any((~valid)&(values!=0)).item()
    return mx.zeros((288,),dtype=mx.int32).at[dest].add(mx.where(valid,values,0).reshape(-1))
def check():
    terms=[]
    for j in range(7):
        e=[0]*7;e[j]=1;terms.append(basis(e))
    u,h,x,y,z,w,k=terms;one=basis([0]*7)
    a=product(x,u)+product(y,z)-h;b=product(x,w)+product(y,u)-h
    c=product(one+k,u)-z-product(k,w)
    lhs=product(x,a)+product(product(k,y),b)+product(product(x,y),c)
    rhs=product(x+product(k,y),product(u,x+y)-h)
    assert mx.all(lhs==rhs).item()
    d=torch.device('mps');ti=torch.arange(288,device=d,dtype=torch.int32)
    sz=torch.tensor([2,2,3,3,2,2,2],device=d,dtype=torch.int32);wt=torch.tensor([144,72,24,8,4,2,1],device=d,dtype=torch.int32)
    ex=(ti[:,None]//wt)%sz;sm=ex[:,None,:]+ex[None,:,:];good=(sm<sz).all(2)
    dt=(torch.minimum(sm,sz-1)*wt).sum(2).reshape(-1)
    def tb(e):return (ex==torch.tensor(e,device=d,dtype=torch.int32)).all(1).to(torch.int32)
    def tp(a,b):
        v=a[:,None]*b[None,:];assert not ((~good)&(v!=0)).any().item()
        return torch.zeros(288,device=d,dtype=torch.int32).scatter_add(0,dt,torch.where(good,v,0).reshape(-1))
    tt=[]
    for j in range(7):
        e=[0]*7;e[j]=1;tt.append(tb(e))
    u,h,x,y,z,w,k=tt;o=tb([0]*7)
    a=tp(x,u)+tp(y,z)-h;b=tp(x,w)+tp(y,u)-h;c=tp(o+k,u)-z-tp(k,w)
    left=tp(x,a)+tp(tp(k,y),b)+tp(tp(x,y),c);right=tp(x+tp(k,y),tp(u,x+y)-h)
    assert torch.equal(left,right) and torch.equal(left,torch.tensor(lhs.tolist(),device=d,dtype=torch.int32))
    result={'passed':True,'variables':['u','h','nx','ny','z1','z2','k'],
      'identity':'nx*A1+k*ny*A2+nx*ny*C = (nx+k*ny)*(u*(nx+ny)-h)',
      'premises':['A1>=0','A2>=0','C>=0','nx>0','ny>0','k>0','nx^2+ny^2=1','u=s-h>0','h=sqrt(3)/2'],
      'side_window':'(sqrt(3)+1)/2 < s < sqrt(3)',
      'conclusion':'s >= sqrt(3)/2+sqrt(6)/4',
      'coefficient_identity_independently_checked':True,'CPU_numerical_fallback':False,
      'geometric_derivation':__doc__,'left_coefficients':lhs.tolist(),'right_coefficients':rhs.tolist(),
      'global_optimality_proved':False}
    (ROOT/'certified_spatial/corner_two_bases_certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['left_coefficients','right_coefficients','geometric_derivation']},indent=2),flush=True)

if __name__=='__main__':check()
