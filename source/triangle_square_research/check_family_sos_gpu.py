"""Exact GPU polynomial coefficient check of a family-level SOS certificate.
All coefficient arithmetic uses int32 on GPU. This verifies a polynomial
identity; the supplied family equations remain explicit premises.
"""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import mlx.core as mx
import torch
from contact_catalog_gpu import OUT
mx.set_default_device(mx.gpu)

# Dense coefficient array for u,h,nx,ny, each exponent 0,1,2.
index=mx.arange(81,dtype=mx.int32)
exponents=mx.stack([index//27,(index//9)%3,(index//3)%3,index%3],axis=1)
summed=exponents[:,None,:]+exponents[None,:,:]
valid=mx.all(summed<=2,axis=-1)
destination=mx.sum(mx.minimum(summed,2)*mx.array([27,9,3,1]),axis=-1).reshape(-1)

def basis(e):return mx.all(exponents==mx.array(e)[None,:],axis=1).astype(mx.int32)
def product(a,b):
 raw=a[:,None]*b[None,:]
 overflow=mx.any((~valid)&(raw!=0));mx.eval(overflow)
 assert not overflow.item(),'Polynomial degree exceeds the coefficient basis'
 values=mx.where(valid,raw,0).reshape(-1)
 return mx.zeros((81,),dtype=mx.int32).at[destination].add(values)

u=basis([1,0,0,0]);h=basis([0,1,0,0]);nx=basis([0,0,1,0]);ny=basis([0,0,0,1]);one=basis([0,0,0,0])
uu=product(u,u);hh=product(h,h)
unit=product(nx,nx)+product(ny,ny)-one
contact=product(u,nx+ny)-h
gap=2*uu-hh
sos=product(uu,product(nx-ny,nx-ny))
left=gap-sos
right=product(contact,product(u,nx+ny)+h)-2*product(uu,unit)
same=mx.all(left==right)
mx.eval(same,left,right)
assert same.item()

# Independent implementation, building the convolution with expanded exponent
# tensors on the MPS device rather than reusing the MLX destinations.
D=torch.device('mps')
ti=torch.arange(81,device=D,dtype=torch.int32)
te=torch.stack([ti//27,(ti//9)%3,(ti//3)%3,ti%3],dim=1)
ts=te[:,None,:]+te[None,:,:]
tv=(ts<=2).all(-1)
td=(ts.clamp(max=2)*torch.tensor([27,9,3,1],device=D,dtype=torch.int32)).sum(-1).reshape(-1)
def tb(e):return (te==torch.tensor(e,device=D,dtype=torch.int32)).all(-1).to(torch.int32)
def tp(a,b):
 raw=a[:,None]*b[None,:]
 assert not ((~tv)&(raw!=0)).any().item()
 values=torch.where(tv,raw,0).reshape(-1)
 return torch.zeros(81,device=D,dtype=torch.int32).scatter_add(0,td,values)
tu=tb([1,0,0,0]);th=tb([0,1,0,0]);tx=tb([0,0,1,0]);ty=tb([0,0,0,1]);to=tb([0,0,0,0])
tuu=tp(tu,tu);thh=tp(th,th)
g1=tp(tx,tx)+tp(ty,ty)-to;g2=tp(tu,tx+ty)-th
tl=2*tuu-thh-tp(tuu,tp(tx-ty,tx-ty))
tr=tp(g2,tp(tu,tx+ty)+th)-2*tp(tuu,g1)
assert torch.equal(tl,tr)
assert torch.equal(tl,torch.tensor(left.tolist(),device=D,dtype=torch.int32))
result={'passed':True,'arithmetic':'exact small int32 polynomial coefficients on MLX Metal and PyTorch MPS',
 'variables':['u','h','nx','ny'],'monomial_exponents':exponents.tolist(),
 'left_coefficients':left.tolist(),'right_coefficients':right.tolist(),
 'premises':['nx^2+ny^2=1','u*(nx+ny)=h','u=s-h>0','h=sqrt(3)/2'],
 'identity':'2*u^2-h^2-u^2*(nx-ny)^2 = (u*(nx+ny)-h)*(u*(nx+ny)+h)-2*u^2*(nx^2+ny^2-1)',
 'consequence':'2*u^2-h^2 = u^2*(nx-ny)^2 >= 0; hence s >= sqrt(3)/2+sqrt(6)/4',
 'scope':'Known wall/corner/shared-vertex contact family only. Premises are supplied by that geometry; other families are not excluded.',
 'global_optimality_proved':False,'cpu_numerical_fallback':False}
(OUT/'family_sos_certificate.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['monomial_exponents','left_coefficients','right_coefficients']},indent=2))
