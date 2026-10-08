"""Independent MPS child-union validation; avoids integer-rounded gather."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import numpy as np
import torch
from check_contact_catalog_mps import D

def transfer(a):
    # Byte transport only; no NumPy numeric computation.
    return torch.frombuffer(bytearray(np.asarray(a).tobytes()),dtype=torch.int32).to(D).reshape(a.shape)

def check(kept,dims,children,terminal,min_width):
    p=transfer(kept);d=transfer(dims).reshape(-1)
    lo=torch.zeros(p.shape[0],device=D,dtype=torch.int32);hi=torch.zeros_like(lo)
    for j in range(10):
        lo=torch.where(d==j,p[:,3+2*j],lo);hi=torch.where(d==j,p[:,4+2*j],hi)
    branch=(d>=0)&(hi-lo>min_width)&(p[:,2]<180)
    ix=torch.nonzero(branch,as_tuple=True)[0];tx=torch.nonzero(~branch,as_tuple=True)[0]
    valid=torch.ones((),device=D,dtype=torch.bool)
    if children is not None:
        child=transfer(children).reshape(-1,2,23);parent=p[ix]
        assert child.shape[0]==parent.shape[0]
        left=child[:,0];right=child[:,1]
        active=torch.arange(10,device=D)[None,:]==d[ix,None]
        valid &= (left[:,:2]==parent[:,:2]).all() & (right[:,:2]==parent[:,:2]).all()
        valid &= (left[:,2]==parent[:,2]+1).all() & (right[:,2]==parent[:,2]+1).all()
        valid &= (left[:,3::2]==parent[:,3::2]).all() & (right[:,4::2]==parent[:,4::2]).all()
        valid &= torch.where(active,left[:,4::2]==right[:,3::2],left[:,4::2]==parent[:,4::2]).all()
        valid &= torch.where(active,left[:,4::2]==right[:,3::2],right[:,3::2]==parent[:,3::2]).all()
        valid &= torch.where(active,(left[:,4::2]>parent[:,3::2])&(left[:,4::2]<parent[:,4::2]),True).all()
        valid &= (left[:,3::2]<=left[:,4::2]).all() & (right[:,3::2]<=right[:,4::2]).all()
    else:valid &= ix.numel()==0
    if terminal is not None:
        t=transfer(terminal);assert t.shape==p[tx].shape
        valid &= (t==p[tx]).all()
    else:valid &= tx.numel()==0
    assert valid.item(),'Independent MPS bisection coverage check failed'
    return True
