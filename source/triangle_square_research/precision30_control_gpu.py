"""GPU-only interval row helpers with overflow-safe midpoints and deeper trees."""
import mlx.core as mx
from certified_spatial_search import filter_rows
from check_compact_splits_mps import transfer,D
import torch

def split_rows(rows,dimensions,min_width):
    dim=mx.maximum(dimensions,0)
    low=mx.take_along_axis(rows[:,3::2],dim[:,None],axis=1)[:,0].astype(mx.int64)
    high=mx.take_along_axis(rows[:,4::2],dim[:,None],axis=1)[:,0].astype(mx.int64)
    can=(dimensions>=0)&(high-low>min_width)&(rows[:,2]<360)
    parent=filter_rows(rows,can);terminal=filter_rows(rows,~can)
    if parent is None:return None,terminal,None
    d=filter_rows(dimensions[:,None],can)[:,0]
    lo=mx.take_along_axis(parent[:,3::2],d[:,None],axis=1)[:,0].astype(mx.int64)
    hi=mx.take_along_axis(parent[:,4::2],d[:,None],axis=1)[:,0].astype(mx.int64)
    mid=lo+(hi-lo)//2;ix=mx.arange(parent.shape[0])
    a=parent.at[ix,4+2*d].add((mid-hi).astype(mx.int32)).at[:,2].add(1)
    b=parent.at[ix,3+2*d].add((mid-lo).astype(mx.int32)).at[:,2].add(1)
    return mx.stack([a,b],axis=1).reshape(-1,23),terminal,d

def check_split_union(kept,dims,children,terminal,min_width):
    p=transfer(kept);d=transfer(dims).reshape(-1)
    lo=torch.zeros(p.shape[0],device=D,dtype=torch.int64);hi=torch.zeros_like(lo)
    for j in range(10):
        lo=torch.where(d==j,p[:,3+2*j].to(torch.int64),lo);hi=torch.where(d==j,p[:,4+2*j].to(torch.int64),hi)
    branch=(d>=0)&(hi-lo>min_width)&(p[:,2]<360)
    ix=torch.nonzero(branch,as_tuple=True)[0];tx=torch.nonzero(~branch,as_tuple=True)[0]
    valid=torch.ones((),device=D,dtype=torch.bool)
    if children is not None:
        child=transfer(children).reshape(-1,2,23);parent=p[ix]
        assert child.shape[0]==parent.shape[0]
        a,b=child[:,0],child[:,1];active=torch.arange(10,device=D)[None,:]==d[ix,None]
        valid &= (a[:,:2]==parent[:,:2]).all()&(b[:,:2]==parent[:,:2]).all()
        valid &= (a[:,2]==parent[:,2]+1).all()&(b[:,2]==parent[:,2]+1).all()
        valid &= (a[:,3::2]==parent[:,3::2]).all()&(b[:,4::2]==parent[:,4::2]).all()
        valid &= torch.where(active,a[:,4::2]==b[:,3::2],a[:,4::2]==parent[:,4::2]).all()
        valid &= torch.where(active,a[:,4::2]==b[:,3::2],b[:,3::2]==parent[:,3::2]).all()
        valid &= torch.where(active,(a[:,4::2]>parent[:,3::2])&(a[:,4::2]<parent[:,4::2]),True).all()
    else:valid &= ix.numel()==0
    if terminal is not None:
        t=transfer(terminal);assert t.shape==p[tx].shape;valid &= (t==p[tx]).all()
    else:valid &= tx.numel()==0
    assert valid.item(),'Precision30 split coverage failure'
    return True
