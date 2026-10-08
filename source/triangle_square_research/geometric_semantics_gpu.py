"""Exact polynomial audit of the geometry, velocities and feasible construction.

Two GPU implementations verify coefficient identities. MLX groups monomials
by an equality matrix; MPS sorts and sums groups. Floating point and CPU
geometric arithmetic are absent. Logical implications of the identities and
the interval contractors are documented separately in the coverage review.
"""
import os,json,time,hashlib
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT

NAMES=['h','q','a','c','s','t','x','y','nx','ny','u','z','w','k','omega','eta','vx','vy','wx','wy']
DEN=1728

class Ring:
    def __init__(self,backend):self.backend=backend;self.checks=[]
    def arr(self,a):
        return mx.array(a,dtype=mx.int64) if self.backend=='mlx' else torch.tensor(a,device='mps',dtype=torch.int64)
    def cat(self,a):return mx.concatenate(a) if self.backend=='mlx' else torch.cat(a)
    def all(self,a):return bool((mx.all(a) if self.backend=='mlx' else a.all()).item())
    def const(self,numerator):return self.arr([0]),self.arr([numerator])
    def var(self,name):
        p=self.arr([NAMES.index(name)])
        return self.arr([1])<<(3*p),self.arr([DEN])
    def add(self,*polys):return self.cat([p[0] for p in polys]),self.cat([p[1] for p in polys])
    def neg(self,p):return p[0],-p[1]
    def sub(self,a,b):return self.add(a,self.neg(b))
    def mul(self,a,b):
        c=(a[0][:,None]+b[0][None,:]).reshape(-1)
        raw=(a[1][:,None]*b[1][None,:]).reshape(-1)
        assert self.all(raw%DEN==0)
        v=raw//DEN
        for name,num,den in [('h',3,4),('q',3,8),('a',1,8)]:
            shift=3*NAMES.index(name);power=(c>>shift)&7
            # All present expressions have degree at most four in a radical.
            for _ in range(2):
                hit=power>=2;scaled=v*num
                assert self.all((~hit)|(scaled%den==0))
                if self.backend=='mlx':
                    v=mx.where(hit,scaled//den,v);c=mx.where(hit,c-(self.arr([2])<<shift),c)
                else:
                    v=torch.where(hit,scaled//den,v);c=torch.where(hit,c-(self.arr([2])<<shift),c)
                power=(c>>shift)&7
        return c,v
    def scale(self,p,n):return p[0],p[1]*n
    def derivative(self,p,name):
        shift=3*NAMES.index(name);degree=(p[0]>>shift)&7
        if self.backend=='mlx':
            code=mx.where(degree>0,p[0]-(self.arr([1])<<shift),0)
        else:code=torch.where(degree>0,p[0]-(self.arr([1])<<shift),0)
        return code,p[1]*degree
    def zero(self,p):
        code,value=p
        if self.backend=='mlx':
            grouped=mx.sum((code[:,None]==code[None,:]).astype(mx.int64)*value[None,:],axis=1)
            return self.all(grouped==0)
        order=torch.argsort(code);c=code[order];v=value[order]
        change=torch.cat([torch.ones(1,device='mps',dtype=torch.int64),(c[1:]!=c[:-1]).to(torch.int64)])
        group=torch.cumsum(change,0)-1
        sums=torch.zeros(code.numel(),device='mps',dtype=torch.int64).scatter_add(0,group,v)
        return self.all(sums==0)
    def equal(self,label,a,b):
        assert self.zero(self.sub(a,b)),label
        self.checks.append(label)
    def dot(self,a,b):return self.add(self.mul(a[0],b[0]),self.mul(a[1],b[1]))
    def vector_add(self,a,b):return self.add(a[0],b[0]),self.add(a[1],b[1])
    def vector_sub(self,a,b):return self.sub(a[0],b[0]),self.sub(a[1],b[1])
    def turn(self,a):return self.neg(a[1]),a[0]
    def rotate(self,a):
        c,s=self.var('c'),self.var('s')
        return self.sub(self.mul(c,a[0]),self.mul(s,a[1])),self.add(self.mul(s,a[0]),self.mul(c,a[1]))

def verify(r):
    zero,one,half=r.const(0),r.const(DEN),r.const(864)
    third,sixth=r.const(576),r.const(288)
    h,q,a=[r.var(v) for v in ['h','q','a']]
    radius=r.mul(h,third)
    base=[(half,r.neg(radius)),(zero,r.scale(radius,2)),(r.neg(half),r.neg(radius))]
    normals=[(h,half),(r.neg(h),half),(zero,r.neg(one))]
    radial=[(h,r.neg(half)),(zero,one),(r.neg(h),r.neg(half))]
    for i in range(3):
        diff=r.vector_sub(base[(i+1)%3],base[i])
        r.equal(f'body edge {i} has squared length one',r.dot(diff,diff),one)
        r.equal(f'edge normal {i} is unit',r.dot(normals[i],normals[i]),one)
        r.equal(f'edge {i} first endpoint support',r.dot(normals[i],base[i]),radius)
        r.equal(f'edge {i} second endpoint support',r.dot(normals[i],base[(i+1)%3]),radius)
        r.equal(f'edge {i} opposite support',r.dot(normals[i],base[(i+2)%3]),r.scale(radius,-2))
        r.equal(f'edge {i} support difference is altitude',r.scale(radius,3),h)
        r.equal(f'edge {i} tangent first endpoint',r.dot(r.turn(normals[i]),base[i]),r.neg(half))
        r.equal(f'edge {i} tangent second endpoint',r.dot(r.turn(normals[i]),base[(i+1)%3]),half)
        r.equal(f'radial normal {i} is unit',r.dot(radial[i],radial[i]),one)
        r.equal(f'radial normal {i} support is twice inradius',r.dot(radial[i],base[i]),r.scale(radius,2))
        r.equal(f'radial normal {i} tangent support zero',r.dot(r.turn(radial[i]),base[i]),zero)
    for coordinate in range(2):
        r.equal(f'body centroid coordinate {coordinate}',r.add(*[b[coordinate] for b in base]),zero)
    c,s,t=[r.var(v) for v in ['c','s','t']]
    norm=r.add(r.mul(c,c),r.mul(s,s))
    x,y=r.var('x'),r.var('y');v=(x,y);rv=r.rotate(v)
    r.equal('rotation preserves squared length with unit factor',r.dot(rv,rv),r.mul(norm,r.dot(v,v)))
    tt=r.mul(t,t);cn=r.sub(one,tt);sn=r.scale(t,2);den=r.add(one,tt)
    r.equal('rational rotation numerator is unit',r.add(r.mul(cn,cn),r.mul(sn,sn)),r.mul(den,den))
    xn=r.sub(r.mul(cn,x),r.mul(sn,y));yn=r.add(r.mul(sn,x),r.mul(cn,y))
    dx=r.sub(r.mul(r.derivative(xn,'t'),den),r.mul(xn,r.derivative(den,'t')))
    dy=r.sub(r.mul(r.derivative(yn,'t'),den),r.mul(yn,r.derivative(den,'t')))
    r.equal('bounded rotation X derivative numerator',dx,r.scale(yn,-2))
    r.equal('bounded rotation Y derivative numerator',dy,r.scale(xn,2))
    for e in range(3):
        n=normals[e]
        for w,nw in enumerate([(r.neg(one),zero),(one,zero),(zero,r.neg(one)),(zero,one)]):
            cc=r.dot(n,nw);ss=r.sub(r.mul(n[0],nw[1]),r.mul(n[1],nw[0]))
            for coordinate in range(2):
                rotated=(r.sub(r.mul(cc,n[0]),r.mul(ss,n[1])),r.add(r.mul(ss,n[0]),r.mul(cc,n[1])))
                r.equal(f'fixed wall {w} edge {e} normal coordinate {coordinate}',rotated[coordinate],nw[coordinate])
    nx,ny,z,w,omega,eta,vx,vy,wx,wy=[r.var(n) for n in ['nx','ny','z','w','omega','eta','vx','vy','wx','wy']]
    # Independent differentiation of the owner supporting-plane gap.
    gap=r.sub(r.add(r.mul(nx,r.add(z,x)),r.mul(ny,r.add(w,y))),radius)
    velocities={'nx':r.neg(r.mul(omega,ny)),'ny':r.mul(omega,nx),
        'z':r.sub(vx,wx),'w':r.sub(vy,wy),'x':r.neg(r.mul(eta,y)),'y':r.mul(eta,x)}
    derivative=r.add(*[r.mul(r.derivative(gap,name),vel) for name,vel in velocities.items()])
    coefficient_form=r.add(r.mul(nx,r.sub(vx,wx)),r.mul(ny,r.sub(vy,wy)),
        r.mul(omega,r.add(r.mul(r.neg(ny),r.add(z,x)),r.mul(nx,r.add(w,y)))),
        r.neg(r.mul(eta,r.add(r.mul(r.neg(ny),x),r.mul(nx,y)))))
    r.equal('strict descent contact coefficients',derivative,coefficient_form)
    u,z,w,k=[r.var(n) for n in ['u','z','w','k']]
    a1=r.sub(r.add(r.mul(nx,u),r.mul(ny,z)),h)
    a2=r.sub(r.add(r.mul(nx,w),r.mul(ny,u)),h)
    sep=r.sub(r.sub(r.mul(r.add(one,k),u),z),r.mul(k,w))
    weighted=r.add(r.mul(nx,a1),r.mul(r.mul(k,ny),a2),r.mul(r.mul(nx,ny),sep))
    resultant=r.mul(r.add(nx,r.mul(k,ny)),r.sub(r.mul(u,r.add(nx,ny)),h))
    r.equal('corner two-base positive-weight identity',weighted,resultant)
    left=r.sub(r.mul(r.sub(r.mul(u,r.add(nx,ny)),h),r.add(r.mul(u,r.add(nx,ny)),h)),
        r.scale(r.mul(r.mul(u,u),r.sub(r.add(r.mul(nx,nx),r.mul(ny,ny)),one)),2))
    right=r.sub(r.sub(r.scale(r.mul(u,u),2),r.mul(h,h)),r.mul(r.mul(u,u),r.mul(r.sub(nx,ny),r.sub(nx,ny))))
    r.equal('corner normal sum of squares bound',left,right)
    # Feasible construction in [0,S]^2. This is checked with exact radicals.
    side=r.add(h,q);origin=(zero,zero);apex=(q,q)
    corner=[origin,(r.add(q,a),r.sub(q,a)),(r.sub(q,a),r.add(q,a))]
    right_tri=[apex,(side,r.sub(q,half)),(side,r.add(q,half))]
    top_tri=[apex,(r.add(q,half),side),(r.sub(q,half),side)]
    for name,triangle in [('corner',corner),('right',right_tri),('top',top_tri)]:
        for i in range(3):
            edge=r.vector_sub(triangle[(i+1)%3],triangle[i])
            r.equal(f'exact construction {name} edge {i}',r.dot(edge,edge),one)
    normal=(one,one)
    for i in [1,2]:r.equal(f'corner separating line endpoint {i}',r.dot(normal,corner[i]),r.scale(q,2))
    for label,tri in [('right',right_tri),('top',top_tri)]:
        r.equal(f'corner vs {label} apex gap',r.sub(r.dot(normal,tri[0]),r.scale(q,2)),zero)
        r.equal(f'corner vs {label} first base gap',r.sub(r.dot(normal,tri[1]),r.scale(q,2)),r.sub(h,half) if label=='right' else r.add(h,half))
        r.equal(f'corner vs {label} second base gap',r.sub(r.dot(normal,tri[2]),r.scale(q,2)),r.add(h,half) if label=='right' else r.sub(h,half))
    facing=(r.neg(half),h)
    for i,expected in enumerate([zero,r.neg(h),zero]):
        r.equal(f'right vs top right sign {i}',r.dot(facing,r.vector_sub(right_tri[i],apex)),expected)
    for i,expected in enumerate([zero,half,one]):
        r.equal(f'right vs top top sign {i}',r.dot(facing,r.vector_sub(top_tri[i],apex)),expected)
    r.equal('h greater than one half squared margin',r.sub(r.mul(h,h),r.mul(half,half)),r.const(864))
    r.equal('q greater than one half squared margin',r.sub(r.mul(q,q),r.mul(half,half)),r.const(216))
    r.equal('h greater than a squared margin',r.sub(r.mul(h,h),r.mul(a,a)),r.const(1080))
    r.equal('q greater than a squared margin',r.sub(r.mul(q,q),r.mul(a,a)),r.const(432))
    r.equal('three triangle area squared times sixteen',r.scale(r.mul(r.scale(r.mul(h,half),3),r.scale(r.mul(h,half),3)),16),r.const(46656))
    return r.checks

def main():
    start=time.monotonic();mx.set_default_device(mx.gpu)
    ml=verify(Ring('mlx'));mp=verify(Ring('mps'));assert ml==mp
    result={'passed':True,'exact_polynomial_identities':len(ml),'checks':ml,
        'engines':['MLX Metal exact int64 sparse coefficients','PyTorch MPS exact int64 sorted coefficient groups'],
        'root_relations':['h>0, h^2=3/4','q>0, q^2=3/8','a>0, a^2=1/8'],
        'construction':'S=h+q; corner [(0,0),(q+a,q-a),(q-a,q+a)]; right [(q,q),(S,q-1/2),(S,q+1/2)]; top [(q,q),(q+1/2,S),(q-1/2,S)]',
        'construction_has_unit_edges':True,'construction_containment_by_positive_squared_margins':True,
        'construction_disjoint_interiors_by_exact_separating_planes':True,
        'floating_arithmetic_used':False,'cpu_numerical_fallback':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'logical_scope':'Exact formula and construction audit; the complete contact and interval coverage argument is a separate obligation.',
        'elapsed_seconds':time.monotonic()-start}
    (ROOT/'certified_spatial/geometric_semantics_certificate.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2),flush=True)

if __name__=='__main__':main()
