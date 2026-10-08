"""Independent boundary/construction audit in Q(sqrt(2),sqrt(3)).

This uses a four-element algebraic-field basis, rather than the previous
polynomial verifier's separate radical variables. No original geometry,
polynomial or family-certificate code is imported. All algebra runs on GPU.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import json,hashlib,time
from pathlib import Path
import mlx.core as mx
import torch
from independent_geometry_gpu import Verifier,ROOT,load_file

D=768
VARIABLES=['nx','ny','u','z1','z2','c','s','t','x','y']

class FieldPolynomials:
    def __init__(self,backend):
        self.backend=backend;self.checks=[]
        field=self.array([0,1,2,3]);common=field[:,None]&field[None,:]
        self.destination=field[:,None]^field[None,:]
        self.factor=self.where((common&1)!=0,2,1)*self.where((common&2)!=0,3,1)
    def array(self,value):return mx.array(value,dtype=mx.int64) if self.backend=='mlx' else torch.tensor(value,device='mps',dtype=torch.int64)
    def where(self,p,a,b):return mx.where(p,a,b) if self.backend=='mlx' else torch.where(p,a,b)
    def concat(self,pieces,axis=0):return mx.concatenate(pieces,axis=axis) if self.backend=='mlx' else torch.cat(pieces,dim=axis)
    def all(self,p):return bool((mx.all(p) if self.backend=='mlx' else p.all()).item())
    def divide_exact(self,a,b):
        result=a//b;assert self.all(result*b==a),'Field coefficient denominator was not represented exactly'
        return result
    def constant(self,numerator=1,denominator=1,basis=0):
        basis_word=self.array([0,1,2,3])==basis
        coefficients=self.divide_exact(basis_word.to(mx.int64) if False else basis_word.astype(mx.int64) if self.backend=='mlx' else basis_word.to(torch.int64),1)
        coefficients=self.divide_exact(coefficients*self.array(numerator)*D,self.array(denominator))
        return self.array([0]),coefficients[None,:]
    def var(self,name):
        position=self.array([VARIABLES.index(name)])
        return self.array([1])<<(position*3),self.constant()[1]
    def add(self,*p):return self.concat([x[0] for x in p]),self.concat([x[1] for x in p])
    def negative(self,p):return p[0],-p[1]
    def minus(self,a,b):return self.add(a,self.negative(b))
    def times(self,a,b):
        codes=(a[0][:,None]+b[0][None,:]).reshape(-1)
        products=a[1][:,None,:,None]*b[1][None,:,None,:]*self.factor[None,None,:,:]
        values=[]
        for basis in range(4):
            masked=self.where(self.destination[None,None,:,:]==basis,products,0)
            values.append(mx.sum(masked,axis=(2,3)) if self.backend=='mlx' else masked.sum((2,3)))
        coefficients=mx.stack(values,axis=-1) if self.backend=='mlx' else torch.stack(values,dim=-1)
        coefficients=self.divide_exact(coefficients,D).reshape(-1,4)
        return codes,coefficients
    def scaled(self,p,factor):return p[0],p[1]*factor
    def derivative(self,p,name):
        shift=3*VARIABLES.index(name);degree=(p[0]>>shift)&7
        codes=self.where(degree>0,p[0]-(self.array([1])<<shift),0)
        return codes,p[1]*degree[:,None]
    def identity(self,name,a,b):
        codes,values=self.minus(a,b)
        if self.backend=='mlx':
            identical=codes[:,None]==codes[None,:]
            sums=mx.sum(identical[:,:,None].astype(mx.int64)*values[None,:,:],axis=1)
        else:
            order=torch.argsort(codes);sorted_codes=codes[order]
            starts=torch.cat([torch.ones(1,device='mps',dtype=torch.int64),(sorted_codes[1:]!=sorted_codes[:-1]).to(torch.int64)])
            group=torch.cumsum(starts,0)-1
            sums=torch.zeros((codes.numel(),4),device='mps',dtype=torch.int64).scatter_add(0,group[:,None].expand(-1,4),values[order])
        assert self.all(sums==0),name
        self.checks.append(name)
    def dot(self,a,b):return self.add(self.times(a[0],b[0]),self.times(a[1],b[1]))
    def vec_minus(self,a,b):return self.minus(a[0],b[0]),self.minus(a[1],b[1])
    def turn(self,a):return self.negative(a[1]),a[0]

def algebra(r):
    zero,one,half=r.constant(0),r.constant(),r.constant(1,2)
    root3=r.constant(1,1,2);h=r.constant(1,2,2);q=r.constant(1,4,3);a=r.constant(1,4,1);inradius=r.constant(1,6,2)
    r.identity('positive root h squared',r.times(h,h),r.constant(3,4))
    r.identity('positive root q squared',r.times(q,q),r.constant(3,8))
    r.identity('positive root a squared',r.times(a,a),r.constant(1,8))
    r.identity('exact h/sqrt(2) equals q',r.times(q,r.constant(1,1,1)),h)
    body=[(half,r.negative(inradius)),(zero,r.scaled(inradius,2)),(r.negative(half),r.negative(inradius))]
    normals=[(h,half),(r.negative(h),half),(zero,r.negative(one))]
    radials=[(h,r.negative(half)),(zero,one),(r.negative(h),r.negative(half))]
    directions=[r.vec_minus(body[(i+1)%3],body[i]) for i in range(3)]
    for i in range(3):
        r.identity(f'canonical edge {i} unit',r.dot(directions[i],directions[i]),one)
        r.identity(f'canonical neighbouring edges {i} dot',r.dot(directions[i],directions[(i+1)%3]),r.negative(half))
        r.identity(f'outward normal {i} unit',r.dot(normals[i],normals[i]),one)
        r.identity(f'radial normal {i} unit',r.dot(radials[i],radials[i]),one)
        for j in range(3):
            edge_support=inradius if j in [i,(i+1)%3] else r.scaled(inradius,-2)
            radial_support=r.scaled(inradius,2) if j==i else r.negative(inradius)
            r.identity(f'edge {i} vertex {j} projection',r.dot(normals[i],body[j]),edge_support)
            r.identity(f'radial {i} vertex {j} projection',r.dot(radials[i],body[j]),radial_support)
        r.identity(f'edge {i} tangent beginning',r.dot(r.turn(normals[i]),body[i]),r.negative(half))
        r.identity(f'edge {i} tangent end',r.dot(r.turn(normals[i]),body[(i+1)%3]),half)
        r.identity(f'radial {i} tangent value zero',r.dot(r.turn(radials[i]),body[i]),zero)
        r.identity(f'radial {i} is sum of incident normals',r.add(normals[i][0],normals[(i+2)%3][0]),radials[i][0])
        r.identity(f'radial {i} incident normal y',r.add(normals[i][1],normals[(i+2)%3][1]),radials[i][1])
    nx,ny,u,z1,z2=[r.var(n) for n in ['nx','ny','u','z1','z2']]
    A1=r.minus(r.add(r.times(nx,u),r.times(ny,z1)),h)
    A2=r.minus(r.add(r.times(nx,z2),r.times(ny,u)),h)
    left=r.minus(r.minus(r.times(r.add(one,root3),u),z1),r.times(root3,z2))
    right=r.minus(r.minus(r.times(r.add(one,root3),u),r.times(root3,z1)),z2)
    gap=r.minus(r.times(u,r.add(nx,ny)),h)
    dual_left=r.add(r.times(nx,A1),r.times(r.times(root3,ny),A2),r.times(r.times(nx,ny),left))
    dual_right=r.add(r.times(r.times(root3,nx),A1),r.times(ny,A2),r.times(r.times(nx,ny),right))
    r.identity('linear elimination dual for top facing edge',dual_left,r.times(r.add(nx,r.times(root3,ny)),gap))
    r.identity('linear elimination dual for right facing edge',dual_right,r.times(r.add(r.times(root3,nx),ny),gap))
    sum_normal=r.add(nx,ny);difference=r.minus(nx,ny)
    r.identity('unit-normal Cauchy identity',r.add(r.times(sum_normal,sum_normal),r.times(difference,difference)),r.scaled(r.add(r.times(nx,nx),r.times(ny,ny)),2))
    # Direct facing-edge planes, without assuming that the two apices coincide.
    P=(u,z1);T=(z2,u)
    facing_right=(r.negative(half),h);facing_top=(h,r.negative(half))
    r.identity('top-facing apex inequality',r.scaled(r.dot(facing_top,r.vec_minus(P,T)),2),left)
    r.identity('right-facing apex inequality',r.scaled(r.dot(facing_right,r.vec_minus(T,P)),2),right)
    c,s,t,x,y=[r.var(n) for n in ['c','s','t','x','y']]
    tt=r.times(t,t);cn=r.minus(one,tt);sn=r.scaled(t,2);den=r.add(one,tt)
    r.identity('complete rational rotation is unit',r.add(r.times(cn,cn),r.times(sn,sn)),r.times(den,den))
    xn=r.minus(r.times(cn,x),r.times(sn,y));yn=r.add(r.times(sn,x),r.times(cn,y))
    dx=r.minus(r.times(r.derivative(xn,'t'),den),r.times(xn,r.derivative(den,'t')))
    dy=r.minus(r.times(r.derivative(yn,'t'),den),r.times(yn,r.derivative(den,'t')))
    r.identity('projection X monotonic derivative',dx,r.scaled(yn,-2))
    r.identity('projection Y monotonic derivative',dy,r.scaled(xn,2))
    S=r.add(h,q);origin=(zero,zero);apex=(q,q)
    triangles=[('corner',[origin,(r.add(q,a),r.minus(q,a)),(r.minus(q,a),r.add(q,a))]),
        ('right',[apex,(S,r.minus(q,half)),(S,r.add(q,half))]),
        ('top',[apex,(r.add(q,half),S),(r.minus(q,half),S)])]
    for name,vertices in triangles:
        for i in range(3):
            d=r.vec_minus(vertices[(i+1)%3],vertices[i]);r.identity(f'exact {name} edge {i}',r.dot(d,d),one)
    for name,vertices in triangles[1:]:
        r.identity(f'{name} apex corner-plane gap',r.minus(r.add(*vertices[0]),r.scaled(q,2)),zero)
        for i in [1,2]:
            expected=r.minus(h,half) if (name=='right' and i==1) or (name=='top' and i==2) else r.add(h,half)
            r.identity(f'{name} corner-plane base gap {i}',r.minus(r.add(*vertices[i]),r.scaled(q,2)),expected)
    for i in [1,2]:r.identity(f'corner edge plane {i}',r.add(*triangles[0][1][i]),r.scaled(q,2))
    for i,expected in enumerate([zero,r.negative(h),zero]):r.identity(f'right-top separating right gap {i}',r.dot(facing_right,r.vec_minus(triangles[1][1][i],apex)),expected)
    for i,expected in enumerate([zero,half,one]):r.identity(f'right-top separating top gap {i}',r.dot(facing_right,r.vec_minus(triangles[2][1][i],apex)),expected)
    for label,p,b,expected in [('h>half',h,half,r.constant(1,2)),('q>half',q,half,r.constant(1,8)),
        ('h>a',h,a,r.constant(5,8)),('q>a',q,a,r.constant(1,4)),('h>q',h,q,r.constant(3,8))]:
        r.identity('positive squared margin '+label,r.minus(r.times(p,p),r.times(b,b)),expected)
    return r.checks

WINDOW=r'''
Range h=divide_small(Range{roots[0],roots[1]},2);
Range side={limits[0],limits[1]},u=subtract(side,h);
bool unknown=false;
out[0]=2*side.lower>roots[1]+UNIT;
out[1]=side.upper<roots[0];
out[2]=u.lower>0;
out[3]=subtract(h,u).lower>0;
out[4]=subtract(multiply(Range{roots[0],roots[1]},u,unknown),subtract(side,value(UNIT/2))).lower>0;
out[5]=unknown?0:1;
'''

def classification(v):
    words=v.masks[v.triples];flags=(words[:,:,None]>>mx.array([0,4,8],dtype=mx.uint32))&15
    bases=(flags&mx.roll(flags,-1,axis=2))
    base_word=bases[:,:,0]|bases[:,:,1]|bases[:,:,2]
    covered=mx.zeros((words.shape[0],),dtype=mx.bool_)
    for corner in [5,6,9,10]:
        has=mx.any((flags&corner)==corner,axis=2)
        for i,j,k in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
            # Opposite horizontal/vertical walls, derived by swapping adjacent
            # bits within each opposing pair.
            x=(corner&3);y=(corner&12)
            opposite_x=((x<<1)|(x>>1))&3;opposite_y=((y<<1)|(y>>1))&12
            covered|=has[:,i]&((base_word[:,j]&opposite_x)!=0)&((base_word[:,k]&opposite_y)!=0)
    expected=load_file('/Volumes/SquarePackingProof/SquarePacking/runs/boundary_seed_8e6474bd/covered_wall_families.npy')
    assert mx.all(covered==expected).item()
    ids=load_file(ROOT/'certified_spatial/spatial_candidates.npy').astype(mx.int32)
    lookup=load_file(ROOT/'certified_spatial/rotation_critical_lookup.npy')
    total=mx.sum(((lookup[ids]>0)&covered[ids//50653]).astype(mx.int32)).item()
    assert total==8257;return total

def main():
    start=time.monotonic();ml=algebra(FieldPolynomials('mlx'));mp=algebra(FieldPolynomials('mps'));assert ml==mp
    v=Verifier(30);seed=json.loads(Path('/Volumes/SquarePackingProof/SquarePacking/runs/boundary_seed_8e6474bd/boundary_seed_certificate.json').read_text())
    kernel=mx.fast.metal_kernel(name='independent_boundary_window',input_names=['roots','limits'],output_names=['out'],header=v.header,source=WINDOW)
    conditions=kernel(inputs=[v.roots,mx.array([seed['side_lower_fixed'],seed['side_upper_fixed']],dtype=mx.int64)],
        grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(6,)],output_dtypes=[mx.int32])[0]
    assert mx.all(conditions==1).item();profiles=classification(v)
    result={'passed':True,'field_basis':['1','sqrt(2)','sqrt(3)','sqrt(6)'],'exact_identities_checked':len(ml),'checks':ml,
        'both_MLX_and_MPS_coefficients_passed':True,'alternative_derivation':'Eliminate the independent apex coordinates using both possible facing-edge inequalities; unit-normal Cauchy gives u>=h/sqrt(2).',
        'boundary_window_and_common_corner_cone_checked':True,'all_analytic_wall_classifications_match':True,
        'analytic_profiles_checked':profiles,'exact_construction_unit_edges_containment_and_disjoint_interiors_checked':True,
        'imports_original_boundary_or_geometry_algebra':False,'cpu_numerical_fallback':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_seconds':time.monotonic()-start}
    (ROOT/'proof_n3/independent_verification/boundary_and_construction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:value for k,value in result.items() if k!='checks'},indent=2),flush=True)

if __name__=='__main__':main()
