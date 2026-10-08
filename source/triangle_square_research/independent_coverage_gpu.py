"""Independent finite coverage audit, using a 15-degree endpoint relation join.

No catalogue/search geometry functions are imported. Continuous-angle
completeness is justified by the integer difference-bound flooring argument;
this script verifies all finite memberships, symmetries, inherited contacts,
geometric exclusions and component-criticality decisions from scratch on MPS.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import ast,struct,json,hashlib,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
D=torch.device('mps')

def load(path):
    data=Path(path).read_bytes();assert data[:6]==b'\x93NUMPY'
    size=2 if data[6]==1 else 4
    length=struct.unpack('<H' if size==2 else '<I',data[8:8+size])[0];offset=8+size+length
    metadata=ast.literal_eval(data[8+size:offset].decode('latin1'))
    dtype=torch.uint8 if metadata['descr'] in ['|u1','|b1'] else torch.int32
    assert metadata['descr'] in ['|u1','|b1','<u4','<i4']
    return torch.frombuffer(bytearray(data),dtype=dtype,offset=offset).to(D).reshape(metadata['shape']).to(torch.int32)

def support():
    # Exact ray directions of the canonical unit triangle, in units of 15°.
    normals=torch.tensor([12,0,18,6],device=D,dtype=torch.int32)
    radial=torch.tensor([22,6,14],device=D,dtype=torch.int32)
    turn=torch.arange(24,device=D,dtype=torch.int32)
    delta=torch.remainder(normals[None,:,None]-radial[:,None,None]-turn[None,None,:],24)
    return torch.minimum(delta,24-delta)<=4

def wall_catalogue():
    words=torch.arange(4096,device=D,dtype=torch.int32)
    active=((words[:,None]>>torch.arange(12,device=D,dtype=torch.int32))&1).reshape(4096,3,4).bool()
    touched=active.any(1)
    structural=~((touched[:,0]&touched[:,1])|(touched[:,2]&touched[:,3]))
    possible=((~active[:,:,:,None])|support()[None,:,:,:]).all((1,2)).any(1)
    rotated=((words<<4)&4095)|(words>>8);rotated_twice=((words<<8)&4095)|(words>>4)
    canonical=words==torch.minimum(words,torch.minimum(rotated,rotated_twice))
    table=words[structural&possible&canonical]
    assert torch.equal(table,load(ROOT/'contact_catalog/wall_masks.npy'))
    matrices=torch.tensor([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]],
        [[-1,0],[0,1]],[[0,-1],[-1,0]],[[1,0],[0,-1]],[[0,1],[1,0]]],device=D,dtype=torch.int32)
    axes=torch.tensor([[-1,0],[1,0],[0,-1],[0,1]],device=D,dtype=torch.int32)
    image=(matrices[:,None,:,:]*axes[None,:,None,:]).sum(-1)
    mapping=(image[:,:,None,:]==axes[None,None,:,:]).all(-1).to(torch.int32).argmax(-1)
    determinant=matrices[:,0,0]*matrices[:,1,1]-matrices[:,0,1]*matrices[:,1,0]
    flag=((table[:,None]>>torch.arange(12,device=D,dtype=torch.int32))&1)
    transformations=[]
    for symmetry in range(8):
        vertex=torch.arange(3,device=D,dtype=torch.int32)
        vertex=torch.where(determinant[symmetry]<0,(-vertex)%3,vertex)
        position=(4*vertex[:,None]+mapping[symmetry][None,:]).reshape(12)
        w=(flag*(torch.ones(12,device=D,dtype=torch.int32)<<position)[None,:]).sum(1).to(torch.int32)
        ra=((w<<4)&4095)|(w>>8);rb=((w<<8)&4095)|(w>>4);w=torch.minimum(w,torch.minimum(ra,rb))
        ident=(table[None,:]<w[:,None]).sum(1).to(torch.int32)
        assert torch.equal(table[ident.to(torch.int64)],w);transformations.append(ident)
    index=torch.arange(table.numel()**3,device=D,dtype=torch.int32)
    triples=torch.stack([index//(table.numel()**2),(index//table.numel())%table.numel(),index%table.numel()],1)
    orbit=[]
    for transform in transformations:
        transformed=transform[triples.to(torch.int64)].sort(1).values
        orbit.append(transformed[:,0]*(table.numel()**2)+transformed[:,1]*table.numel()+transformed[:,2])
    representative=index==torch.stack(orbit).amin(0)
    vflags=(table[:,None]>>(4*torch.arange(3,device=D,dtype=torch.int32)))&15
    is_corner=((vflags&3)!=0)&((vflags&12)!=0)
    corner=torch.where(is_corner,((vflags>>1)&1)+2*((vflags>>3)&1),99).amin(1)
    labels=corner[triples.to(torch.int64)]
    duplicate=((labels[:,0]==labels[:,1])&(labels[:,0]<99))|((labels[:,0]==labels[:,2])&(labels[:,0]<99))|((labels[:,1]==labels[:,2])&(labels[:,1]<99))
    actual=triples[representative&~duplicate]
    assert torch.equal(actual,load(ROOT/'contact_catalog/wall_triplets.npy'))
    return table,actual

def pair_relations():
    angle=torch.arange(24,device=D,dtype=torch.int32);a=angle[None,:,None];b=angle[None,None,:]
    radial=torch.tensor([22,6,14],device=D,dtype=torch.int32);normal=torch.tensor([2,10,18],device=D,dtype=torch.int32)
    code=torch.arange(37,device=D,dtype=torch.int32)
    def circular(delta):
        u=delta%24;return torch.minimum(u,24-u)
    vv=(code-1).clamp(0,8);ve=(code-10).clamp(0,17);ee=(code-28).clamp(0,8)
    vertex_vertex=circular(a+radial[(vv//3).to(torch.int64),None,None]-b-radial[(vv%3).to(torch.int64),None,None])>=4
    e=(ve%9)//3;v=ve%3
    left=circular(a+normal[e.to(torch.int64),None,None]+12-b-radial[v.to(torch.int64),None,None])<=4
    right=circular(b+normal[e.to(torch.int64),None,None]+12-a-radial[v.to(torch.int64),None,None])<=4
    vertex_edge=torch.where((ve//9==0)[:,None,None],left,right)
    edge_edge=(a+normal[(ee//3).to(torch.int64),None,None]-b-normal[(ee%3).to(torch.int64),None,None])%24==12
    permitted=torch.where((code==0)[:,None,None],True,torch.where((code<10)[:,None,None],vertex_vertex,torch.where((code<28)[:,None,None],vertex_edge,edge_edge)))
    bits=torch.ones(24,device=D,dtype=torch.int32)<<angle
    return (permitted.to(torch.int32)*bits[None,None,:]).sum(-1).to(torch.int32),bits

def decisions(ids,masks,triples,relations,bits):
    wi=ids//50653;mode=ids-wi*50653
    assert ((mode>=0)&(mode<50653)).all().item()
    codes=torch.stack([mode//1369,(mode//37)%37,mode%37],1)
    assert torch.equal(codes[:,0]*1369+codes[:,1]*37+codes[:,2],mode)
    words=masks[triples[wi.to(torch.int64)].to(torch.int64)]
    flags=((words[:,:,None]>>(4*torch.arange(3,device=D,dtype=torch.int32)))&15).reshape(-1,9)
    reach=torch.eye(9,device=D,dtype=torch.bool)[None,:,:].expand(ids.numel(),-1,-1).clone()
    row=torch.arange(ids.numel(),device=D,dtype=torch.int64)
    pairs=[(0,1),(0,2),(1,2)]
    for pair,(i,j) in enumerate(pairs):
        ident=(codes[:,pair]-1).clamp(0,8);a=3*i+ident//3;b=3*j+ident%3
        vv=(codes[:,pair]>0)&(codes[:,pair]<10)
        reach[row,a.to(torch.int64),b.to(torch.int64)]=vv;reach[row,b.to(torch.int64),a.to(torch.int64)]=vv
    for _ in range(3):reach=torch.bmm(reach.to(torch.float32),reach.to(torch.float32))>0
    inherited=torch.zeros_like(flags)
    for vertex in range(9):inherited|=torch.where(reach[:,:,vertex],flags[:,vertex,None],0)
    impossible=torch.zeros(ids.numel(),device=D,dtype=torch.bool)
    for i in range(3):impossible|=reach[:,3*i:3*i+3,3*i:3*i+3].to(torch.int32).sum((1,2))>3
    per_piece=inherited.reshape(-1,3,3);combined=per_piece[:,:,0]|per_piece[:,:,1]|per_piece[:,:,2]
    impossible|=(((combined&3)==3)|((combined&12)==12)).any(1)
    corner=((inherited&3)!=0)&((inherited&12)!=0)
    for vertex in range(9):
        other=reach[:,vertex,:].clone();other[:,3*(vertex//3):3*(vertex//3)+3]=False
        impossible|=corner[:,vertex]&other.any(1)
    wallbits=(per_piece[:,:,:,None]>>torch.arange(4,device=D,dtype=torch.int32))&1
    allowed=((wallbits[:,:,:,:,None]==0)|support()[None,None,:,:,:]).all((2,3))
    angular_masks=(allowed.to(torch.int32)*bits[None,None,:]).sum(-1).to(torch.int32)
    ab=angular_masks[:,1,None]&relations[codes[:,0].to(torch.int64)]
    ac=angular_masks[:,2,None]&relations[codes[:,1].to(torch.int64)]
    bc=relations[codes[:,2].to(torch.int64)]
    found=((ac[:,:,None]&bc[:,None,:])!=0)&((ab[:,:,None]&bits[None,None,:])!=0)
    found&=(angular_masks[:,0,None,None]&bits[None,:,None])!=0
    angular=(~impossible)&found.any((1,2))
    # Components of whole triangles, distinct from shared-vertex components.
    graph=torch.eye(3,device=D,dtype=torch.bool)[None,:,:].expand(ids.numel(),-1,-1).clone()
    for pair,(i,j) in enumerate(pairs):graph[:,i,j]=codes[:,pair]>0;graph[:,j,i]=codes[:,pair]>0
    for _ in range(2):graph=torch.bmm(graph.to(torch.float32),graph.to(torch.float32))>0
    original=flags.reshape(-1,3,3)
    counts=torch.stack([((original&(1<<w))!=0).sum(2) for w in range(4)],2)
    component=torch.zeros((ids.numel(),3,4),device=D,dtype=torch.int32)
    for i in range(3):component+=torch.where(graph[:,:,i,None],counts[:,i,None,:],0).to(torch.int32)
    horizontal=(component[:,:,0]>0)&(component[:,:,1]>0);vertical=(component[:,:,2]>0)&(component[:,:,3]>0)
    basic=(horizontal|vertical).any(1)
    critical=((horizontal&vertical)|(horizontal&((component[:,:,0]>1)|(component[:,:,1]>1)))|
        (vertical&((component[:,:,2]>1)|(component[:,:,3]>1)))).any(1)
    edge_wall=torch.zeros((ids.numel(),3,3),device=D,dtype=torch.bool)
    full_wall=torch.zeros((ids.numel(),3,4),device=D,dtype=torch.bool)
    for e in range(3):
        both=original[:,:,e]&original[:,:,(e+1)%3]
        edge_wall[:,:,e]=both!=0
        for w in range(4):full_wall[:,:,w]|=(both&(1<<w))!=0
    bad=((full_wall[:,0]&full_wall[:,1])|(full_wall[:,0]&full_wall[:,2])|(full_wall[:,1]&full_wall[:,2])).any(1)
    for pair,(i,j) in enumerate(pairs):
        c=codes[:,pair];ve=(c-10).clamp(0,17);e=(ve%9)//3
        owner_edge=torch.where(ve//9==0,edge_wall[row,i,e.to(torch.int64)],edge_wall[row,j,e.to(torch.int64)])
        bad|=(c>=10)&(c<28)&owner_edge
        ee=(c-28).clamp(0,8)
        bad|=(c>=28)&(edge_wall[row,i,(ee//3).to(torch.int64)]|edge_wall[row,j,(ee%3).to(torch.int64)])
    spatial=angular&basic&~bad
    return angular,spatial,spatial&critical

def main():
    start=time.monotonic();last=start;masks,triples=wall_catalogue();relations,bits=pair_relations()
    total=triples.shape[0]*50653
    summary=json.loads((ROOT/'contact_catalog/screening_summary.json').read_text())
    expected=torch.zeros(total,device=D,dtype=torch.bool)
    for part in summary['parts']:
        if part['ids_file']:expected[load(part['ids_file']).to(torch.int64)]=True
    spatial_expected=torch.zeros_like(expected);spatial_expected[load(ROOT/'certified_spatial/spatial_candidates.npy').to(torch.int64)]=True
    lookup=load(ROOT/'certified_spatial/rotation_critical_lookup.npy')
    count_a=count_s=count_c=0
    for first in range(0,total,8192):
        ids=torch.arange(first,min(first+8192,total),device=D,dtype=torch.int32)
        angular,spatial,critical=decisions(ids,masks,triples,relations,bits)
        end=first+ids.numel()
        assert torch.equal(angular,expected[first:end]),('angular',first)
        assert torch.equal(spatial,spatial_expected[first:end]),('spatial',first)
        assert torch.equal(critical,lookup[first:end]>0),('critical',first)
        count_a+=angular.sum().item();count_s+=spatial.sum().item();count_c+=critical.sum().item()
        if time.monotonic()-last>10:print('Independent 15-degree coverage cases',end,'/',total,flush=True);last=time.monotonic()
    result={'passed':True,'wall_subsets_checked':4096,'wall_types':masks.numel(),'wall_triple_representatives':triples.shape[0],
        'contact_cases_checked':total,'angular_survivors':count_a,'spatial_profiles':count_s,'rotation_critical_profiles':count_c,
        'endpoint_grid_degrees':15,'all_angular_spatial_and_critical_memberships_match':True,
        'arbitrary_real_angles_argument':'Flooring unwrapped real angular variables preserves integer difference bounds and equalities. The endpoint grid is an angular necessity test; actual rotations remain continuous in two complete rational charts.',
        'degenerate_contacts_argument':'Disjoint convex polygon boundaries meet in no point, one point, or a segment; VV, proper VE and positive EE cover these, including limiting cases.',
        'flexible_singular_profiles_removed_by_rank':False,'imports_original_catalogue_geometry':False,
        'cpu_numerical_fallback':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-start}
    (ROOT/'proof_n3/independent_verification/coverage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
