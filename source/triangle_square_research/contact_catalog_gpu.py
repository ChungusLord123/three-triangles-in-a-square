"""Complete finite contact-requirement catalogue and exact angular screening.

Numerical/discrete computation runs on Metal. CPU handles code construction,
metadata and file I/O. The catalogue covers contact requirements; surviving
angular patterns are NOT spatial solutions or a global optimality proof.
"""
import json,time,argparse
from pathlib import Path
import mlx.core as mx
from n3_branches_gpu import ROOT

mx.set_default_device(mx.gpu)
if not mx.metal.is_available():raise RuntimeError('Metal GPU required')
OUT=ROOT/'contact_catalog';OUT.mkdir(exist_ok=True)
WALL_NORMAL_30=mx.array([6,0,9,3],dtype=mx.int32) # left,right,bottom,top
VERTEX_RADIAL_30=mx.array([11,3,7],dtype=mx.int32)
EDGE_NORMAL_30=mx.array([1,5,9],dtype=mx.int32)
WALL_PERMUTATIONS=[[0,1,2,3],[2,3,1,0],[1,0,3,2],[3,2,0,1],
                   [1,0,2,3],[3,2,1,0],[0,1,3,2],[2,3,0,1]]

def cyclic_min(m):
 a=((m<<4)&4095)|(m>>8)
 b=((m<<8)&4095)|(m>>4)
 return mx.minimum(m,mx.minimum(a,b))

def wall_catalog():
 masks=mx.arange(4096,dtype=mx.uint32)
 bits=((masks[:,None]>>mx.arange(12,dtype=mx.uint32)[None,:])&1).reshape(4096,3,4)
 touched=mx.any(bits>0,axis=1)
 no_opposites=~((touched[:,0]&touched[:,1])|(touched[:,2]&touched[:,3]))
 max_two=mx.sum(touched.astype(mx.int32),axis=1)<=2
 # Every wall contact must be a support feature of the triangle.
 ang=mx.arange(12,dtype=mx.int32)
 delta=mx.remainder(WALL_NORMAL_30[None,:,None]-VERTEX_RADIAL_30[:,None,None]-ang[None,None,:]+6,12)-6
 support=mx.abs(delta)<=2
 angular=mx.all((bits[:,:,:,None]==0)|support[None,:,:,:],axis=(1,2))
 possible=mx.any(angular,axis=1)&no_opposites&max_two
 canonical=(masks==cyclic_min(masks))&possible
 number=mx.sum(canonical.astype(mx.int32));mx.eval(number)
 index=mx.argsort(mx.where(canonical,masks,mx.array(999999,dtype=mx.uint32)))[:number.item()]
 wall_masks=masks[index];allowed=angular[index]
 # Square D4 transformations, followed by legal cyclic vertex relabelling.
 base_bits=((wall_masks[:,None]>>mx.arange(12,dtype=mx.uint32)[None,:])&1)
 transforms=[]
 for d,perm in enumerate(WALL_PERMUTATIONS):
  newbits=[]
  for v in range(3):
   for w in range(4):
    vv=(-v)%3 if d>=4 else v
    newbits.append(vv*4+perm[w])
  weights=mx.left_shift(mx.array(1,dtype=mx.uint32),mx.array(newbits,dtype=mx.uint32))
  transformed=cyclic_min(mx.sum(base_bits*weights[None,:],axis=1))
  ids=mx.sum((wall_masks[None,:]<transformed[:,None]).astype(mx.int32),axis=1)
  mx.eval(ids,transformed)
  assert mx.all(wall_masks[ids]==transformed).item()
  transforms.append(ids)
 transform_table=mx.stack(transforms)
 w=wall_masks.size
 ix=mx.arange(w*w*w,dtype=mx.int32)
 triplets=mx.stack([ix//(w*w),(ix//w)%w,ix%w],axis=1)
 candidates=[]
 for d in range(8):
  t=mx.sort(transform_table[d][triplets],axis=1)
  candidates.append(t[:,0]*w*w+t[:,1]*w+t[:,2])
 canon=mx.min(mx.stack(candidates),axis=0)
 reps=ix==canon
 # No two triangle vertices may occupy the same square corner: two 60-degree
 # interior sectors cannot be disjoint inside a 90-degree corner.
 bbits=bits[index]
 corner=(bbits[:,:,0]|bbits[:,:,1])&(bbits[:,:,2]|bbits[:,:,3])
 cornercode=mx.where(corner,bbits[:,:,1]+2*bbits[:,:,3],mx.array(99,dtype=mx.uint32))
 corner_tri=mx.min(cornercode,axis=1)
 c=corner_tri[triplets]
 duplicate=((c[:,0]==c[:,1])&(c[:,0]<99))|((c[:,0]==c[:,2])&(c[:,0]<99))|((c[:,1]==c[:,2])&(c[:,1]<99))
 keep=reps&~duplicate
 count=mx.sum(keep.astype(mx.int32));sym_count=mx.sum(reps.astype(mx.int32));mx.eval(count,sym_count)
 selected=mx.argsort(mx.where(keep,ix,99999999))[:count.item()]
 rep_triplets=triplets[selected]
 mx.eval(wall_masks,allowed,rep_triplets,transform_table)
 mx.save(str(OUT/'wall_masks.npy'),wall_masks)
 mx.save(str(OUT/'wall_angles.npy'),allowed)
 mx.save(str(OUT/'wall_triplets.npy'),rep_triplets)
 mx.save(str(OUT/'wall_symmetry_table.npy'),transform_table)
 wall_rows=[]
 wall_names=['left','right','bottom','top']
 contact_lists=bbits.tolist()
 for k,(mask,angles,cbits) in enumerate(zip(wall_masks.tolist(),allowed.tolist(),contact_lists)):
  wall_rows.append({'id':k,'mask':mask,'requirements':[{'vertex':v,'wall':wall_names[ww]}
     for v in range(3) for ww in range(4) if cbits[v][ww]],
     'allowed_support_angles_30':[j for j,a in enumerate(angles) if a]})
 summary={'wall_types_up_to_cyclic_vertex_labels':w,
          'wall_triple_representatives_before_corner_pruning':sym_count.item(),
          'wall_triple_representatives_after_corner_pruning':count.item(),
          'wall_contact_rows':wall_rows,'wall_triplets':rep_triplets.tolist(),
          'coverage_reason':'For s>1, a unit triangle cannot touch opposite walls. Every touched wall must support one or two triangle vertices. Support-cone endpoints are integer multiples of 30 degrees; feasibility of these angular requirements is completely tested on that endpoint grid.',
          'contact_modes_per_triangle_pair':37,
          'pair_mode_encoding':{'0':'no required contact','1..9':'vertex-vertex: (code-1)//3, (code-1)%3',
             '10..27':'vertex in relative interior of edge: owner=(code-10)//9, edge=((code-10)%9)//3, vertex=(code-10)%3; endpoint contacts are covered by vertex-vertex cases',
             '28..36':'edge-edge positive-length overlap: (code-28)//3, (code-28)%3; zero-length overlap is covered by vertex-vertex cases; coincident full edges remain included'},
          'not_an_optimality_proof':True,'gpu':mx.device_info()['device_name'],
          'cpu_numerical_fallback':False}
 (OUT/'catalogue.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k not in ['wall_contact_rows','wall_triplets']},indent=2),flush=True)
 return summary

SOURCE=r'''
uint index = thread_position_in_grid.x;
if (index >= total[0]) return;
uint t = first[0] + index;
uint wi = t / 50653u;
uint mode = t % 50653u;
uint pc[3] = {mode/1369u,(mode/37u)%37u,mode%37u};
uint wids[3] = {triples[3*wi],triples[3*wi+1],triples[3*wi+2]};
uint walls[9];
for(uint i=0;i<3;i++) {
 uint mask=wmasks[wids[i]];
 for(uint v=0;v<3;v++) walls[3*i+v]=(mask>>(4*v))&15u;
}
uint closure[9];
for(uint i=0;i<9;i++) closure[i]=1u<<i;
uint li[3]={0,0,1},ri[3]={1,2,2};
for(uint p=0;p<3;p++) if(pc[p]>0 && pc[p]<10) {
 uint z=pc[p]-1;
 uint a=3*li[p]+z/3,b=3*ri[p]+z%3;
 closure[a]|=1u<<b;closure[b]|=1u<<a;
}
for(uint k=0;k<9;k++) for(uint i=0;i<9;i++)
 if(closure[i]&(1u<<k)) closure[i]|=closure[k];
uint reason=0;
uint inherited[9];
for(uint i=0;i<9;i++) {
 uint mask=0;
 for(uint j=0;j<9;j++) if(closure[i]&(1u<<j)) mask|=walls[j];
 inherited[i]=mask;
 uint triBits=closure[i]&(7u<<(3*(i/3)));
 if((triBits&(triBits-1u))!=0) reason|=1u;
 if(((mask&3u)==3u)||((mask&12u)==12u)) reason|=2u;
 bool corner=((mask&3u)!=0)&&((mask&12u)!=0);
 bool another=(closure[i]&(~(7u<<(3*(i/3)))))!=0;
 if(corner&&another) reason|=4u;
}
for(uint i=0;i<3;i++) {
 uint flags=inherited[3*i]|inherited[3*i+1]|inherited[3*i+2];
 if(((flags&3u)==3u)||((flags&12u)==12u)) reason|=2u;
}
uint allowed[3]={4095u,4095u,4095u};
for(uint i=0;i<3;i++) for(uint v=0;v<3;v++) for(uint w=0;w<4;w++)
 if(inherited[3*i+v]&(1u<<w)) allowed[i]&=support[4*v+w];
if(!allowed[0]||!allowed[1]||!allowed[2]) reason|=8u;
// Exact feasibility of necessary angular contact conditions. The cones have
// integer 30-degree endpoints and only integer difference bounds/equalities.
// Flooring a real angular solution preserves one of the circular branches.
int beta[3]={11,3,7};
int gamma[3]={1,5,9};
bool found=false;
uint witness=65535u;
if(reason==0) {
 for(int a=0;a<12 && !found;a++) if(allowed[0]&(1u<<a))
 for(int b=0;b<12 && !found;b++) if(allowed[1]&(1u<<b))
 for(int c=0;c<12 && !found;c++) if(allowed[2]&(1u<<c)) {
  int angle[3]={a,b,c}; bool ok=true;
  for(uint p=0;p<3 && ok;p++) {
   uint code=pc[p];int x=angle[li[p]],y=angle[ri[p]];
   if(code>0 && code<10) {
    uint z=code-1;int d=(x+beta[z/3]-y-beta[z%3]+48)%12;
    if(d>6)d=12-d;if(d<2)ok=false;
   } else if(code>=10 && code<28) {
    uint z=code-10;uint owner=z/9,e=(z%9)/3,v=z%3;
    int d=(owner==0 ? x+gamma[e]+6-y-beta[v] : y+gamma[e]+6-x-beta[v]);
    d=(d+48)%12;if(d>6)d=12-d;if(d>2)ok=false;
   } else if(code>=28) {
    uint z=code-28;int d=(x+gamma[z/3]-y-gamma[z%3]+48)%12;
    if(d!=6)ok=false;
   }
  }
  if(ok) {found=true;witness=uint(a+12*b+144*c);}
 }
 if(!found) reason|=8u;
}
reasons[index]=reason;
angles[index]=witness;
'''

def screen_catalogue(chunk=131072,seconds=180):
 start=time.monotonic()
 catalogue=json.loads((OUT/'catalogue.json').read_text())
 masks=mx.load(str(OUT/'wall_masks.npy')).astype(mx.uint32)
 triples=mx.load(str(OUT/'wall_triplets.npy')).astype(mx.uint32)
 # Exact integer support cones, twelve possible endpoint orientations.
 angles=mx.arange(12,dtype=mx.int32)
 diff=mx.remainder(WALL_NORMAL_30[None,:,None]-VERTEX_RADIAL_30[:,None,None]-angles[None,None,:]+6,12)-6
 sup=mx.sum((mx.abs(diff)<=2).astype(mx.uint32)*(mx.array(1,dtype=mx.uint32)<<mx.arange(12,dtype=mx.uint32)),axis=-1).reshape(12)
 kernel=mx.fast.metal_kernel(name='contact_screen_exact_integer',
    input_names=['wmasks','triples','support','first','total'],
    output_names=['reasons','angles'],source=SOURCE)
 count=triples.shape[0]*50653
 counts=mx.zeros((16,),dtype=mx.uint32)
 pending=[];processed=0;parts=[]
 for first in range(0,count,chunk):
  size=min(chunk,count-first)
  reasons,witness=kernel(inputs=[masks,triples,sup,mx.array([first],dtype=mx.uint32),mx.array([size],dtype=mx.uint32)],
       grid=(size,1,1),threadgroup=(256,1,1),output_shapes=[(size,),(size,)],output_dtypes=[mx.uint32,mx.uint32])
  number=mx.sum((reasons==0).astype(mx.int32))
  histogram=mx.zeros((16,),dtype=mx.uint32).at[reasons].add(mx.ones_like(reasons))
  counts=counts+histogram
  mx.eval(reasons,witness,number,counts)
  valid_ix=mx.argsort(mx.where(reasons==0,mx.arange(size),mx.array(999999999)))[:number.item()]
  ids=valid_ix.astype(mx.uint32)+mx.array(first,dtype=mx.uint32)
  angular=witness[valid_ix]
  part=OUT/f'survivors_{first:010d}.npy';anglepart=OUT/f'angular_witness_{first:010d}.npy'
  if number.item()>0:
   mx.save(str(part),ids);mx.save(str(anglepart),angular)
  parts.append({'first':first,'count':size,'survivors':number.item(),
                'ids_file':str(part) if number.item()>0 else None,
                'angles_file':str(anglepart) if number.item()>0 else None})
  processed=first+size
  if len(parts)%16==0:print('screened',processed,'of',count,'contact requirements',flush=True)
  if time.monotonic()-start>seconds:break
 mx.eval(counts)
 summary={'wall_triple_representatives':triples.shape[0],
     'contact_requirements_in_representative_product':count,'processed':processed,
     'complete_angular_screen':processed==count,'surviving_angular_cases':counts[0].item(),
     'reason_bit_histogram':counts.tolist(),
     'reason_bits':{'1':'two vertices of one unit triangle identified','2':'opposite walls forced within one unit triangle',
                    '4':'two triangles forced to share a square corner','8':'necessary continuous angular system infeasible'},
     'elapsed_seconds':time.monotonic()-start,'budget_seconds':seconds,
     'global_optimality_proved':False,'spatial_roots_certified':0,
     'interpretation':'Exact discrete/angular exclusions only. Angular survivors need positional equation solving and complete root/critical-family certification.',
     'parts':parts,'gpu':mx.device_info()['device_name'],'cpu_numerical_fallback':False}
 (OUT/'screening_summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({k:v for k,v in summary.items() if k!='parts'},indent=2),flush=True)
 return summary

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--catalogue-only',action='store_true')
 parser.add_argument('--seconds',type=float,default=180);args=parser.parse_args()
 wall_catalog()
 if not args.catalogue_only:screen_catalogue(seconds=args.seconds)
