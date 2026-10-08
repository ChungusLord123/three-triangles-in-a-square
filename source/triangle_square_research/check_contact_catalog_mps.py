"""Independent exact-integer/Boolean GPU check of every catalogue decision.

Uses a relation join (12x12 angular compatibility matrices) rather than the
Metal enumerator's triple loops. Reachability uses Boolean graph products;
0/1 float32 dot products sum at most nine, hence are exact small integers.
"""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch,json,struct,ast,time
from pathlib import Path
from contact_catalog_gpu import OUT

D=torch.device('mps')
if not torch.backends.mps.is_available():raise RuntimeError('MPS GPU required')

def read_u32(path):
 data=Path(path).read_bytes()
 assert data[:6]==b'\x93NUMPY'
 if data[6]==1:
  n=struct.unpack('<H',data[8:10])[0];offset=10+n;head=data[10:offset]
 else:
  n=struct.unpack('<I',data[8:12])[0];offset=12+n;head=data[12:offset]
 meta=ast.literal_eval(head.decode('latin1'))
 assert meta['descr'] in ['<u4','<i4']
 # File decoding / transfer only. Operations begin after .to(MPS).
 return torch.frombuffer(bytearray(data),dtype=torch.int32,offset=offset).to(D).reshape(meta['shape'])

def relations():
 codes=torch.arange(37,device=D,dtype=torch.int32)
 a=torch.arange(12,device=D,dtype=torch.int32)[None,:,None]
 b=torch.arange(12,device=D,dtype=torch.int32)[None,None,:]
 beta=torch.tensor([11,3,7],device=D,dtype=torch.int32)
 gamma=torch.tensor([1,5,9],device=D,dtype=torch.int32)
 def distance(x):
  y=torch.remainder(x,12);return torch.minimum(y,12-y)
 z=torch.clamp(codes-1,min=0,max=8)
 vv=distance(a+beta[z//3,None,None]-b-beta[z%3,None,None])>=2
 z=torch.clamp(codes-10,min=0,max=17)
 e=(z%9)//3;v=z%3
 ve_a=distance(a+gamma[e,None,None]+6-b-beta[v,None,None])<=2
 ve_b=distance(b+gamma[e,None,None]+6-a-beta[v,None,None])<=2
 ve=torch.where((z//9==0)[:,None,None],ve_a,ve_b)
 z=torch.clamp(codes-28,min=0,max=8)
 ee=torch.remainder(a+gamma[z//3,None,None]-b-gamma[z%3,None,None],12)==6
 table=torch.where((codes==0)[:,None,None],torch.ones_like(vv),
        torch.where((codes<10)[:,None,None],vv,torch.where((codes<28)[:,None,None],ve,ee)))
 bit=torch.bitwise_left_shift(torch.ones(12,device=D,dtype=torch.int32),torch.arange(12,device=D,dtype=torch.int32))
 return (table.to(torch.int32)*bit[None,None,:]).sum(-1).to(torch.int32),bit

def decisions(ids,masks,triples,rel,bit):
 wi=ids//50653;m=ids%50653
 code=torch.stack([m//1369,(m//37)%37,m%37],dim=1)
 wall=masks[triples[wi]]
 flags=torch.bitwise_and(torch.bitwise_right_shift(wall[:,:,None],
           4*torch.arange(3,device=D,dtype=torch.int32)[None,None,:]),15).reshape(-1,9)
 batch=ids.numel()
 adjacency=torch.eye(9,device=D,dtype=torch.bool)[None,:,:].expand(batch,-1,-1).clone()
 own=torch.arange(batch,device=D,dtype=torch.int32)
 for p,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
  vv=(code[:,p]>0)&(code[:,p]<10)
  z=torch.clamp(code[:,p]-1,min=0,max=8)
  ia=3*i+z//3;ib=3*j+z%3
  adjacency[own,ia,ib]=vv;adjacency[own,ib,ia]=vv
 for _ in range(3):
  f=adjacency.to(torch.float32)
  adjacency=torch.bmm(f,f)>0
 inherited=torch.zeros_like(flags)
 for k in range(9):inherited|=torch.where(adjacency[:,:,k],flags[:,k,None],0)
 reason=torch.zeros(batch,device=D,dtype=torch.int32)
 same=torch.zeros(batch,device=D,dtype=torch.bool)
 for i in range(3):
  within=adjacency[:,3*i:3*i+3,3*i:3*i+3]
  same|=(within.to(torch.int32).sum(dim=(1,2))>3)
 reason|=same.to(torch.int32)
 combined=inherited.reshape(-1,3,3)
 whole=combined[:,:,0]|combined[:,:,1]|combined[:,:,2]
 opposite=(((whole&3)==3)|((whole&12)==12)).any(dim=1)
 reason|=opposite.to(torch.int32)*2
 corner=((inherited&3)!=0)&((inherited&12)!=0)
 corner_shared=torch.zeros(batch,device=D,dtype=torch.bool)
 for i in range(9):
  other=adjacency[:,i,:].clone();other[:,3*(i//3):3*(i//3)+3]=False
  corner_shared|=corner[:,i]&other.any(dim=1)
 reason|=corner_shared.to(torch.int32)*4
 phi=torch.tensor([6,0,9,3],device=D,dtype=torch.int32)
 beta=torch.tensor([11,3,7],device=D,dtype=torch.int32)
 angle=torch.arange(12,device=D,dtype=torch.int32)
 d=torch.remainder(phi[None,:,None]-beta[:,None,None]-angle[None,None,:]+6,12)-6
 support=d.abs()<=2
 present=(combined[:,:,:,None]>>torch.arange(4,device=D,dtype=torch.int32)[None,None,None,:])&1
 permitted=(~((present[:,:,:,:,None]>0)&(~support[None,None,:,:,:]))).all(dim=(2,3))
 mask=(permitted.to(torch.int32)*bit[None,None,:]).sum(-1).to(torch.int32)
 b_for_a=mask[:,1,None]&rel[code[:,0]]
 c_for_a=mask[:,2,None]&rel[code[:,1]]
 c_for_b=rel[code[:,2]]
 possible=((c_for_a[:,:,None]&c_for_b[:,None,:])!=0)&((b_for_a[:,:,None]&bit[None,None,:])!=0)
 possible&=(mask[:,0,None,None]&bit[None,:,None])!=0
 ok=possible.any(dim=(1,2))
 empty=(mask==0).any(dim=1)
 # Match the enumerator's staged accounting: after a positional contradiction
 # only empty support cones are tallied; the pair-angle test is skipped.
 angular_failure=empty|((reason==0)&(~ok))
 reason|=angular_failure.to(torch.int32)*8
 return reason,mask,code

def run(chunk=65536):
 start=time.monotonic()
 summary=json.loads((OUT/'screening_summary.json').read_text())
 masks=read_u32(OUT/'wall_masks.npy');triples=read_u32(OUT/'wall_triplets.npy')
 rel,bit=relations()
 histogram=torch.zeros(16,device=D,dtype=torch.int32)
 survivor_mismatch=torch.tensor(False,device=D)
 checked=0
 for part in summary['parts']:
  surviving=read_u32(part['ids_file']) if part['ids_file'] else torch.empty(0,device=D,dtype=torch.int32)
  for first in range(part['first'],part['first']+part['count'],chunk):
   size=min(chunk,part['first']+part['count']-first)
   ids=torch.arange(first,first+size,device=D,dtype=torch.int32)
   reason,_,_=decisions(ids,masks,triples,rel,bit)
   histogram+=torch.stack([(reason==k).sum() for k in range(16)]).to(torch.int32)
   if surviving.numel()>0:
    mine=surviving[(surviving>=first)&(surviving<first+size)]
    # Exact membership of every index in this chunk, not a sample.
    expected=torch.zeros(size,device=D,dtype=torch.bool)
    expected[mine-first]=True
    survivor_mismatch|=(expected!=(reason==0)).any()
   else:survivor_mismatch|=(reason==0).any()
   checked+=size
 expected_hist=torch.tensor(summary['reason_bit_histogram'],device=D,dtype=torch.int32)
 passed=bool((torch.equal(histogram,expected_hist)) and not survivor_mismatch.item())
 result={'passed':passed,'cases_checked':checked,'all_survivor_memberships_checked':True,
   'reason_histogram':histogram.tolist(),'independent_algorithm':'PyTorch MPS Boolean graph closure and angular relation join',
   'cpu_numerical_fallback':False,'elapsed_seconds':time.monotonic()-start,
   'scope':'Verifies finite/angular enumeration. Does not certify positional roots or global optimality.'}
 (OUT/'independent_check.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
 assert passed

if __name__=='__main__':run()
