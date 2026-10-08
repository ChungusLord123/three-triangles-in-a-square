"""One bounded numerical contact-system attempt per angular survivor.
These are approximate fits, not root isolation or infeasibility certificates.
Only geometrically feasible, independently-checkable layouts are saved.
"""
import json,time,argparse
import mlx.core as mx
from n3_branches_gpu import ROOT,LOCAL,H,R,PI,vertices,six_options
from contact_catalog_gpu import OUT

mx.set_default_device(mx.gpu)
NORMAL=mx.stack([mx.stack([H,mx.array(0.5)]),mx.stack([-H,mx.array(0.5)]),mx.array([0.0,-1.0])])
WALLN=mx.array([[-1.0,0.0],[1.0,0.0],[0.0,-1.0],[0.0,1.0]])
LEFT=mx.array([0,0,1],dtype=mx.int32);RIGHT=mx.array([1,2,2],dtype=mx.int32)

def decode(ids,triples,masks):
 mode=ids%50653
 code=mx.stack([mode//1369,(mode//37)%37,mode%37],axis=1).astype(mx.int32)
 wm=masks[triples[ids//50653]]
 bits=((wm[:,:,None,None]>>mx.arange(12,dtype=mx.uint32).reshape(1,1,3,4))&1).astype(mx.float32)
 return code,bits

def rotated_normals(p):
 co=mx.cos(p[:,:,2])[:,:,None];si=mx.sin(p[:,:,2])[:,:,None]
 return mx.stack([co*NORMAL[None,None,:,0]-si*NORMAL[None,None,:,1],
                  si*NORMAL[None,None,:,0]+co*NORMAL[None,None,:,1]],axis=-1)

def equations(q,code,bits):
 b=q.shape[0];p=q[:,:9].reshape(b,3,3);s=q[:,-1]
 vs=vertices(p);ns=rotated_normals(p)
 wall=(mx.sum(vs[:,:,:,None,:]*WALLN[None,None,None,:,:],axis=-1)-s[:,None,None,None]/2)*bits
 a=vs[:,LEFT];bb=vs[:,RIGHT];na=ns[:,LEFT];nb=ns[:,RIGHT]
 pos=p[:,:,:2];diff=pos[:,RIGHT]-pos[:,LEFT]
 vv=(code>0)&(code<10);ve=(code>=10)&(code<28);ee=code>=28
 zv=mx.clip(code-1,0,8)
 va=mx.take_along_axis(a,(zv//3)[:,:,None,None],axis=2)[:,:,0,:]
 vb=mx.take_along_axis(bb,(zv%3)[:,:,None,None],axis=2)[:,:,0,:]
 vv_eq=(va-vb)*vv[:,:,None]
 ze=mx.clip(code-10,0,17);owner=ze//9;e=(ze%9)//3;v=ze%3
 nea=mx.take_along_axis(na,e[:,:,None,None],axis=2)[:,:,0,:]
 neb=mx.take_along_axis(nb,e[:,:,None,None],axis=2)[:,:,0,:]
 vpa=mx.take_along_axis(a,v[:,:,None,None],axis=2)[:,:,0,:]
 vpb=mx.take_along_axis(bb,v[:,:,None,None],axis=2)[:,:,0,:]
 owner_center_a=pos[:,LEFT];owner_center_b=pos[:,RIGHT]
 eq_a=mx.sum(nea*(vpb-owner_center_a),axis=-1)-R
 eq_b=mx.sum(neb*(vpa-owner_center_b),axis=-1)-R
 ve_eq=mx.where(owner==0,eq_a,eq_b)*ve
 zp=mx.clip(code-28,0,8)
 en_a=mx.take_along_axis(na,(zp//3)[:,:,None,None],axis=2)[:,:,0,:]
 en_b=mx.take_along_axis(nb,(zp%3)[:,:,None,None],axis=2)[:,:,0,:]
 ee_angle=(en_a+en_b)*ee[:,:,None]
 ee_plane=(mx.sum(en_a*diff,axis=-1)-2*R)*ee
 eq=mx.concatenate([wall.reshape(b,36),vv_eq.reshape(b,6),ve_eq,ee_angle.reshape(b,6),ee_plane],axis=1)
 gaps=mx.max(six_options(vs),axis=-1)
 return eq,gaps,vs

def run(batch=16384,steps=192,seconds=180):
 started=time.monotonic();mx.random.seed(9417)
 screen=json.loads((OUT/'screening_summary.json').read_text())
 all_ids=mx.concatenate([mx.load(p['ids_file']) for p in screen['parts'] if p['ids_file']])
 all_angle=mx.concatenate([mx.load(p['angles_file']) for p in screen['parts'] if p['angles_file']])
 masks=mx.load(str(OUT/'wall_masks.npy'));triples=mx.load(str(OUT/'wall_triplets.npy'))
 reference=H+mx.sqrt(mx.array(6.0))/4
 minside=mx.sqrt(3*mx.sqrt(mx.array(3.0))/4)
 guard=mx.array(0.00005)
 candidates=[];attempted=0;fit_count=mx.array(0,dtype=mx.int32)
 best_side=mx.array(1e9);best_pose=mx.zeros((3,3));best_id=mx.array(0,dtype=mx.uint32);best_res=mx.array(1e9)
 @mx.compile
 def loss(q,code,bits,weight):
  eq,g,v=equations(q,code,bits)
  pair=mx.maximum(guard-g,0)
  boundary=mx.maximum(mx.abs(v)-q[:,-1,None,None,None]/2+guard,0)
  return mx.sum(q[:,-1]+weight*(mx.sum(eq*eq,axis=1)+mx.sum(pair*pair,axis=1)+mx.sum(boundary*boundary,axis=(1,2,3))))
 grad=mx.value_and_grad(loss)
 @mx.compile
 def step(q,m,v,code,bits,initial_angle,lr,weight,bc1,bc2):
  value,g=grad(q,code,bits,weight)
  g=mx.clip(g,-100,100);m=0.9*m+0.1*g;v=0.999*v+0.001*g*g
  q=q-lr*(m/bc1)/(mx.sqrt(v/bc2)+1e-8)
  p=q[:,:9].reshape(batch,3,3)
  angle=mx.clip(p[:,:,2],initial_angle-PI/6,initial_angle+PI/6)
  p=mx.concatenate([mx.clip(p[:,:,:2],-0.85,0.85),angle[:,:,None]],axis=-1)
  q=mx.concatenate([p.reshape(batch,9),mx.clip(q[:,-1:],minside,reference+0.03)],axis=1)
  return q,m,v,value
 for first in range(0,all_ids.size,batch):
  size=min(batch,all_ids.size-first)
  ids=all_ids[first:first+size];aw=all_angle[first:first+size]
  if size<batch:
   ids=mx.concatenate([ids,mx.broadcast_to(ids[:1],(batch-size,))])
   aw=mx.concatenate([aw,mx.broadcast_to(aw[:1],(batch-size,))])
  code,bits=decode(ids,triples,masks)
  theta=mx.stack([aw%12,(aw//12)%12,aw//144],axis=1).astype(mx.float32)*PI/6
  p=mx.concatenate([mx.zeros((batch,3,2)),theta[:,:,None]],axis=-1)
  vv=vertices(p)
  present=mx.any(bits>0,axis=2)
  x=mx.random.uniform(-0.16,0.16,(batch,3));y=mx.random.uniform(-0.16,0.16,(batch,3))
  x=mx.where(present[:,:,0],-reference/2-mx.min(vv[:,:,:,0],axis=-1),x)
  x=mx.where(present[:,:,1],reference/2-mx.max(vv[:,:,:,0],axis=-1),x)
  y=mx.where(present[:,:,2],-reference/2-mx.min(vv[:,:,:,1],axis=-1),y)
  y=mx.where(present[:,:,3],reference/2-mx.max(vv[:,:,:,1],axis=-1),y)
  p=mx.stack([x,y,theta],axis=-1)
  q=mx.concatenate([p.reshape(batch,9),mx.broadcast_to(reference+0.0001,(batch,1))],axis=1)
  m=mx.zeros_like(q);v=mx.zeros_like(q);bc1=mx.array(1.0);bc2=mx.array(1.0)
  for i in range(steps):
   lr=mx.array(0.005 if i<steps//2 else 0.001 if i<steps*4//5 else 0.00012)
   weight=mx.array(1000.0 if i<steps*4//5 else 10000.0)
   bc1=bc1*0.9;bc2=bc2*0.999
   q,m,v,value=step(q,m,v,code,bits,theta,lr,weight,1-bc1,1-bc2)
   mx.eval(q,m,v,value,bc1,bc2)
  eq,g,vs=equations(q,code,bits)
  side=2*mx.max(mx.abs(vs),axis=(1,2,3))+2*guard
  residual=mx.max(mx.abs(eq),axis=1)
  feasible=(mx.min(g,axis=1)>=guard/2)&(residual<0.002)&(mx.arange(batch)<size)
  ranked=mx.where(feasible,side,1e9);index=mx.argmin(ranked)
  fit_count=fit_count+mx.sum(feasible.astype(mx.int32))
  improves=ranked[index]<best_side
  best_side=mx.minimum(best_side,ranked[index]);best_pose=mx.where(improves,q[index,:9].reshape(3,3),best_pose)
  best_id=mx.where(improves,ids[index],best_id);best_res=mx.where(improves,residual[index],best_res)
  # Save the best of each chunk for independent geometric checking.
  mx.eval(index,best_side,best_pose,best_id,best_res,fit_count,ranked,residual)
  if ranked[index].item()<1e8:
   pp=q[index,:9].reshape(3,3);mx.eval(pp)
   candidates.append({'requirement_id':ids[index].item(),'square_side':ranked[index].item(),
       'poses':pp.tolist(),'contact_residual':residual[index].item(),
       'status':'approximate contact fit; not certified root'})
  attempted=first+size
  if first%(batch*16)==0:print('attempted',attempted,'of',all_ids.size,flush=True)
  if time.monotonic()-started>seconds:break
 gain=reference-best_side;mx.eval(gain)
 result={'angular_survivors':all_ids.size,'attempted':attempted,'all_survivors_attempted':attempted==all_ids.size,
  'iterations_per_attempt':steps,'near_contact_feasible_fits':fit_count.item(),
  'best_side':best_side.item() if best_side.item()<1e8 else None,
  'best_poses':best_pose.tolist() if best_side.item()<1e8 else None,
  'best_requirement_id':best_id.item(),'best_contact_residual':best_res.item(),
  'reference_side':reference.item(),'potential_record':gain.item()>0,
  'elapsed_seconds':time.monotonic()-started,'budget_seconds':seconds,
  'gpu':mx.device_info()['device_name'],'cpu_numerical_fallback':False,
  'rules':{'starts_per_requirement':1,'angle_departure_radians':0.5235987756,'max_square_side':'reference+0.03'},
  'certified_spatial_roots':0,'global_optimality_proved':False,'candidates':candidates,
  'interpretation':'Bounded numerical fits only. Every unsuccessful or approximate fit remains unresolved for global certification. Candidate physical feasibility requires independent checking.'}
 (OUT/'fit_summary.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['candidates','best_poses']},indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=180)
 parser.add_argument('--steps',type=int,default=192);parser.add_argument('--batch',type=int,default=16384)
 args=parser.parse_args();run(args.batch,args.steps,args.seconds)
