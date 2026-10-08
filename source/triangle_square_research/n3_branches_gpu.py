"""Bounded GPU numerical exploration of the complete 216 separating-edge
skeleton for three unit equilateral triangles in a square. Continuous spaces
within these branches are sampled, not exhausted. No proof is inferred from
failure to improve a record. CPU handles metadata and file I/O only.
"""
import os,json,time
from pathlib import Path
import mlx.core as mx

ROOT=Path(__file__).resolve().parent
mx.set_default_device(mx.gpu)
if not mx.metal.is_available():raise RuntimeError('Metal GPU required')
PI=mx.array(3.141592653589793,dtype=mx.float32)
H=mx.sqrt(mx.array(3.0))/2
R=H/3
LOCAL=mx.stack([mx.stack([mx.array(0.5),-R]),mx.stack([mx.array(0.0),2*R]),mx.stack([mx.array(-0.5),-R])])
LEFT=mx.array([0,0,1],dtype=mx.int32);RIGHT=mx.array([1,2,2],dtype=mx.int32)

def vertices(p):
 c=mx.cos(p[:,:,2])[:,:,None];s=mx.sin(p[:,:,2])[:,:,None]
 x=c*LOCAL[None,None,:,0]-s*LOCAL[None,None,:,1]+p[:,:,None,0]
 y=s*LOCAL[None,None,:,0]+c*LOCAL[None,None,:,1]+p[:,:,None,1]
 return mx.stack([x,y],axis=-1)

def six_options(v):
 a=v[:,LEFT];b=v[:,RIGHT]
 ea=mx.roll(a,-1,axis=2)-a;eb=mx.roll(b,-1,axis=2)-b
 na=mx.stack([ea[...,1],-ea[...,0]],axis=-1)
 nb=mx.stack([eb[...,1],-eb[...,0]],axis=-1)
 # Unit edge lengths: normals are already unit, up to float32 error.
 ab=mx.sum(b[:,:,None,:,:]*na[:,:,:,None,:],axis=-1)
 aa=mx.sum(a[:,:,None,:,:]*na[:,:,:,None,:],axis=-1)
 ba=mx.sum(a[:,:,None,:,:]*nb[:,:,:,None,:],axis=-1)
 bb=mx.sum(b[:,:,None,:,:]*nb[:,:,:,None,:],axis=-1)
 return mx.concatenate([mx.min(ab,axis=-1)-mx.max(aa,axis=-1),
                        mx.min(ba,axis=-1)-mx.max(bb,axis=-1)],axis=-1)

def explore(replicas=8,steps=7000,seconds=90):
 start=time.monotonic();mx.random.seed(9103)
 branch=mx.arange(216,dtype=mx.int32)
 codes=mx.stack([branch//36,(branch//6)%6,branch%6],axis=1)
 codes_b=mx.repeat(codes,replicas,axis=0)
 b=216*replicas
 public=json.loads((ROOT/'layouts'/'n3_public.json').read_text())
 p0=mx.array(public['best_values'],dtype=mx.float32).reshape(3,3)
 factor=1/mx.sqrt(mx.array(3.0))
 p0=mx.concatenate([p0[:,:2]*factor,p0[:,2:]],axis=1)
 reference=H+mx.sqrt(mx.array(6.0))/4
 mode=mx.arange(b)%replicas
 p=mx.broadcast_to(p0[None,:,:],(b,3,3))
 perturbed=mx.concatenate([p[:,:,:2]+mx.random.normal((b,3,2))*0.08,
                           p[:,:,2:]+mx.random.normal((b,3,1))*0.12],axis=-1)
 random=mx.concatenate([mx.random.uniform(-0.65,0.65,(b,3,2)),
                         mx.random.uniform(-PI/3,PI/3,(b,3,1))],axis=-1)
 p=mx.where((mode<2)[:,None,None],perturbed,random)
 initial=p
 q=mx.concatenate([p.reshape(b,9),mx.broadcast_to(reference+0.1,(b,1))],axis=1)
 guard=mx.array(0.00008)
 def selected(v):return mx.take_along_axis(six_options(v),codes_b[:,:,None],axis=-1)[:,:,0]
 def loss(q,lam):
  v=vertices(q[:,:9].reshape(b,3,3));g=selected(v)
  overlap=mx.maximum(guard-g,0)
  boundary=mx.maximum(mx.abs(v)-(q[:,-1]/2)[:,None,None,None]+guard,0)
  return mx.sum(q[:,-1]+lam*(mx.sum(overlap**2,axis=1)+mx.sum(boundary**2,axis=(1,2,3))))
 vg=mx.value_and_grad(loss)
 @mx.compile
 def update(q,m,v,lr,lam,bc1,bc2):
  value,g=vg(q,lam);g=mx.clip(g,-100,100)
  m=0.9*m+0.1*g;v=0.999*v+0.001*g*g
  q=q-lr*(m/bc1)/(mx.sqrt(v/bc2)+1e-8)
  pp=q[:,:9].reshape(b,3,3)
  pp=mx.concatenate([pp[:,:,:2],(mx.remainder(pp[:,:,2:]+PI/3,2*PI/3)-PI/3)],axis=-1)
  q=mx.concatenate([pp.reshape(b,9),mx.maximum(q[:,-1:],mx.array(1.1))],axis=1)
  return q,m,v,value
 m=mx.zeros_like(q);v=mx.zeros_like(q);bc1=mx.array(1.0);bc2=mx.array(1.0)
 best_s=mx.full((216,),1e9);best_pose=mx.zeros((216,3,3));best_g=mx.full((216,),-1e9)
 def capture(q,best_s,best_pose,best_g):
  pp=q[:,:9].reshape(b,3,3);vv=vertices(pp);gg=selected(vv)
  minimum=mx.min(gg,axis=1);ss=2*mx.max(mx.abs(vv),axis=(1,2,3))+2*guard
  valid=minimum>=guard/2
  ranked=mx.where(valid,ss,1e9).reshape(216,replicas)
  idx=mx.argmin(ranked,axis=1)
  vals=mx.take_along_axis(ranked,idx[:,None],axis=1)[:,0]
  poses=pp.reshape(216,replicas,3,3)[mx.arange(216),idx]
  gaps=minimum.reshape(216,replicas)[mx.arange(216),idx]
  improves=vals<best_s
  return mx.minimum(best_s,vals),mx.where(improves[:,None,None],poses,best_pose),mx.where(improves,gaps,best_g)
 for step in range(steps):
  if step<steps//2:lr=mx.array(0.006);lam=mx.array(100.0)
  elif step<steps*4//5:lr=mx.array(0.001);lam=mx.array(2000.0)
  else:lr=mx.array(0.00005);lam=mx.array(100000.0)
  bc1=bc1*0.9;bc2=bc2*0.999
  q,m,v,value=update(q,m,v,lr,lam,1-bc1,1-bc2)
  mx.eval(q,m,v,value,bc1,bc2)
  if step%100==0 or step==steps-1:
   best_s,best_pose,best_g=capture(q,best_s,best_pose,best_g)
   mx.eval(best_s,best_pose,best_g)
   if time.monotonic()-start>seconds:break
 # Positive-clearance repair; outer side recomputed from final vertices.
 @mx.compile
 def repair(q):
  _,g=vg(q,mx.array(1000.0))
  # Small bounded moves; no inference that infeasible branches cannot exist.
  return q-0.000001*mx.clip(g,-20,20)
 for _ in range(100):q=repair(q);mx.eval(q)
 best_s,best_pose,best_g=capture(q,best_s,best_pose,best_g)
 found=best_s<1e8;count=mx.sum(found.astype(mx.int32));globalidx=mx.argmin(best_s)
 gain=reference-mx.min(best_s)
 mx.eval(best_s,best_pose,best_g,count,globalidx,gain,reference,codes,initial)
 rows=[]
 for code,side,pose,g in zip(codes.tolist(),best_s.tolist(),best_pose.tolist(),best_g.tolist()):
  rows.append({'separating_edges':code,'feasible_found':side<1e8,
               'square_side':side if side<1e8 else None,
               'poses':pose if side<1e8 else None,'minimum_selected_separation':g if side<1e8 else None})
 result={'problem':'Three unit equilateral triangles in a square',
         'geometric_variables':10,'fixed_side_geometric_variables':9,
         'algebraic_variables_with_side':13,'unit_circle_equalities':3,
         'discrete_branches':216,'branch_constraints':'36 containment inequalities and 9 selected separating halfspace inequalities',
         'gpu':mx.device_info()['device_name'],'cpu_numerical_fallback':False,
         'seed':9103,'replicas_per_branch':replicas,'starts':b,'steps':step+1,
         'elapsed_seconds':time.monotonic()-start,'time_budget_seconds':seconds,
         'reference_formula':'sqrt(3)/2 + sqrt(6)/4','reference_side_gpu':reference.item(),
         'feasible_branches_found':count.item(),'best_branch':globalidx.item(),
         'best_side_gpu':mx.min(best_s).item(),'gain_gpu':gain.item(),
         'potential_counterexample':gain.item()>0,'global_proof':False,
         'interpretation':'Every separating-edge choice considered, but its continuous parameter space was only sampled. Failure to find feasibility is not an infeasibility certificate.',
         'initial_poses':initial.tolist(),'branches':rows}
 (ROOT/'results'/'n3_branches.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['initial_poses','branches']},indent=2),flush=True)
 return result

if __name__=='__main__':explore()
