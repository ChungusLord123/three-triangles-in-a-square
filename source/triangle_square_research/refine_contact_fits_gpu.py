"""Refine saved contact fits and recover the singular benchmark pattern.
This remains numerical optimization, not complete polynomial root isolation.
"""
import json,time
import mlx.core as mx
from n3_branches_gpu import H,PI,ROOT
from contact_catalog_gpu import OUT
from contact_fit_gpu import equations,decode

mx.set_default_device(mx.gpu)
start=time.monotonic()
f=json.loads((OUT/'fit_summary.json').read_text())
critical=json.loads((OUT/'critical_family.json').read_text())
reference=H+mx.sqrt(mx.array(6.0))/4
rows=list(f['candidates'])
benchmark=critical['known_pattern']
for _ in range(8):
 rows.append({'requirement_id':benchmark['requirement_id'],'square_side':reference.item(),
              'poses':benchmark['canonical_poses'],'status':'known-pattern benchmark'})
batch=len(rows)
ids=mx.array([r['requirement_id'] for r in rows],dtype=mx.uint32)
masks=mx.load(str(OUT/'wall_masks.npy'));triples=mx.load(str(OUT/'wall_triplets.npy'))
code,bits=decode(ids,triples,masks)
mx.random.seed(9451)
poses=mx.array([r['poses'] for r in rows])
q=mx.concatenate([poses.reshape(batch,9),mx.array([r['square_side'] for r in rows])[:,None]],axis=1)
perturb=mx.random.normal((batch,10))*0.0005
q=q+perturb
guard=mx.array(0.00005)
best_s=mx.array(1e9);best_p=mx.zeros((3,3));best_id=mx.array(0,dtype=mx.uint32)
def loss(q,weight):
 eq,g,v=equations(q,code,bits)
 ov=mx.maximum(guard-g,0)
 bound=mx.maximum(mx.abs(v)-q[:,-1,None,None,None]/2+guard,0)
 return mx.sum(q[:,-1]+weight*(mx.sum(eq**2,axis=1)+4*mx.sum(ov**2,axis=1)+4*mx.sum(bound**2,axis=(1,2,3))))
vg=mx.value_and_grad(loss)
@mx.compile
def step(q,m,v,lr,weight,bc1,bc2):
 value,g=vg(q,weight);g=mx.clip(g,-50,50)
 m=0.9*m+0.1*g;v=0.999*v+0.001*g*g
 q=q-lr*(m/bc1)/(mx.sqrt(v/bc2)+1e-8)
 return q,m,v,value
m=mx.zeros_like(q);v=mx.zeros_like(q);b1=mx.array(1.0);b2=mx.array(1.0)
for i in range(10000):
 if i<5000:lr=mx.array(0.0007);weight=mx.array(2000.0)
 elif i<8000:lr=mx.array(0.0001);weight=mx.array(20000.0)
 else:lr=mx.array(0.000015);weight=mx.array(100000.0)
 b1=b1*0.9;b2=b2*0.999
 q,m,v,value=step(q,m,v,lr,weight,1-b1,1-b2);mx.eval(q,m,v,value,b1,b2)
 if i%100==0:
  eq,g,verts=equations(q,code,bits)
  s=2*mx.max(mx.abs(verts),axis=(1,2,3))+2*guard
  residual=mx.max(mx.abs(eq),axis=1)
  feasible=(mx.min(g,axis=1)>=guard/2)&(residual<0.001)
  ranked=mx.where(feasible,s,1e9);k=mx.argmin(ranked)
  improves=ranked[k]<best_s
  best_s=mx.minimum(best_s,ranked[k]);best_p=mx.where(improves,q[k,:9].reshape(3,3),best_p)
  best_id=mx.where(improves,ids[k],best_id)
  mx.eval(best_s,best_p,best_id)
eq,g,verts=equations(q,code,bits)
s=2*mx.max(mx.abs(verts),axis=(1,2,3))+2*guard
residual=mx.max(mx.abs(eq),axis=1)
feasible=(mx.min(g,axis=1)>=guard/2)&(residual<0.001)
mx.eval(q,s,residual,feasible,best_s,best_p,best_id,reference)
saved=[]
for old,pp,side,res,ok in zip(rows,q[:,:9].reshape(batch,3,3).tolist(),s.tolist(),residual.tolist(),feasible.tolist()):
 if ok:saved.append({'requirement_id':old['requirement_id'],'square_side':side,'poses':pp,
                     'contact_residual':res,'status':'numerical near-contact fit, not certified root'})
saved.append({'requirement_id':best_id.item(),'square_side':best_s.item(),
              'poses':best_p.tolist(),'status':'best feasible numerical fit, not certified root'})
gain=reference-best_s;mx.eval(gain)
out={'fits_refined':batch,'steps':10000,'best_side':best_s.item(),'reference_side':reference.item(),
     'best_requirement_id':best_id.item(),'potential_record':gain.item()>0,
     'elapsed_seconds':time.monotonic()-start,'gpu':mx.device_info()['device_name'],
     'cpu_numerical_fallback':False,'certified_spatial_roots':0,'global_optimality_proved':False,
     'candidates':saved}
(OUT/'refinement_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='candidates'},indent=2),flush=True)
