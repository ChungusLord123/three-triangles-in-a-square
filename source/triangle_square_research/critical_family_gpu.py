"""Recover the known contact pattern, diagnose its singular fixed-side
system, and check a polynomial certificate for its one-parameter family.
The certificate is conditional on the contact family, not globally complete.
"""
import json
import mlx.core as mx
import torch
from n3_branches_gpu import ROOT,PI,H,R,LOCAL,vertices
from contact_catalog_gpu import OUT
from contact_fit_gpu import WALLN

mx.set_default_device(mx.gpu)

def exact_fixture():
 q=mx.sqrt(mx.array(6.0))/4;s=H+q
 center=mx.stack([mx.stack([2*q/3-s/2,2*q/3-s/2]),
                  mx.stack([q-s/2,q+2*H/3-s/2]),
                  mx.stack([q+2*H/3-s/2,q-s/2])])
 angle=mx.array([255.0,300.0,210.0])*PI/180
 p=mx.concatenate([center,angle[:,None]],axis=1)
 return mx.concatenate([p.reshape(9),s[None]])

def equalities(z):
 v=vertices(z[:9].reshape(1,3,3))[0];s=z[-1]
 # Corner triangle has vertex 0 at the southwest corner, top/right triangle
 # wall edges have endpoints 1,2. All three meet at top/right vertex 0.
 w=mx.stack([v[0,0,0]+s/2,v[0,0,1]+s/2,
             v[1,1,1]-s/2,v[1,2,1]-s/2,
             v[2,1,0]-s/2,v[2,2,0]-s/2])
 shared=v[1,0]-v[2,0]
 e=v[0,2]-v[0,1];n=mx.stack([e[1],-e[0]])
 touch=mx.stack([mx.sum(n*(v[1,0]-v[0,1])),mx.sum(n*(v[2,0]-v[0,1]))])
 return mx.concatenate([w,shared,touch])

JACOBI_SOURCE=r'''
float a[100];uint n=dimension[0];
for(uint i=0;i<n*n;i++)a[i]=matrix[i];
for(uint sweep=0;sweep<60;sweep++)for(uint p=0;p<n;p++)for(uint q=p+1;q<n;q++){
 float apq=a[p*n+q];if(fabs(apq)<1e-12f)continue;
 float tau=(a[q*n+q]-a[p*n+p])/(2.0f*apq);
 float t=(tau>=0.0f?1.0f:-1.0f)/(fabs(tau)+sqrt(1.0f+tau*tau));
 float c=rsqrt(1.0f+t*t),s=t*c;
 float app=a[p*n+p],aqq=a[q*n+q];
 a[p*n+p]=app-t*apq;a[q*n+q]=aqq+t*apq;
 a[p*n+q]=0;a[q*n+p]=0;
 for(uint k=0;k<n;k++)if(k!=p&&k!=q){
  float akp=a[k*n+p],akq=a[k*n+q];
  a[k*n+p]=c*akp-s*akq;a[p*n+k]=a[k*n+p];
  a[k*n+q]=s*akp+c*akq;a[q*n+k]=a[k*n+q];
 }
}
for(uint i=0;i<n;i++)eigenvalues[i]=a[i*n+i];
'''

def spectrum(j):
 gram=j.T@j;n=j.shape[1]
 kernel=mx.fast.metal_kernel(name='small_jacobi_gpu',input_names=['matrix','dimension'],
                  output_names=['eigenvalues'],source=JACOBI_SOURCE)
 eig=kernel(inputs=[gram.reshape(-1),mx.array([n],dtype=mx.uint32)],grid=(1,1,1),threadgroup=(1,1,1),
            output_shapes=[(n,)],output_dtypes=[mx.float32])[0]
 eig=mx.sort(eig);rank=mx.sum((eig>mx.max(eig)*0.00001).astype(mx.int32))
 mx.eval(eig,rank);return eig.tolist(),rank.item()

def wall_masks(v,s):
 gap=mx.sum(v[:,:,:,None,:]*WALLN[None,None,None,:,:],axis=-1)-s/2
 bits=(mx.abs(gap)<0.00001).astype(mx.uint32)
 return mx.sum(bits.reshape(v.shape[0],3,12)*(mx.array(1,dtype=mx.uint32)<<mx.arange(12,dtype=mx.uint32)),axis=-1)

def pair_modes(v):
 left=mx.array([0,0,1]);right=mx.array([1,2,2]);a=v[:,left];b=v[:,right]
 vv=mx.sum((a[:,:,:,None,:]-b[:,:,None,:,:])**2,axis=-1)
 vv_idx=mx.argmin(vv.reshape(v.shape[0],3,9),axis=-1)
 hasvv=mx.min(vv,axis=(2,3))<1e-10
 ea=mx.roll(a,-1,axis=2)-a;eb=mx.roll(b,-1,axis=2)-b
 na=mx.stack([ea[...,1],-ea[...,0]],axis=-1)
 nb=mx.stack([eb[...,1],-eb[...,0]],axis=-1)
 da=b[:,:,None,:,:]-a[:,:,:,None,:];db=a[:,:,None,:,:]-b[:,:,:,None,:]
 crossa=mx.abs(mx.sum(na[:,:,:,None,:]*da,axis=-1));crossb=mx.abs(mx.sum(nb[:,:,:,None,:]*db,axis=-1))
 ta=mx.sum(ea[:,:,:,None,:]*da,axis=-1);tb=mx.sum(eb[:,:,:,None,:]*db,axis=-1)
 hit_a=(crossa<1e-5)&(ta>1e-4)&(ta<1-1e-4)
 hit_b=(crossb<1e-5)&(tb>1e-4)&(tb<1-1e-4)
 options=mx.concatenate([hit_a.reshape(v.shape[0],3,9),hit_b.reshape(v.shape[0],3,9)],axis=-1)
 veidx=mx.argmax(options.astype(mx.int32),axis=-1)
 hasve=mx.any(options,axis=-1)
 return mx.where(hasvv,1+vv_idx,mx.where(hasve,10+veidx,0)).astype(mx.int32)

def canonical_id(z):
 mats=mx.array([[[1,0],[0,1]],[[0,-1],[1,0]],[[-1,0],[0,-1]],[[0,1],[-1,0]],
               [[-1,0],[0,1]],[[0,-1],[-1,0]],[[1,0],[0,-1]],[[0,1],[1,0]]],dtype=mx.float32)
 v=vertices(z[:9].reshape(1,3,3))[0]
 vd=mx.sum(v[None,:,:,:,None]*mats[:,None,None,:,:],axis=-2)
 # The matrix expression above gives row-vector multiplication. Both D4
 # rotation senses are present; reflections still reverse CCW ordering.
 vd=mx.concatenate([vd[:4],vd[4:,:,[0,2,1],:]],axis=0)
 cycles=[[0,1,2],[1,2,0],[2,0,1]]
 alternatives=mx.stack([vd[:,:,order,:] for order in cycles],axis=1)
 wm=mx.stack([wall_masks(alternatives[:,k],z[-1]) for k in range(3)],axis=1)
 choose=mx.argmin(wm,axis=1)
 vc=mx.take_along_axis(alternatives.transpose(0,2,1,3,4),choose[:,:,None,None,None],axis=2)[:,:,0]
 permutations=[[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]
 vp=mx.stack([vc[:,p] for p in permutations],axis=1).reshape(48,3,3,2)
 masks=mx.load(str(OUT/'wall_masks.npy'))
 walls=wall_masks(vp,z[-1])
 ids=mx.sum((masks[None,None,:]<walls[:,:,None]).astype(mx.int32),axis=-1)
 modes=pair_modes(vp)
 code=ids[:,0]*25*25+ids[:,1]*25+ids[:,2]
 bestwall=mx.min(code)
 contact=modes[:,0]*1369+modes[:,1]*37+modes[:,2]
 ranked=mx.where(code==bestwall,contact,999999)
 pick=mx.argmin(ranked)
 reps=mx.load(str(OUT/'wall_triplets.npy'))
 repcode=reps[:,0]*25*25+reps[:,1]*25+reps[:,2]
 match=repcode==bestwall;row=mx.argmax(match.astype(mx.int32))
 requirement=row*50653+contact[pick]
 cp=mx.mean(vp[pick],axis=1)
 radial=vp[pick,:,0,:]-cp
 angle=mx.arctan2(radial[:,1],radial[:,0])+PI/6
 poses=mx.concatenate([cp,angle[:,None]],axis=1)
 mx.eval(requirement,ids,modes,pick,match,poses)
 assert mx.any(match).item()
 screening=json.loads((OUT/'screening_summary.json').read_text())
 selected_part=next(p for p in screening['parts'] if p['first']<=requirement.item()<p['first']+p['count'])
 stored=mx.load(selected_part['ids_file']);present=mx.any(stored==requirement)
 mx.eval(present)
 assert present.item()
 return {'requirement_id':requirement.item(),'wall_type_ids':ids[pick].tolist(),'canonical_poses':poses.tolist(),
         'pair_mode_codes':modes[pick].tolist(),'present_in_angular_survivors':present.item()}

def certificate():
 # Formal expansion with exact small int32 coefficients. Let u=s-h and q>0.
 # Contact equation: u(1-t²)-q(1+t²)=0.
 contact=mx.array([1,-1,-1,-1],dtype=mx.int32) # u,q,u t²,q t²
 expansion=(mx.array([1,-1],dtype=mx.int32)[:,None]*mx.array([1,-1],dtype=mx.int32)[None,:]).T.reshape(4)
 certificate=expansion-mx.array([0,0,0,2],dtype=mx.int32)
 identity=mx.all(contact==certificate);mx.eval(identity,contact,certificate)
 device=torch.device('mps')
 a=torch.tensor([1,-1],device=device,dtype=torch.int32)
 b=torch.tensor([1,-1],device=device,dtype=torch.int32)
 other=(a[:,None]*b[None,:]).T.reshape(4)-torch.tensor([0,0,0,2],device=device,dtype=torch.int32)
 independent=torch.equal(other,torch.tensor(contact.tolist(),device=device,dtype=torch.int32))
 assert identity.item() and independent
 return {'identity_verified':identity.item(),'independent_MPS_identity_verified':independent,
   'variables':{'u':'s-sqrt(3)/2','q':'sqrt(6)/4','t':'tan(delta/2)'},
   'contact_equation':'u*(1-t^2)-q*(1+t^2)=0',
   'certificate_identity':'(u-q)*(1-t^2)-2*q*t^2 = u*(1-t^2)-q*(1+t^2)',
   'coefficient_order':['u','q','u*t^2','q*t^2'],'coefficients':contact.tolist(),
   'domain':['q>0','1-t^2>0'],
   'consequence':'s-(sqrt(3)/2+sqrt(6)/4)=2*q*t^2/(1-t^2)>=0',
   'equality':'t=0',
   'conditional_family':'Corner vertex at southwest corner; other two triangles have bases on top and right walls and share an inward vertex; the corner triangle opposite edge passes through that shared vertex.',
   'geometry_to_family_equation':'sqrt(2)*(s-h)*cos(delta)=h, with delta the corner-edge outward normal angle relative to 45 degrees',
   'global_completeness_proved':False,
   'note':'Exact coefficient identity and positivity certificate for this family. Does not exclude other contact families.'}

if __name__=='__main__':
 z=exact_fixture();residual=equalities(z)
 jac=mx.stack([mx.grad(lambda x,k=k:equalities(x)[k])(z) for k in range(10)])
 fixed_eig,fixed_rank=spectrum(jac[:,:9]);full_eig,full_rank=spectrum(jac)
 maximum=mx.max(mx.abs(residual));mx.eval(z,jac,maximum)
 result={'known_pattern':canonical_id(z),'max_equality_residual_float32':maximum.item(),
   'fixed_side_numeric_jacobian_rank':fixed_rank,'fixed_side_geometric_variables':9,
   'variable_side_numeric_jacobian_rank':full_rank,'variable_side_geometric_variables':10,
   'fixed_side_gram_spectrum':fixed_eig,'variable_side_gram_spectrum':full_eig,
   'rank_precision':'GPU float32 diagnostic, relative eigenvalue threshold 1e-5; not an exact rigidity certificate',
   'singular_fixed_side_motion':'The corner triangle can rotate to first order while maintaining its corner vertex; the shared-vertex contact fails at second order.',
   'family_certificate':certificate(),'global_optimality_proved':False}
 (OUT/'critical_family.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2),flush=True)
