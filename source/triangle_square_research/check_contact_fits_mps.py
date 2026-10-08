"""Independent MPS checks of saved physical layouts from contact fitting."""
import os,json
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
from contact_catalog_gpu import OUT
D=torch.device('mps')
if not torch.backends.mps.is_available():raise RuntimeError('MPS GPU required')

for name in ['fit_summary.json','refinement_summary.json']:
 path=OUT/name
 if not path.exists():continue
 result=json.loads(path.read_text());rows=result['candidates']
 if not rows:continue
 p=torch.tensor([r['poses'] for r in rows],device=D,dtype=torch.float32)
 side=torch.tensor([r['square_side'] for r in rows],device=D,dtype=torch.float32)
 h=torch.sqrt(torch.tensor(3.0,device=D))/2;r=h/3
 local=torch.stack([torch.stack([torch.tensor(0.5,device=D),-r]),
                    torch.stack([torch.tensor(0.0,device=D),2*r]),torch.stack([torch.tensor(-0.5,device=D),-r])])
 co=torch.cos(p[:,:,2])[:,:,None];si=torch.sin(p[:,:,2])[:,:,None]
 v=torch.stack([co*local[None,None,:,0]-si*local[None,None,:,1]+p[:,:,None,0],
                si*local[None,None,:,0]+co*local[None,None,:,1]+p[:,:,None,1]],dim=-1)
 a=v[:,[0,0,1]];b=v[:,[1,2,2]]
 edges=torch.cat([torch.roll(a,-1,dims=2)-a,torch.roll(b,-1,dims=2)-b],dim=2)
 axes=torch.stack([-edges[...,1],edges[...,0]],dim=-1)
 pa=torch.einsum('bpvd,bpad->bpav',a,axes);pb=torch.einsum('bpvd,bpad->bpav',b,axes)
 gap=torch.maximum(pb.amin(-1)-pa.amax(-1),pa.amin(-1)-pb.amax(-1))/axes.square().sum(-1).sqrt()
 separation=gap.amax(-1).amin(-1)
 clearance=side/2-v.abs().amax(dim=(1,2,3))
 edgeerr=((torch.roll(v,-1,dims=2)-v).square().sum(-1).sqrt()-1).abs().amax(dim=(1,2))
 passed=(separation>0.000015)&(clearance>0.000015)&(edgeerr<0.00001)&torch.isfinite(p).all(dim=(1,2))
 for row,ok,sep,clr,err in zip(rows,passed.tolist(),separation.tolist(),clearance.tolist(),edgeerr.tolist()):
  row['independent_check']={'passed':ok,'engine':'PyTorch MPS','min_pair_separation':sep,
                            'min_square_clearance':clr,'max_unit_edge_error':err,'cpu_numerical_fallback':False}
 result['independent_layout_check']={'all_passed':passed.all().item(),'layouts_checked':len(rows),
                                  'engine':'PyTorch MPS','scope':'Physical feasibility only; not contact-root certification'}
 path.write_text(json.dumps(result,indent=2))
 print(name,result['independent_layout_check'],flush=True)
