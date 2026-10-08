"""Checkpointed certified interval exclusions for n=3 minimum candidates.

Every initial profile gets all eight full-circle rotation-chart combinations.
Unresolved boxes are bisected, never treated as infeasible. A rational lower
bound is declared only when the full initial domain and all descendants are
excluded and each exclusion is reproduced by the independent reference.
"""
import argparse,json,time,uuid,hashlib
from pathlib import Path
import mlx.core as mx
from n3_branches_gpu import ROOT
from contact_catalog_gpu import OUT
from certified_intervals_gpu import Q,HEADER
from certified_model_gpu import model_inputs,evaluate,build_kernel,MODEL_HEADER,SOURCE
from certified_reference_gpu import reference_kernel,reference_header

mx.set_default_device(mx.gpu)

def root_states(ids,ordinals,co):
 selected=ids[ordinals//8].astype(mx.int32);chart=(ordinals%8).astype(mx.int32)
 half=(co[6].astype(mx.int64)+1)//2
 domain=mx.concatenate([mx.broadcast_to(mx.stack([-half,half])[None,:],(6,2)),
            mx.broadcast_to(mx.array([-Q,Q],dtype=mx.int64)[None,:],(3,2)),
            mx.stack([co[4].astype(mx.int64),co[6].astype(mx.int64)])[None,:]],axis=0).astype(mx.int32).reshape(20)
 metadata=mx.stack([selected,chart,mx.zeros_like(chart)],axis=1)
 return mx.concatenate([metadata,mx.broadcast_to(domain[None,:],(ordinals.size,20))],axis=1)

def filter_rows(rows,mask):
 count=mx.sum(mask.astype(mx.int32));mx.eval(count)
 if count.item()==0:return None
 ix=mx.argsort(mx.where(mask,mx.arange(mask.size),2147483647))[:count.item()]
 return rows[ix]

def split_rows(rows,dimensions,min_width):
 dim=mx.maximum(dimensions,0)
 lo=mx.take_along_axis(rows[:,3::2],dim[:,None],axis=1)[:,0]
 hi=mx.take_along_axis(rows[:,4::2],dim[:,None],axis=1)[:,0]
 can=(dimensions>=0)&(hi-lo>min_width)&(rows[:,2]<180)
 branch=filter_rows(rows,can);terminal=filter_rows(rows,~can)
 if branch is None:return None,terminal,None
 dims=filter_rows(dimensions[:,None],can)[:,0]
 lower=mx.take_along_axis(branch[:,3::2],dims[:,None],axis=1)[:,0]
 upper=mx.take_along_axis(branch[:,4::2],dims[:,None],axis=1)[:,0]
 middle=lower+(upper-lower)//2
 index=mx.arange(branch.shape[0])
 a=branch.at[index,4+2*dims].add(middle-upper).at[:,2].add(1)
 b=branch.at[index,3+2*dims].add(middle-lower).at[:,2].add(1)
 children=mx.stack([a,b],axis=1).reshape(-1,23)
 return children,terminal,dims

def run(numerator=14783,denominator=10000,seconds=180,batch=32768,max_frontier=16000000,min_width=16,resume=None,disk_budget=12884901888):
 started=time.monotonic()
 base=ROOT/'certified_spatial'
 if resume:
  directory=Path(resume);old=json.loads((directory/'checkpoint.json').read_text())
  numerator=old['target_numerator'];denominator=old['target_denominator']
  initial_next=old['initial_jobs_processed'];records=old['records']
  frontier_files=list(old['frontier_files']);terminal_files=list(old['terminal_files'])
  histogram=mx.array(old['primary_reason_histogram'],dtype=mx.int64)
  processed=old['boxes_evaluated'];verified=old['excluded_boxes_independently_replayed']
  previous_elapsed=old.get('total_elapsed_seconds',old['elapsed_seconds'])
 else:
  directory=base/f'run_{numerator}_{denominator}_{uuid.uuid4().hex[:8]}'
  directory.mkdir(parents=True)
  initial_next=0;records=[];frontier_files=[];terminal_files=[]
  histogram=mx.zeros((10,),dtype=mx.int64);processed=0;verified=0;previous_elapsed=0
 profiles=mx.load(str(base/'spatial_candidates.npy')).astype(mx.uint32)
 total=profiles.size*8
 inputs=model_inputs(numerator,denominator);co=inputs[3];mx.eval(co)
 primary=build_kernel();reference=reference_kernel()
 source=HEADER+MODEL_HEADER+SOURCE;source_digest=hashlib.sha256(source.encode()).hexdigest()
 if resume:assert old['primary_model_sha256']==source_digest,'Model changed; cannot resume this certificate tree'
 (directory/'primary_model.metal').write_text(source)
 (directory/'reference_model.metal').write_text(reference_header()+MODEL_HEADER+SOURCE)
 (directory/'constants.json').write_text(json.dumps(co.tolist()))
 replay_ok=True;last_announce=time.monotonic();phase='initial';invocation=uuid.uuid4().hex[:8]
 stored_bytes=sum(p.stat().st_size for p in directory.glob('*') if p.is_file())

 def process(states,label,initial_range=None):
  nonlocal histogram,processed,verified,replay_ok,stored_bytes
  updated,reasons,dims=evaluate(states,inputs,kernel=primary,rounds=4)
  mx.eval(updated,reasons,dims)
  excluded=(reasons!=0)&(reasons!=9)
  rejected=filter_rows(states,excluded)
  excluded_count=mx.sum(excluded.astype(mx.int32));mx.eval(excluded_count)
  # Stronger independent enclosures must reproduce every primary exclusion.
  if rejected is not None:
   _,rr,_=evaluate(rejected,inputs,kernel=reference,rounds=4)
   check=mx.all((rr!=0)&(rr!=9));mx.eval(check)
   if not check.item():
    mx.save(str(directory/f'{label}_replay_failure.npy'),rejected)
    replay_ok=False;raise RuntimeError('An exclusion was not independently reproduced; no proof can be claimed')
   verified+=excluded_count.item()
  histogram+=mx.stack([mx.sum((reasons==k).astype(mx.int64)) for k in range(10)])
  mx.eval(histogram)
  input_file=None
  if initial_range is None:
   input_file=str(directory/f'{label}_input.npy');mx.save(input_file,states);stored_bytes+=Path(input_file).stat().st_size
  reason_file=str(directory/f'{label}_reasons.npy');mx.save(reason_file,reasons);stored_bytes+=Path(reason_file).stat().st_size
  alive=(reasons==0)|(reasons==9)
  kept=filter_rows(updated,alive)
  split=filter_rows(dims[:,None],alive)
  kept_file=None;split_file=None
  if kept is not None:
   kept_file=str(directory/f'{label}_contracted.npy');mx.save(kept_file,kept);stored_bytes+=Path(kept_file).stat().st_size
   split_file=str(directory/f'{label}_dimensions.npy');mx.save(split_file,split[:,0]);stored_bytes+=Path(split_file).stat().st_size
  records.append({'kind':'initial' if initial_range else 'branch','input_file':input_file,
      'initial_range':initial_range,'boxes':states.shape[0],'reason_file':reason_file,
      'contracted_file':kept_file,'dimensions_file':split_file,
      'excluded':excluded_count.item(),'all_exclusions_replayed':True})
  processed+=states.shape[0]
  return kept,split[:,0] if split is not None else None

 # Complete the implicit initial domain first. Unvisited roots remain explicit
 # in the checkpoint's ordinal range and cannot disappear on a timeout.
 while initial_next<total:
  size=min(batch,total-initial_next)
  ordinal=mx.arange(initial_next,initial_next+size,dtype=mx.uint32)
  states=root_states(profiles,ordinal,co)
  kept,_=process(states,f'initial_{initial_next:010d}',[initial_next,size])
  if kept is not None:frontier_files.append(records[-1]['contracted_file'])
  initial_next+=size
  if time.monotonic()-last_announce>10:
   print('initial root boxes processed',initial_next,'/',total,'excluded',verified,flush=True);last_announce=time.monotonic()
  if time.monotonic()-started>=seconds or stored_bytes>=disk_budget:break

 if initial_next==total:
  phase='branching'
  # Each pass consumes every current frontier box once and records all children.
  pass_index=0
  while frontier_files and time.monotonic()-started<seconds:
   current=list(frontier_files);frontier_files=[];next_files=[]
   stopped=False
   for file_index,file in enumerate(current):
    data=mx.load(file)
    for first in range(0,data.shape[0],batch):
     states=data[first:first+batch]
     kept,dims=process(states,f'pass_{invocation}_{pass_index:04d}_{file_index:05d}_{first:08d}')
     if kept is not None:
      children,terminal,which=split_rows(kept,dims,min_width)
      if terminal is not None:
       path=str(directory/f'terminal_{len(records):08d}.npy');mx.save(path,terminal);stored_bytes+=Path(path).stat().st_size;terminal_files.append(path);records[-1]['terminal_file']=path
      if children is not None:
       path=str(directory/f'children_{len(records):08d}.npy');mx.save(path,children);stored_bytes+=Path(path).stat().st_size;next_files.append(path);records[-1]['children_file']=path
     records[-1]['minimum_split_width']=min_width
     if time.monotonic()-last_announce>10:
      print('spatial boxes evaluated',processed,'excluded',verified,'branch pass',pass_index,flush=True);last_announce=time.monotonic()
     if time.monotonic()-started>=seconds or stored_bytes>=disk_budget:
      if first+batch<data.shape[0]:
       path=str(directory/f'unprocessed_{len(records):08d}.npy');mx.save(path,data[first+batch:]);stored_bytes+=Path(path).stat().st_size;next_files.append(path)
      next_files.extend(current[file_index+1:]);stopped=True;break
    if stopped:break
   frontier_files=next_files;pass_index+=1
   pending_count=sum(mx.load(f).shape[0] for f in frontier_files)
   if pending_count>max_frontier:phase='frontier_limit';break
   if stored_bytes>=disk_budget:phase='storage_budget';break

 pending=sum(mx.load(f).shape[0] for f in frontier_files)
 terminal=sum(mx.load(f).shape[0] for f in terminal_files)
 complete=initial_next==total and pending==0 and terminal==0 and replay_ok
 mx.eval(histogram)
 result={'target_numerator':numerator,'target_denominator':denominator,'fixed_point_bits':24,
   'critical_spatial_profiles':profiles.size,'initial_jobs_total':total,'initial_jobs_processed':initial_next,
   'initial_jobs_unprocessed':total-initial_next,'boxes_evaluated':processed,
   'excluded_boxes_independently_replayed':verified,'primary_reason_histogram':histogram.tolist(),
   'unresolved_active_boxes':pending,'unresolved_precision_boxes':terminal,
   'complete_below_target_exclusion':complete,'certified_lower_bound':f's_min > {numerator}/{denominator}' if complete else None,
   'global_optimality_proved':False,'phase':phase,'elapsed_seconds':time.monotonic()-started,
   'total_elapsed_seconds':previous_elapsed+time.monotonic()-started,
   'stored_certificate_bytes':stored_bytes,'storage_budget_bytes':disk_budget,
  'time_budget_seconds':seconds,'primary_model_sha256':source_digest,
   'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'cpu_numerical_fallback':False,'gpu':mx.device_info()['device_name'],
   'reference_exclusion_checks_passed':replay_ok,'frontier_files':frontier_files,
   'terminal_files':terminal_files,'records':records,
   'scope':'Certified exclusion of minimum-candidate contact roots below a rational target. Flexible and singular regions are retained unless universally excluded. A stopped or nonempty search is not a proof.',
   'coverage_premises':'Compactness gives an attained optimum; its full proper contact profile is represented by the catalogue. Noncritical profiles and wall-owned proper contact cases are excluded by documented geometric reductions.'}
 (directory/'checkpoint.json').write_text(json.dumps(result,indent=2))
 (base/'latest_run.json').write_text(json.dumps({'directory':str(directory)},indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ['frontier_files','terminal_files','records']},indent=2),flush=True)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seconds',type=float,default=180);p.add_argument('--batch',type=int,default=32768)
 p.add_argument('--numerator',type=int,default=14783);p.add_argument('--denominator',type=int,default=10000)
 p.add_argument('--max-frontier',type=int,default=16000000)
 p.add_argument('--disk-budget-gib',type=int,default=20)
 p.add_argument('--resume');args=p.parse_args()
 run(args.numerator,args.denominator,args.seconds,args.batch,args.max_frontier,resume=args.resume,disk_budget=args.disk_budget_gib*1073741824)
