"""Certified rigid-rotation descent test for full optimum profiles.

For a component spanning left/right, the right derivative of its width is
max(y_left)-min(y_right); its left derivative is min(y_left)-max(y_right).
A width minimum with strict height slack needs positive right and negative
left derivatives. A zero derivative still permits descent: every active
projection difference has second derivative -side<0. The vertical analogue
uses x_top-x_bottom. If both dimensions bind, a direction decreasing both
rules out a minimum. Connected components are rotated rigidly; untapped-wall
and intercomponent clearances are strict in a FULL active profile. Prefixes
with extra contacts are represented by other profiles, not declared infeasible.
"""
import argparse,json,uuid,hashlib,shutil
from pathlib import Path
import mlx.core as mx
from n3_branches_gpu import ROOT
from certified_model_gpu import model_inputs
from certified_intervals_gpu import HEADER
from certified_reference_gpu import reference_header
from certified_model_v2_gpu import ROTATION_HELPERS
from compact_certified_search_gpu import digest,load_rows,filter_rows,sha
from contact_catalog_gpu import OUT
mx.set_default_device(mx.gpu)

SOURCE=r'''
uint row=thread_position_in_grid.x;if(row>=amount[0])return;
bool unsafe=false;CI v[10];for(uint j=0;j<10;j++)v[j]=CI{states[23*row+3+2*j],states[23*row+4+2*j]};
uint id=(uint)states[23*row],chart=(uint)states[23*row+1],wi=id/50653u,m=id%50653u;
uint code[3]={m/1369u,(m/37u)%37u,m%37u},left[3]={0,0,1},right[3]={1,2,2};
uint wall[3][3],connection[3]={1u,2u,4u};
for(uint p=0;p<3;p++)if(code[p]){connection[left[p]]|=1u<<right[p];connection[right[p]]|=1u<<left[p];}
for(uint k=0;k<3;k++)for(uint i=0;i<3;i++)if(connection[i]&(1u<<k))connection[i]|=connection[k];
CI radical={constants_data[0],constants_data[1]},r=quotient(radical,point(6*CQ),unsafe);
CP base[3]={CP{point(CQ/2),negative(r)},CP{point(0),times_int(r,2)},CP{point(-CQ/2),negative(r)}};
CP pt[3][3];CI sidehalf=quotient(v[9],point(2*CQ),unsafe);
for(uint i=0;i<3;i++){
 uint mask=wmasks[triples[3*wi+i]];
 CI c,s,cl,sl,ch,sh;rational_rotation(v[6+i],c,s,unsafe);
 rational_rotation(point(v[6+i].lo),cl,sl,unsafe);rational_rotation(point(v[6+i].hi),ch,sh,unsafe);
 if((chart>>i)&1u){c=negative(c);s=negative(s);cl=negative(cl);sl=negative(sl);ch=negative(ch);sh=negative(sh);}
 for(uint a=0;a<3;a++){
  wall[i][a]=(mask>>(4*a))&15u;CP b=bounded_rotate(c,s,cl,sl,ch,sh,base[a],2*r.hi,unsafe);
  pt[i][a]=CP{ci_add(v[2*i],b.x),ci_add(v[2*i+1],b.y)};
  for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
   CI target=times_int(sidehalf,(w==0||w==2)?-1:1);
   if(w<2){if(!intersect(pt[i][a].x,target))unsafe=true;}
   else if(!intersect(pt[i][a].y,target))unsafe=true;
  }
 }
}
bool candidate=false;
for(uint component=0;component<3;component++){
 long low[4]={CWIDE,CWIDE,CWIDE,CWIDE},high[4]={-CWIDE,-CWIDE,-CWIDE,-CWIDE};uint flags=0;
 for(uint i=0;i<3;i++)if(connection[component]&(1u<<i))for(uint a=0;a<3;a++)for(uint w=0;w<4;w++)if(wall[i][a]&(1u<<w)){
  CI z=w<2?pt[i][a].y:pt[i][a].x;low[w]=min(low[w],z.lo);high[w]=max(high[w],z.hi);flags|=1u<<w;
 }
 bool horizontal=(flags&3u)==3u,vertical=(flags&12u)==12u;
 if(horizontal&&vertical){
  bool positive_descent=high[0]<=low[1]&&high[3]<=low[2];
  bool negative_descent=low[0]>=high[1]&&low[3]>=high[2];
  if(!positive_descent&&!negative_descent)candidate=true;
 }else if(horizontal){if(high[0]>low[1]&&high[1]>low[0])candidate=true;}
 else if(vertical){if(high[2]>low[3]&&high[3]>low[2])candidate=true;}
}
keep[row]=(candidate||unsafe)?1u:0u;
'''

def kernel(reference=False):
    return mx.fast.metal_kernel(name='minimum_rotation_reference' if reference else 'minimum_rotation_primary',
      input_names=['states','wmasks','triples','constants_data','amount'],output_names=['keep'],
      source=SOURCE,header=(reference_header() if reference else HEADER)+ROTATION_HELPERS)

def evaluate(states,inputs,reference=False):
    return kernel(reference)(inputs=[states,inputs[0],inputs[1],inputs[3],mx.array([states.shape[0]],dtype=mx.uint32)],
      grid=(states.shape[0],1,1),threadgroup=(128,1,1),output_shapes=[(states.shape[0],)],output_dtypes=[mx.uint32])[0]

def reduce(parent):
    parent=Path(parent).resolve();cp=json.loads((parent/'checkpoint.json').read_text())
    states=load_rows(cp['frontier_files']);inputs=model_inputs(cp['target_numerator'],cp['target_denominator'])
    keep=evaluate(states,inputs);mx.eval(keep)
    rejected=filter_rows(states,keep==0)
    if rejected is not None:
        rk=evaluate(rejected,inputs,reference=True);mx.eval(rk)
        assert mx.all(rk==0).item(),'Independent rotation-descent replay failed'
    # All-four-wall known singular family must survive this optimality test.
    known=json.loads((ROOT/'contact_catalog/critical_family.json').read_text())['known_pattern']['requirement_id']
    co=model_inputs(151,100)[3];half=(co[6]+1)//2
    domain=mx.concatenate([mx.broadcast_to(mx.stack([-half,half])[None,:],(6,2)),mx.broadcast_to(mx.array([-16777216,16777216],dtype=mx.int32)[None,:],(3,2)),mx.stack([co[4],co[6]])[None,:]],axis=0).reshape(1,20)
    knownbox=mx.concatenate([mx.array([[known,1,0]],dtype=mx.int32),domain],axis=1)
    assert evaluate(knownbox,model_inputs(151,100))[0].item()==1
    survivor=filter_rows(states,keep!=0)
    if survivor is None:survivor=mx.zeros((0,23),dtype=mx.int32)
    directory=ROOT/'certified_spatial'/f'minimum_rotation_{uuid.uuid4().hex[:8]}';directory.mkdir()
    if survivor.shape[0]:mx.save(str(directory/'frontier.npy'),survivor)
    sorted_id=mx.sort(survivor[:,0]);unique=mx.sum(mx.concatenate([mx.array([1]),(sorted_id[1:]!=sorted_id[:-1]).astype(mx.int32)])) if survivor.shape[0] else mx.array(0)
    mx.eval(unique)
    record={'passed':True,'parent_checkpoint':str(parent/'checkpoint.json'),'parent_checkpoint_sha256':sha(parent/'checkpoint.json'),
      'input_frontier_sha256':digest(states),'classification_sha256':digest(keep),'output_frontier_sha256':digest(survivor),
      'input_rows':states.shape[0],'minimum_regions_removed':states.shape[0]-survivor.shape[0],
      'remaining_rows':survivor.shape[0],'remaining_profiles':unique.item(),
      'independent_reference_all_rejections_passed':True,'known_singular_family_retained':True,
      'justification':__doc__,'cpu_numerical_fallback':False,'global_optimality_proved':False}
    record['primary_source_sha256']=hashlib.sha256((HEADER+ROTATION_HELPERS+SOURCE).encode()).hexdigest()
    record['reference_source_sha256']=hashlib.sha256((reference_header()+ROTATION_HELPERS+SOURCE).encode()).hexdigest()
    record['input_file_sha256']={str(p):sha(p) for p in [OUT/'wall_masks.npy',OUT/'wall_triplets.npy']}
    record['controller_sha256']=sha(__file__)
    (directory/'minimum_rotation_check.json').write_text(json.dumps(record,indent=2))
    (directory/'primary_model.metal').write_text(HEADER+ROTATION_HELPERS+SOURCE)
    (directory/'reference_model.metal').write_text(reference_header()+ROTATION_HELPERS+SOURCE)
    shutil.copyfile(__file__,directory/'controller_snapshot.py')
    result=dict(cp);result.update({'format':'minimum-rotation-reduction-v1','parent_directory':str(parent),
      'unresolved_active_boxes':survivor.shape[0],'frontier_profiles':unique.item(),
      'frontier_files':[str(directory/'frontier.npy')] if survivor.shape[0] else [],
      'new_minimum_rotation_regions_removed':record['minimum_regions_removed'],
      'minimum_rotation_regions_removed_total':cp.get('minimum_rotation_regions_removed_total',0)+record['minimum_regions_removed'],
      'new_boxes_evaluated':0,'new_interval_exclusions':0,'new_noncritical_boxes_removed':record['minimum_regions_removed'],
      'new_splits':0,'phase':'minimum_rotation_reduction','offline_replay_passed':False,
      'complete_below_target_exclusion':False,'all_minimum_candidate_regions_excluded':survivor.shape[0]==0 and cp['unresolved_precision_boxes']==0,
      'certified_lower_bound':None})
    (directory/'checkpoint.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='justification'},indent=2),flush=True)
    print('REDUCTION_DIRECTORY',directory,flush=True)
    (ROOT/'certified_spatial/minimum_rotation_pending.json').write_text(json.dumps({'directory':str(directory)},indent=2))
    return directory

def replay(directory):
    directory=Path(directory).resolve();cp=json.loads((directory/'checkpoint.json').read_text())
    r=json.loads((directory/'minimum_rotation_check.json').read_text())
    assert sha(__file__)==r['controller_sha256']
    assert sha(directory/'controller_snapshot.py')==r['controller_sha256']
    assert hashlib.sha256((HEADER+ROTATION_HELPERS+SOURCE).encode()).hexdigest()==r['primary_source_sha256']
    assert hashlib.sha256((reference_header()+ROTATION_HELPERS+SOURCE).encode()).hexdigest()==r['reference_source_sha256']
    assert sha(directory/'primary_model.metal')==r['primary_source_sha256']
    assert sha(directory/'reference_model.metal')==r['reference_source_sha256']
    for path,h in r['input_file_sha256'].items():assert sha(path)==h
    assert sha(r['parent_checkpoint'])==r['parent_checkpoint_sha256']
    parent=json.loads(Path(r['parent_checkpoint']).read_text());states=load_rows(parent['frontier_files'])
    assert digest(states)==r['input_frontier_sha256']
    inputs=model_inputs(cp['target_numerator'],cp['target_denominator'])
    keep=evaluate(states,inputs);mx.eval(keep);assert digest(keep)==r['classification_sha256']
    reject=filter_rows(states,keep==0)
    if reject is not None:assert mx.all(evaluate(reject,inputs,reference=True)==0).item()
    survivor=filter_rows(states,keep!=0)
    if survivor is None:survivor=mx.zeros((0,23),dtype=mx.int32)
    assert digest(survivor)==r['output_frontier_sha256']
    stored=load_rows(cp['frontier_files']);assert stored.shape==survivor.shape and mx.all(stored==survivor).item()
    audit={'passed':True,'rotation_reduction_reconstructed':True,'minimum_regions_removed':r['minimum_regions_removed'],
      'saved_splits_checked':0,'frontier_boxes':survivor.shape[0],'frontier_profiles':r['remaining_profiles'],
      'shared_geometric_model_independently_verified':False,'cpu_numerical_fallback':False,'global_optimality_proved':False}
    if cp.get('offline_replay_passed'):
        receipts=ROOT/'certified_spatial/replay_receipts';receipts.mkdir(exist_ok=True)
        (receipts/f'{directory.name}_{uuid.uuid4().hex[:8]}.json').write_text(json.dumps(audit,indent=2))
        print(json.dumps(audit,indent=2),flush=True);return
    (directory/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2))
    cp['offline_replay_passed']=True;cp['complete_below_target_exclusion']=cp['all_minimum_candidate_regions_excluded']
    if cp['complete_below_target_exclusion']:cp['certified_lower_bound']=f"s_min > {cp['target_numerator']}/{cp['target_denominator']} under documented coverage premises"
    cp['stored_certificate_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    (ROOT/'certified_spatial/latest_run.json').write_text(json.dumps({'directory':str(directory)},indent=2))
    print(json.dumps(audit,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--replay');a=p.parse_args()
    if a.replay:replay(a.replay)
    else:reduce(a.parent or json.loads((ROOT/'certified_spatial/latest_run.json').read_text())['directory'])
