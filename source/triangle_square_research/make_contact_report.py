"""Research report; display geometry and numerical aggregation computed on GPU."""
import json,html
from pathlib import Path
import mlx.core as mx
from n3_branches_gpu import ROOT,vertices
from contact_catalog_gpu import OUT
from critical_family_gpu import exact_fixture

mx.set_default_device(mx.gpu)

def report_url(path):
 p=Path(path).resolve()
 try:return str(p.relative_to(ROOT))
 except ValueError:
  alias=ROOT/'certified_spatial/wd_game'
  if alias.is_symlink():return 'certified_spatial/wd_game/'+str(p.relative_to(alias.resolve()))
  raise

def exact_reference_display():
 # Decimal display bounds for the radical, using exact GPU integer arithmetic.
 kernel=mx.fast.metal_kernel(name='decimal_reference_side',input_names=['scale'],output_names=['out'],source=r'''
 long d=(long)scale[0],root[2];
 for(uint i=0;i<2;i++){
  long a=0,b=3*d,k=(i==0?3L:6L)*d*d;
  while(b-a>1){long m=a+(b-a)/2;if(m*m<=k)a=m;else b=m;}root[i]=a;
 }
 long n=2*root[0]+root[1];
 out[0]=(n+20)/40;out[1]=(n+3+20)/40;
 ''')
 out=kernel(inputs=[mx.array([1000000000],dtype=mx.int64)],grid=(1,1,1),threadgroup=(1,1,1),
       output_shapes=[(2,)],output_dtypes=[mx.int64])[0]
 mx.eval(out);assert mx.all(out==out[0]).item()
 digits=str(out[0].item())
 return digits[:-8]+'.'+digits[-8:]

def fraction_display(numerator,denominator):
 value=mx.array(numerator,dtype=mx.int64)*100000000
 rounded=(value+mx.array(denominator//2,dtype=mx.int64))//mx.array(denominator,dtype=mx.int64)
 mx.eval(rounded);digits=str(rounded.item()).zfill(9)
 return (digits[:-8]+'.'+digits[-8:]).rstrip('0').rstrip('.')

def completed_ancestor(directory):
 seen=set()
 while directory not in seen:
  seen.add(directory);cp=json.loads((directory/'checkpoint.json').read_text())
  if cp.get('complete_below_target_exclusion') and cp.get('offline_replay_passed'):
   return directory,cp
  parent=cp.get('parent_directory')
  if not parent:return None,None
  directory=Path(parent).resolve()
 return None,None

def drawing(p,side,label):
 v=vertices(p[None,:,:])[0]*mx.array([1.0,-1.0])
 extent=mx.stack([-side*0.54,-side*0.54,side*1.08,side*1.08])
 centers=mx.mean(v,axis=1);font=side*0.038
 mx.eval(v,extent,centers,font)
 forms=[]
 for k,points in enumerate(v.tolist()):
  forms.append('<polygon points="'+' '.join(f'{x},{y}' for x,y in points)+'"/>')
 text=''.join(f'<text x="{x}" y="{y}">{k+1}</text>' for k,(x,y) in enumerate(centers.tolist()))
 return '<svg role="img" aria-label="'+html.escape(label)+'" viewBox="'+' '.join(str(x) for x in extent.tolist())+'"><rect class="square" x="'+str((-side/2).item())+'" y="'+str((-side/2).item())+'" width="'+str(side.item())+'" height="'+str(side.item())+'"/><g class="triangles">'+''.join(forms)+'</g><g class="labels" font-size="'+str(font.item())+'" text-anchor="middle">'+text+'</g></svg>'

def make():
 cat=json.loads((OUT/'catalogue.json').read_text())
 screen=json.loads((OUT/'screening_summary.json').read_text())
 check=json.loads((OUT/'independent_check.json').read_text())
 wallcheck=json.loads((OUT/'wall_independent_check.json').read_text())
 fit=json.loads((OUT/'fit_summary.json').read_text())
 refined=json.loads((OUT/'refinement_summary.json').read_text())
 critical=json.loads((OUT/'critical_family.json').read_text())
 sos=json.loads((OUT/'family_sos_certificate.json').read_text())
 assert check['passed'] and wallcheck['passed'] and sos['passed']
 assert fit['all_survivors_attempted'] and not refined['potential_record']
 assert fit['independent_layout_check']['all_passed'] and refined['independent_layout_check']['all_passed']
 counts=mx.array([screen['processed'],screen['surviving_angular_cases']],dtype=mx.int32)
 removed=counts[0]-counts[1];mx.eval(removed)
 z=exact_fixture();exact_svg=drawing(z[:9].reshape(3,3),z[-1],'Known exact three-triangle construction, drawing rounded')
 accepted=[r for r in refined['candidates'] if r['independent_check']['passed']]
 best_index=mx.argmin(mx.array([r['square_side'] for r in accepted]));mx.eval(best_index)
 best=accepted[best_index.item()]
 numerical_svg=drawing(mx.array(best['poses']),mx.array(best['square_side']),'Independently checked numerical fit of three triangles')
 wallnames=['left','right','bottom','top']
 wallrows=[]
 for r in cat['wall_contact_rows']:
  req=', '.join(f"vertex {c['vertex']} → {c['wall']}" for c in r['requirements']) or 'No required wall contact; flexible case retained'
  wallrows.append(f'<tr><td>{r["id"]}</td><td>{html.escape(req)}</td></tr>')
 summary={'catalogue_complete':True,'angular_screen_complete':screen['complete_angular_screen'],
  'wall_types':cat['wall_types_up_to_cyclic_vertex_labels'],'wall_representatives':screen['wall_triple_representatives'],
  'contact_combinations_screened':screen['processed'],'angular_survivors':screen['surviving_angular_cases'],
  'all_survivors_received_bounded_fit':fit['all_survivors_attempted'],
  'identity_certificates_checked':2,'known_family_minimum_certificate':True,
  'global_optimality_proved':False,'fully_certified_spatial_root_catalogue':False,
  'best_checked_numerical_side':best['square_side'],'reference_side':refined['reference_side'],
  'saved_layouts_independently_checked':fit['independent_layout_check']['layouts_checked']+refined['independent_layout_check']['layouts_checked'],
  'cpu_numerical_fallback':False,
  'remaining_work':'Certified isolation of all relevant spatial roots and critical/flexible families, with spatial inequalities and degeneracies covered.'}
 reference_display=exact_reference_display()
 summary['reference_side_display']=reference_display
 run_dir=Path(json.loads((ROOT/'certified_spatial/latest_run.json').read_text())['directory']).resolve()
 spatial=json.loads((run_dir/'checkpoint.json').read_text())
 bound_dir,bound=completed_ancestor(run_dir)
 target_display=fraction_display(spatial['target_numerator'],spatial['target_denominator'])
 bound_display=fraction_display(bound['target_numerator'],bound['target_denominator']) if bound else None
 public_spatial={k:v for k,v in spatial.items() if k not in ['records','frontier_files','terminal_files']}
 public_spatial['run_directory']=str(run_dir)
 audit_path=run_dir/'checkpoint_audit.json'
 public_spatial['checkpoint_audit']=json.loads(audit_path.read_text()) if audit_path.exists() else None
 public_spatial['completed_bound_directory']=str(bound_dir) if bound_dir else None
 public_spatial['completed_bound_target']=bound_display
 public_spatial['actual_generation_bytes']=sum(p.stat().st_size for p in run_dir.iterdir() if p.is_file())
 public_spatial['active_frontier_bytes']=sum(Path(p).stat().st_size for p in spatial['frontier_files'])
 if bound:
  public_spatial['completed_bound_journal_bytes']=(bound_dir/'journal.jsonl').stat().st_size
 (ROOT/'certified_spatial/work_summary.json').write_text(json.dumps(public_spatial,indent=2))
 summary['spatial_certification']=public_spatial
 (OUT/'work_summary.json').write_text(json.dumps(summary,indent=2))
 template=(ROOT/'contacts_report_template.html').read_text()
 for key,value in {'SCREENED':format(screen['processed'],','),'SURVIVORS':format(screen['surviving_angular_cases'],','),
    'REMOVED':format(removed.item(),','),'BEST':format(best['square_side'],'.9f'),
    'REFERENCE':reference_display,'WALL_ROWS':''.join(wallrows),
    'EXACT_SVG':exact_svg,'NUMERIC_SVG':numerical_svg,'LAYOUTS':str(summary['saved_layouts_independently_checked']),
    'BOXES':format(spatial['boxes_evaluated'],','),'EXCLUSIONS':format(spatial['excluded_boxes_independently_replayed'],','),
    'PENDING':format(spatial['unresolved_active_boxes'],','),'INITIAL':format(spatial['initial_jobs_total'],','),
    'PROFILES':format(spatial['critical_spatial_profiles'],','),'RUN_RELATIVE':report_url(run_dir),
    'AUDIT_STATUS':'Passed' if public_spatial['checkpoint_audit'] and public_spatial['checkpoint_audit']['passed'] else 'Pending',
    'SPLITS':format(public_spatial['checkpoint_audit']['saved_splits_checked'],',') if public_spatial['checkpoint_audit'] else 'pending',
    'FRONTIER_PROFILES':format(public_spatial['checkpoint_audit']['frontier_profiles'],',') if public_spatial['checkpoint_audit'] else 'pending',
    'TARGET':f"{spatial['target_numerator']}/{spatial['target_denominator']}",'TARGET_DECIMAL':target_display,
    'BOUND_VALUE':f'Completed through {bound_display}' if bound else 'Not obtained',
    'BOUND_NOTE':'Every minimum-candidate region at this target was excluded and the complete journal replay passed. This is conditional on the documented catalogue, geometric model and necessary profile reductions.' if bound else 'A nonempty frontier does not certify a lower bound.',
    'STATUS_TEXT':f'Computational exclusion is complete through s={bound_display}, under the documented geometric model and catalogue. The current target is {target_display}; {spatial["unresolved_active_boxes"]:,} active boxes and {spatial["unresolved_precision_boxes"]:,} precision-limit boxes remain. The exact optimum has not been proved.' if bound else f'{spatial["unresolved_active_boxes"]:,} active regions remain unresolved. The tested lower bound and exact optimum have not been established.',
    'RUN_NOTE':f'Compact records save final unresolved states and hashes for deterministic reconstruction. The latest generation added {spatial.get("new_boxes_evaluated",0):,} evaluations, {spatial.get("new_interval_exclusions",0):,} interval exclusions and {spatial.get("new_minimum_rotation_regions_removed",0)+spatial.get("new_noncritical_boxes_removed",0):,} removals by necessary rotation conditions. Its stop reason is {spatial["phase"].replace("_"," ")}. Its saved data occupies {public_spatial["actual_generation_bytes"]:,} bytes. Earlier full-history evidence is preserved.' if spatial.get('format')=='compact-deterministic-replay-v1' else 'A separate necessary rotation-descent reduction was independently replayed. Earlier evidence is preserved.',
    'BOUND_LINK':report_url(bound_dir)+'/checkpoint_audit.json' if bound_dir else 'certified_spatial/work_summary.json'}.items():
  template=template.replace('@@'+key+'@@',value)
 (ROOT/'contacts-report.html').write_text(template)
 print(json.dumps({k:v for k,v in summary.items() if k!='spatial_certification'},indent=2),flush=True)

if __name__=='__main__':make()
