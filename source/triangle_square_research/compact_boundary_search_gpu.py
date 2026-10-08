"""Exact-boundary generations with compact, deterministic GPU replay journals.

Only final unresolved states, terminal states, code snapshots and per-batch
hashes are saved. Every intermediate contraction, exclusion and bisection is
reconstructed during replay. Existing full-history evidence is never changed.
Numerical operations use MLX Metal; NumPy only exposes transferred bytes for
SHA-256/file serialization and never evaluates geometric arithmetic.
"""
import argparse,json,time,uuid,hashlib,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
from n3_branches_gpu import ROOT
from contact_catalog_gpu import OUT
from certified_intervals_gpu import HEADER
from certified_reference_gpu import reference_header
from certified_model_gpu import SOURCE,model_inputs,evaluate
from certified_model_v2_gpu import MODEL_HEADER,build_kernel
from certified_spatial_search import filter_rows,split_rows,root_states
from check_compact_splits_mps import check as check_split_union
from spatial_minimum_rotation_gpu import evaluate as evaluate_minimum
mx.set_default_device(mx.gpu)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a):
    if a is None:return None
    mx.eval(a)
    return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
def load_rows(files):
    parts=[mx.load(f) for f in files]
    return mx.concatenate(parts,axis=0) if parts else mx.zeros((0,23),dtype=mx.int32)
def rows_count(a):return 0 if a is None else a.shape[0]
def concatenate(parts):return mx.concatenate(parts,axis=0) if parts else mx.zeros((0,23),dtype=mx.int32)

def batch_step(states,inputs,primary,reference,rounds=4,min_width=16):
    updated,reasons,dims=evaluate(states,inputs,kernel=primary,rounds=rounds)
    mx.eval(updated,reasons,dims)
    excluded=(reasons!=0)&(reasons!=9)
    rejected=filter_rows(states,excluded)
    if rejected is not None:
        _,rr,_=evaluate(rejected,inputs,kernel=reference,rounds=rounds)
        mx.eval(rr);assert mx.all((rr!=0)&(rr!=9)).item(),'Independent exclusion replay failed'
    alive=(reasons==0)|(reasons==9)
    kept=filter_rows(updated,alive);selected=filter_rows(dims[:,None],alive)
    minimum_removed=0;minimum_hash=None
    if kept is not None:
        mk=evaluate_minimum(kept,inputs);mx.eval(mk);minimum_hash=digest(mk)
        rejected_minimum=filter_rows(kept,mk==0);minimum_removed=rows_count(rejected_minimum)
        if rejected_minimum is not None:
            ref_minimum=evaluate_minimum(rejected_minimum,inputs,reference=True)
            mx.eval(ref_minimum);assert mx.all(ref_minimum==0).item()
        selected=filter_rows(selected,mk!=0);kept=filter_rows(kept,mk!=0)
    children=terminal=None
    if kept is not None:
        children,terminal,_=split_rows(kept,selected[:,0],min_width)
        check_split_union(kept,selected[:,0],children,terminal,min_width)
    info={'boxes':states.shape[0],'excluded':rows_count(rejected),'surviving':rows_count(kept),
      'children':rows_count(children),'terminal':rows_count(terminal),
      'input_sha256':digest(states),'reasons_sha256':digest(reasons),
      'contracted_sha256':digest(kept),'dimensions_sha256':digest(selected),
      'children_sha256':digest(children),'terminal_sha256':digest(terminal),
      'minimum_regions_removed':minimum_removed,'minimum_classification_sha256':minimum_hash,
      'all_exclusions_replayed':True,'all_bisections_independently_checked':True}
    assert info['boxes']==info['excluded']+info['minimum_regions_removed']+info['surviving']
    assert 2*info['surviving']==info['children']+2*info['terminal']
    return children,terminal,info

def prepare_basis(parent,directory):
    cp=json.loads((parent/'checkpoint.json').read_text())
    files=list(cp['frontier_files']);terminals=list(cp['terminal_files'])
    basis={'parent_checkpoint':str(parent/'checkpoint.json'),'parent_checkpoint_sha256':sha(parent/'checkpoint.json'),
      'frontier':[{'path':f,'sha256':sha(f)} for f in files],
      'terminal':[{'path':f,'sha256':sha(f)} for f in terminals]}
    (directory/'basis.json').write_text(json.dumps(basis,indent=2))
    return cp,basis

def shell_roots(basis):
    parent=json.loads(Path(basis['parent_checkpoint']).read_text())
    assert parent['complete_below_target_exclusion'] and parent['offline_replay_passed']
    old_co=model_inputs(parent['target_numerator'],parent['target_denominator'])[3]
    co=model_inputs(basis['target_numerator'],basis['target_denominator'])[3]
    assert old_co[5].item()==basis['side_lower_fixed']
    assert (co[6]>old_co[5]).item()
    co=co.at[4].add(old_co[5]-co[4])
    ids=mx.load(str(ROOT/'certified_spatial/spatial_candidates.npy'))
    lookup=mx.load(str(ROOT/'certified_spatial/rotation_critical_lookup.npy'))
    ids=filter_rows(ids[:,None],lookup[ids]>0)[:,0]
    ordinal=mx.arange(ids.size*8,dtype=mx.uint32)
    return root_states(ids,ordinal,co),ids.size

def run(parent,seconds=120,batch=32768,max_frontier=4000000,min_width=16,shell=None,output_base=None):
    start=time.monotonic();base=Path(output_base).resolve() if output_base else ROOT/'certified_spatial'
    directory=base/f'compact_{uuid.uuid4().hex[:8]}';directory.mkdir()
    parent=Path(parent).resolve();old,basis=prepare_basis(parent,directory)
    if shell:
        assert old['complete_below_target_exclusion'] and old['offline_replay_passed']
        old_co=model_inputs(old['target_numerator'],old['target_denominator'])[3]
        basis.update({'kind':'side_shell','target_numerator':shell[0],'target_denominator':shell[1],
          'side_lower_fixed':old_co[5].item(),
          'justification':'The parent excludes all smaller sides. This shell begins at the dyadic floor of the parent rational target, so it overlaps the proved region and leaves no gap.'})
        old=dict(old);old['target_numerator']=shell[0];old['target_denominator']=shell[1]
        source,nprofiles=shell_roots(basis)
        old['critical_spatial_profiles']=nprofiles;old['initial_jobs_total']=source.shape[0]
        old['initial_jobs_processed']=source.shape[0];old['initial_jobs_unprocessed']=0
        old['unresolved_active_boxes']=source.shape[0]
        (directory/'basis.json').write_text(json.dumps(basis,indent=2))
    else:source=load_rows([r['path'] for r in basis['frontier']])
    rotation_check=json.loads((base/'rotation_critical_check.json').read_text())
    model_check=json.loads((base/'model_v2_checks.json').read_text())
    assert rotation_check['passed'] and model_check['passed']
    inputs=model_inputs(old['target_numerator'],old['target_denominator'])
    primary=build_kernel();reference=build_kernel(reference=True)
    input_files=[base/'rotation_critical_lookup.npy',base/'spatial_candidates.npy',OUT/'wall_masks.npy',OUT/'wall_triplets.npy',OUT/'wall_angles.npy']
    input_hashes={str(p):sha(p) for p in input_files}
    (directory/'primary_model.metal').write_text(HEADER+MODEL_HEADER+SOURCE)
    (directory/'reference_model.metal').write_text(reference_header()+MODEL_HEADER+SOURCE)
    shutil.copyfile(__file__,directory/'controller_snapshot.py')
    source_files=[Path(__file__),ROOT/'certified_model_v2_gpu.py',ROOT/'certified_spatial_search.py',ROOT/'certified_intervals_gpu.py',ROOT/'certified_reference_gpu.py',ROOT/'certified_model_gpu.py',ROOT/'check_compact_splits_mps.py',ROOT/'spatial_minimum_rotation_gpu.py']
    dependencies={str(p):sha(p) for p in source_files}
    config={'format':'compact-deterministic-replay-v1','target_numerator':old['target_numerator'],
      'target_denominator':old['target_denominator'],'rounds':4,'minimum_split_width':min_width,
      'batch':batch,'input_file_sha256':input_hashes,'code_file_sha256':dependencies,
      'primary_model_sha256':sha(directory/'primary_model.metal'),
      'reference_model_sha256':sha(directory/'reference_model.metal')}
    (directory/'config.json').write_text(json.dumps(config,indent=2))
    lookup=mx.load(str(base/'rotation_critical_lookup.npy'))
    eligible=lookup[source[:,0]]>0
    keep=filter_rows(source,eligible)
    current=keep if keep is not None else mx.zeros((0,23),dtype=mx.int32)
    reduced=source.shape[0]-current.shape[0];del source,keep
    print('Rotation-critical reduction:',reduced,'removed;',current.shape[0],'kept',flush=True)
    terminal_parts=[];evaluated=excluded=split_count=minimum_removed=0;last=time.monotonic();phase='completed';pass_index=0
    trace=directory/'journal.jsonl'
    with trace.open('w') as out:
        def record(r):out.write(json.dumps(r,separators=(',',':'))+'\n');out.flush()
        record({'kind':'rotation_filter','input_rows':old['unresolved_active_boxes'],'retained_rows':current.shape[0],
                'removed_noncritical_rows':reduced,'retained_sha256':digest(current)})
        while current.shape[0] and time.monotonic()-start<seconds:
            record({'kind':'pass_start','pass':pass_index,'rows':current.shape[0],'sha256':digest(current)})
            children_parts=[];processed=0;stop=False
            for first in range(0,current.shape[0],batch):
                states=current[first:first+batch]
                children,terminal,info=batch_step(states,inputs,primary,reference,min_width=min_width)
                info.update({'kind':'batch','pass':pass_index,'first':first});record(info)
                evaluated+=info['boxes'];excluded+=info['excluded'];split_count+=info['children']//2
                minimum_removed+=info['minimum_regions_removed']
                if children is not None:children_parts.append(children)
                if terminal is not None:terminal_parts.append(terminal)
                processed=first+states.shape[0]
                if time.monotonic()-last>10:
                    print('New evaluations',evaluated,'exclusions',excluded,'pass',pass_index,flush=True);last=time.monotonic()
                if time.monotonic()-start>=seconds:phase='time_budget';stop=True;break
                if sum(p.shape[0] for p in children_parts)+(current.shape[0]-processed)>max_frontier:
                    phase='frontier_limit';stop=True;break
            if processed<current.shape[0]:children_parts.append(current[processed:])
            nxt=concatenate(children_parts)
            record({'kind':'pass_end','pass':pass_index,'processed_rows':processed,'carry_rows':current.shape[0]-processed,
                    'next_rows':nxt.shape[0],'next_sha256':digest(nxt)})
            current=nxt;pass_index+=1
            if stop:break
        if current.shape[0] and phase=='completed':phase='time_budget'
        record({'kind':'final','frontier_rows':current.shape[0],'frontier_sha256':digest(current),
                'terminal_rows':sum(p.shape[0] for p in terminal_parts)})
    if current.shape[0]:mx.save(str(directory/'frontier.npy'),current)
    terminal=concatenate(terminal_parts)
    terminal_files=list(old['terminal_files'])
    if terminal.shape[0]:mx.save(str(directory/'terminal.npy'),terminal);terminal_files.append(str(directory/'terminal.npy'))
    frontier_files=[str(directory/'frontier.npy')] if current.shape[0] else []
    sort=mx.sort(current[:,0]);different=mx.concatenate([mx.array([1],dtype=mx.int64),(sort[1:]!=sort[:-1]).astype(mx.int64)]) if current.shape[0] else mx.array([0],dtype=mx.int64)
    profile_count=mx.sum(different);mx.eval(profile_count)
    terminal_total=old['unresolved_precision_boxes']+terminal.shape[0]
    empty=current.shape[0]==0 and terminal_total==0
    result={k:old[k] for k in ['target_numerator','target_denominator','fixed_point_bits','critical_spatial_profiles','initial_jobs_total','initial_jobs_processed','initial_jobs_unprocessed']}
    result.update({'format':'compact-deterministic-replay-v1','parent_directory':str(parent),
      'boxes_evaluated':old['boxes_evaluated']+evaluated,'excluded_boxes_independently_replayed':old['excluded_boxes_independently_replayed']+excluded,
      'new_boxes_evaluated':evaluated,'new_interval_exclusions':excluded,
      'new_noncritical_boxes_removed':reduced,'noncritical_boxes_removed_total':old.get('noncritical_boxes_removed_total',0)+reduced,
      'new_minimum_rotation_regions_removed':minimum_removed,
      'minimum_rotation_regions_removed_total':old.get('minimum_rotation_regions_removed_total',0)+minimum_removed,
      'new_splits':split_count,'unresolved_active_boxes':current.shape[0],'unresolved_precision_boxes':terminal_total,
      'frontier_profiles':profile_count.item(),'frontier_files':frontier_files,'terminal_files':terminal_files,
      'online_reference_exclusion_checks_passed':True,'reference_exclusion_checks_passed':True,
      'complete_below_target_exclusion':False,'all_minimum_candidate_regions_excluded':empty,
      'certified_lower_bound':None,'global_optimality_proved':False,
      'phase':phase,'elapsed_seconds':time.monotonic()-start,
      'total_elapsed_seconds':old['total_elapsed_seconds']+time.monotonic()-start,
      'cpu_numerical_fallback':False,'gpu':mx.device_info()['device_name'],
      'journal_sha256':sha(trace),'primary_model_sha256':config['primary_model_sha256'],
      'source_proof_scope':'Exact-boundary shell after separate algebraic family coverage; independently verified minimum-profile exclusions cover the remaining catalogue.',
      'scope':'Full continuous interval boxes; unresolved and terminal boxes are retained. Exclusion arithmetic is independently replayed; the geometric model still requires review.',
      'coverage_premises':old.get('coverage_premises',old.get('source_proof_scope')),
      'exact_boundary_goal':True,'analytic_seed_directory':old.get('analytic_seed_directory',str(parent)),
      'analytic_covered_profiles':old.get('analytic_covered_profiles',0)})
    (directory/'checkpoint.json').write_text(json.dumps(result,indent=2))
    result['stored_certificate_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    (directory/'checkpoint.json').write_text(json.dumps(result,indent=2))
    # Publishing latest is deferred until offline replay succeeds.
    print(json.dumps({k:v for k,v in result.items() if k not in ['frontier_files','terminal_files']},indent=2),flush=True)
    print('COMPACT_DIRECTORY',str(directory),flush=True)
    (base/'boundary_pending.json').write_text(json.dumps({'directory':str(directory)},indent=2))
    return directory

def replay(directory):
    start=time.monotonic();directory=Path(directory)
    cp=json.loads((directory/'checkpoint.json').read_text());config=json.loads((directory/'config.json').read_text())
    basis=json.loads((directory/'basis.json').read_text())
    assert sha(basis['parent_checkpoint'])==basis['parent_checkpoint_sha256']
    for entry in basis['frontier']+basis['terminal']:assert sha(entry['path'])==entry['sha256']
    for p,h in {**config['input_file_sha256'],**config['code_file_sha256']}.items():assert sha(p)==h,p
    assert sha(directory/'journal.jsonl')==cp['journal_sha256']
    assert sha(directory/'primary_model.metal')==config['primary_model_sha256']
    assert sha(directory/'reference_model.metal')==config['reference_model_sha256']
    current=shell_roots(basis)[0] if basis.get('kind')=='side_shell' else load_rows([p['path'] for p in basis['frontier']])
    lookup_path=next(Path(p) for p in config['input_file_sha256'] if Path(p).name=='rotation_critical_lookup.npy')
    lookup=mx.load(str(lookup_path))
    mask=lookup[current[:,0]]>0;filtered=filter_rows(current,mask)
    current=filtered if filtered is not None else mx.zeros((0,23),dtype=mx.int32)
    inputs=model_inputs(config['target_numerator'],config['target_denominator'])
    primary=build_kernel();reference=build_kernel(reference=True)
    records=[json.loads(line) for line in (directory/'journal.jsonl').read_text().splitlines()]
    children_parts=[];terminals=[];batches=boxes=exclusions=splits=minimum_removed=0;processed=0;last=time.monotonic()
    for r in records:
        if r['kind']=='rotation_filter':assert r['retained_rows']==current.shape[0] and r['retained_sha256']==digest(current)
        elif r['kind']=='pass_start':
            assert r['rows']==current.shape[0] and r['sha256']==digest(current)
            children_parts=[];processed=0
        elif r['kind']=='batch':
            assert r['first']==processed and r['boxes']<=config['batch']
            states=current[processed:processed+r['boxes']]
            children,terminal,info=batch_step(states,inputs,primary,reference,rounds=config['rounds'],min_width=config['minimum_split_width'])
            assert all(r[k]==v for k,v in info.items()),'Deterministic journal replay mismatch'
            if children is not None:children_parts.append(children)
            if terminal is not None:terminals.append(terminal)
            processed+=r['boxes'];batches+=1;boxes+=r['boxes'];exclusions+=r['excluded'];splits+=r['children']//2
            minimum_removed+=r['minimum_regions_removed']
            if time.monotonic()-last>10:print('Offline replay boxes',boxes,flush=True);last=time.monotonic()
        elif r['kind']=='pass_end':
            assert r['processed_rows']==processed and r['carry_rows']==current.shape[0]-processed
            if processed<current.shape[0]:children_parts.append(current[processed:])
            current=concatenate(children_parts)
            assert r['next_rows']==current.shape[0] and r['next_sha256']==digest(current)
        elif r['kind']=='final':
            assert r['frontier_rows']==current.shape[0] and r['frontier_sha256']==digest(current)
            assert r['terminal_rows']==sum(a.shape[0] for a in terminals)
        else:raise AssertionError('Unknown replay event')
    stored=load_rows(cp['frontier_files'])
    assert stored.shape==current.shape and mx.all(stored==current).item()
    term=concatenate(terminals)
    if term.shape[0]:assert mx.all(term==mx.load(str(directory/'terminal.npy'))).item()
    assert boxes==cp['new_boxes_evaluated'] and exclusions==cp['new_interval_exclusions'] and splits==cp['new_splits']
    assert minimum_removed==cp['new_minimum_rotation_regions_removed']
    result={'passed':True,'all_batches_reconstructed':batches,'all_intermediate_hashes_matched':True,
      'every_exclusion_rechecked_with_separate_arithmetic':True,'parent_and_input_hashes_matched':True,
      'all_new_bisections_independently_checked_on_MPS':True,
      'frontier_exactly_reconstructed':True,'new_boxes_evaluated':boxes,'new_interval_exclusions':exclusions,
      'minimum_regions_removed':minimum_removed,
      'saved_splits_checked':splits,'frontier_boxes':current.shape[0],'frontier_profiles':cp['frontier_profiles'],
      'shared_geometric_model_independently_verified':False,'cpu_numerical_fallback':False,
      'elapsed_seconds':time.monotonic()-start,'global_optimality_proved':False}
    if cp.get('offline_replay_passed'):
        receipt_dir=ROOT/'certified_spatial/replay_receipts';receipt_dir.mkdir(exist_ok=True)
        (receipt_dir/f'{directory.name}_{uuid.uuid4().hex[:8]}.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        return result
    (directory/'checkpoint_audit.json').write_text(json.dumps(result,indent=2))
    cp['offline_replay_passed']=True
    cp['complete_below_target_exclusion']=False
    cp['exact_boundary_remaining_catalogue_cleared']=cp['all_minimum_candidate_regions_excluded']
    if cp['complete_below_target_exclusion']:cp['certified_lower_bound']=f"s_min > {cp['target_numerator']}/{cp['target_denominator']} under documented coverage premises"
    cp['stored_certificate_bytes']=sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    (directory/'checkpoint.json').write_text(json.dumps(cp,indent=2))
    (ROOT/'certified_spatial/latest_boundary_run.json').write_text(json.dumps({'directory':str(directory.resolve())},indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent');p.add_argument('--replay');p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--batch',type=int,default=32768);p.add_argument('--max-frontier',type=int,default=4000000)
    p.add_argument('--shell-numerator',type=int);p.add_argument('--shell-denominator',type=int,default=100000)
    p.add_argument('--min-width',type=int,default=16)
    p.add_argument('--output-base')
    args=p.parse_args()
    if args.replay:replay(args.replay)
    else:
        parent=args.parent or json.loads((ROOT/'certified_spatial/latest_run.json').read_text())['directory']
        shell=(args.shell_numerator,args.shell_denominator) if args.shell_numerator else None
        run(parent,args.seconds,args.batch,args.max_frontier,args.min_width,shell,args.output_base)
