"""Reconstruct every legacy root, contraction, exclusion and queued child.

The old root history predates hash journals. This fresh replay additionally
checks exact queue lineage across interrupted passes, rather than relying on
aggregate row counts. All coordinate/arithmetic operations run on GPU. CPU
decodes files, tracks filenames/ranges and hashes bytes only.
"""
import os,json,time,hashlib,re,struct,ast
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
from pathlib import Path
import mlx.core as mx
import numpy as np
from n3_branches_gpu import ROOT
from certified_spatial_search import root_states,filter_rows,split_rows
from certified_model_gpu import model_inputs,evaluate
from check_compact_splits_mps import check as split_check
from wd_proof_executor import ram_staged_load
mx.set_default_device(mx.gpu);mx.load=ram_staged_load

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def shape(path):
    with Path(path).open('rb') as f:
        head=f.read(8);assert head[:6]==b'\x93NUMPY'
        length=struct.unpack('<H' if head[6]==1 else '<I',f.read(2 if head[6]==1 else 4))[0]
        return ast.literal_eval(f.read(length).decode('latin1'))['shape']
def kernel(path,label):
    full=Path(path).read_text();position=full.index('uint row=thread_position_in_grid.x')
    return mx.fast.metal_kernel(name=label,
        input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
        output_names=['updated','reasons','split_variable'],header=full[:position],source=full[position:])
def cat(parts,columns=None):
    if parts:return mx.concatenate(parts,axis=0)
    return mx.zeros((0,columns),dtype=mx.int32) if columns else mx.zeros((0,),dtype=mx.int32)

def replay():
    start=time.monotonic();last=start
    directory=ROOT/'certified_spatial/run_14783_10000_a8bd59da'
    cp=json.loads((directory/'checkpoint.json').read_text());audit=json.loads((directory/'checkpoint_audit.json').read_text())
    assert audit['passed'] and cp['reference_exclusion_checks_passed']
    for name,expected in audit['input_data_sha256'].items():assert sha(ROOT/name)==expected
    assert sha(directory/'primary_model.metal')==cp['primary_model_sha256']
    primary=kernel(directory/'primary_model.metal','legacy_fresh_primary')
    reference=kernel(directory/'reference_model.metal','legacy_fresh_reference')
    inputs=model_inputs(cp['target_numerator'],cp['target_denominator'])
    profiles=mx.load(str(ROOT/'certified_spatial/spatial_candidates.npy'))
    states_parts=[];expected_parts=[];reason_parts=[];kept_parts=[];dim_parts=[];child_parts=[];terminal_parts=[]
    batch_rows=0;phase=None;evaluated=exclusions=splits=records_checked=0
    queue=[];next_queue=[];group=None;file_index=0;offset=0;next_root=0;initial_rows=0;branch_inputs=0
    cache_path=None;cache_rows=None
    def load(path):return mx.load(str(path))
    def entry(path):return (str(path),0,shape(path)[0])
    def pending():
        result=list(next_queue)
        if group is not None:
            if file_index<len(queue):
                p,lo,count=queue[file_index]
                if offset<count:result.append((p,lo+offset,count-offset))
                result.extend(queue[file_index+1:])
        else:result.extend(queue)
        return result
    def flush():
        nonlocal batch_rows,evaluated,exclusions,splits
        if not states_parts:return
        states=cat(states_parts,23);expected=cat(expected_parts,23);saved_reason=cat(reason_parts)
        assert states.shape==expected.shape and mx.all(states==expected).item(),'Legacy queue input differs from its parent'
        outputs=[]
        for first in range(0,states.shape[0],4096):
            values=evaluate(states[first:first+4096],inputs,kernel=primary,rounds=4)
            mx.eval(*values);outputs.append(values)
        updated,reason,dimension=[mx.concatenate([o[i] for o in outputs]) for i in range(3)]
        assert mx.all(reason==saved_reason).item(),'Legacy saved reasons differ from fresh primary'
        rejected=(reason!=0)&(reason!=9);bad=filter_rows(states,rejected)
        number=mx.sum(rejected.astype(mx.int32)).item()
        if bad is not None:
            for first in range(0,bad.shape[0],4096):
                _,rr,_=evaluate(bad[first:first+4096],inputs,kernel=reference,rounds=4)
                assert mx.all((rr!=0)&(rr!=9)).item(),'Legacy physical exclusion not reproduced'
        alive=~rejected;kept=filter_rows(updated,alive);dims=filter_rows(dimension[:,None],alive)
        k=kept if kept is not None else mx.zeros((0,23),dtype=mx.int32)
        d=dims[:,0] if dims is not None else mx.zeros((0,),dtype=mx.int32)
        sk=cat(kept_parts,23);sd=cat(dim_parts)
        assert sk.shape==k.shape and mx.all(sk==k).item(),'Saved contraction changed a region'
        assert sd.shape==d.shape and mx.all(sd==d).item(),'Saved branch dimension differs'
        if phase=='branch' and kept is not None:
            children,terminal,_=split_rows(kept,d,16)
            sc=cat(child_parts,23);st=cat(terminal_parts,23)
            actual_c=children if children is not None else mx.zeros((0,23),dtype=mx.int32)
            actual_t=terminal if terminal is not None else mx.zeros((0,23),dtype=mx.int32)
            assert sc.shape==actual_c.shape and mx.all(sc==actual_c).item()
            assert st.shape==actual_t.shape and mx.all(st==actual_t).item()
            split_check(kept,d,children,terminal,16)
            splits+=actual_c.shape[0]//2
        evaluated+=states.shape[0];exclusions+=number
        for parts in [states_parts,expected_parts,reason_parts,kept_parts,dim_parts,child_parts,terminal_parts]:parts.clear()
        batch_rows=0;mx.synchronize();mx.clear_cache()
    for record in cp['records']:
        kind=record['kind']
        if phase is not None and phase!=kind:flush()
        phase=kind
        if kind=='initial':
            first,count=record['initial_range'];assert first==next_root and count==record['boxes']
            ordinal=mx.arange(first,first+count,dtype=mx.uint32)
            state=root_states(profiles,ordinal,inputs[3]);expected=state
            next_root+=count;initial_rows+=count
            if record['contracted_file']:queue.append(entry(record['contracted_file']))
        else:
            match=re.fullmatch(r'pass_([0-9a-f]+)_(\d+)_(\d+)_(\d+)_input.npy',Path(record['input_file']).name)
            assert match
            invocation,pass_no,fi,first=match.groups();current=(invocation,int(pass_no));fi=int(fi);first=int(first)
            if current!=group:
                if group is not None:queue=pending()
                next_queue=[];group=current;file_index=0;offset=0;cache_path=None;cache_rows=None
            while file_index<len(queue) and offset==queue[file_index][2]:file_index+=1;offset=0
            assert fi==file_index and first==offset,(record['input_file'],file_index,offset)
            path,lo,count=queue[file_index]
            assert first+record['boxes']<=count
            if path!=cache_path:cache_rows=load(path);cache_path=path
            expected=cache_rows[lo+first:lo+first+record['boxes']]
            state=load(record['input_file']);offset=first+record['boxes'];branch_inputs+=state.shape[0]
            if record.get('children_file'):next_queue.append(entry(record['children_file']))
        assert state.shape[0]==record['boxes']
        states_parts.append(state);expected_parts.append(expected);reason_parts.append(load(record['reason_file']))
        if record['contracted_file']:
            kept_parts.append(load(record['contracted_file']));dim_parts.append(load(record['dimensions_file']))
        if kind=='branch':
            if record.get('children_file'):child_parts.append(load(record['children_file']))
            if record.get('terminal_file'):terminal_parts.append(load(record['terminal_file']))
        batch_rows+=state.shape[0];records_checked+=1
        if batch_rows>=131072:flush()
        if time.monotonic()-last>10:
            print('Fresh legacy replay',evaluated,'boxes;',records_checked,'records',flush=True);last=time.monotonic()
    flush();expected_frontier=pending();wanted=cat([load(p)[lo:lo+count] for p,lo,count in expected_frontier],23)
    stored=cat([load(p) for p in cp['frontier_files']],23)
    assert wanted.shape==stored.shape and mx.all(wanted==stored).item(),'Legacy unresolved frontier does not preserve the exact unprocessed queue'
    assert evaluated==cp['boxes_evaluated'] and exclusions==cp['excluded_boxes_independently_replayed']
    assert next_root==profiles.size*8==cp['initial_jobs_processed']
    result={'passed':True,'legacy_records_reconstructed':records_checked,'boxes_reconstructed':evaluated,
        'physical_exclusions_freshly_rechecked':exclusions,'splits_independently_rechecked_on_MPS':splits,
        'every_branch_input_matches_its_exact_parent_queue':True,'interrupted_pass_carry_regions_exactly_preserved':True,
        'all_primary_reasons_contractions_and_dimensions_match':True,'all_roots_covered':True,
        'unresolved_frontier_exactly_reconstructed':True,'frontier_boxes':stored.shape[0],
        'root_checkpoint_sha256':sha(directory/'checkpoint.json'),'primary_model_sha256':sha(directory/'primary_model.metal'),
        'reference_model_sha256':sha(directory/'reference_model.metal'),'checker_sha256':sha(__file__),
        'cpu_numerical_fallback':False,'elapsed_seconds':time.monotonic()-start}
    dest=ROOT/'certified_spatial/replay_receipts';dest.mkdir(exist_ok=True)
    (dest/'legacy_complete_reconstruction.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':replay()
