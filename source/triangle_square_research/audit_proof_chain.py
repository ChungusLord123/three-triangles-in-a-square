"""Verify saved replay receipts and byte commitments throughout the proof tree.

This module performs file/metadata validation, never numerical geometry. GPU
root, interval, motion and algebraic checks are in the referenced receipts.
"""
import json,time,hashlib,struct,ast
from pathlib import Path
from n3_branches_gpu import ROOT

CACHE={}
def sha(path):
    p=Path(path).resolve();key=(str(p),p.stat().st_size,p.stat().st_mtime_ns)
    if key not in CACHE:
        h=hashlib.sha256()
        with p.open('rb') as f:
            while True:
                data=f.read(32*1024*1024)
                if not data:break
                h.update(data)
        CACHE[key]=h.hexdigest()
    return CACHE[key]
def json_read(path):return json.loads(Path(path).read_text())
def check_map(mapping):
    for p,h in mapping.items():assert sha(p)==h,str(p)
def payload_sha(files):
    h=hashlib.sha256()
    for path in files:
        with Path(path).open('rb') as f:
            head=f.read(8);assert head[:6]==b'\x93NUMPY'
            length=struct.unpack('<H' if head[6]==1 else '<I',f.read(2 if head[6]==1 else 4))[0]
            meta=ast.literal_eval(f.read(length).decode('latin1'));assert not meta['fortran_order']
            while True:
                data=f.read(32*1024*1024)
                if not data:break
                h.update(data)
    return h.hexdigest()

def main():
    start=time.monotonic();last=start
    current=Path('/Volumes/SquarePackingProof/SquarePacking/runs/compact_a4156676')
    nodes=[];seen=set()
    while str(current) not in seen:
        seen.add(str(current));cp=json_read(current/'checkpoint.json');audit=json_read(current/'checkpoint_audit.json')
        assert audit['passed']
        if 'format' in cp:assert cp['offline_replay_passed']
        fmt=cp.get('format','legacy-full-history')
        if fmt=='compact-deterministic-replay-v1':
            conf=json_read(current/'config.json');basis=json_read(current/'basis.json')
            assert sha(basis['parent_checkpoint'])==basis['parent_checkpoint_sha256']
            assert Path(basis['parent_checkpoint']).parent==Path(cp['parent_directory'])
            for part in basis['frontier']+basis['terminal']:assert sha(part['path'])==part['sha256']
            check_map(conf['code_file_sha256']);check_map(conf['input_file_sha256'])
            assert sha(current/'primary_model.metal')==conf['primary_model_sha256']==cp['primary_model_sha256']
            assert sha(current/'reference_model.metal')==conf['reference_model_sha256']
            assert sha(current/'journal.jsonl')==cp['journal_sha256']
            for key in ['new_boxes_evaluated','new_interval_exclusions']:
                assert audit[key]==cp[key]
            assert audit['saved_splits_checked']==cp['new_splits']
            assert audit['frontier_boxes']==cp['unresolved_active_boxes']
            assert audit['all_intermediate_hashes_matched'] and audit['every_exclusion_rechecked_with_separate_arithmetic']
            parent=json_read(basis['parent_checkpoint'])
            if basis.get('kind')=='side_shell':assert parent['complete_below_target_exclusion']
            else:assert [f['path'] for f in basis['frontier']]==parent['frontier_files']
            assert [f['path'] for f in basis['terminal']]==parent['terminal_files']
        elif fmt=='strict-descent-reduction-v1':
            cert=json_read(current/'descent_certificate.json')
            assert cert['passed'] and audit['rational_witness_replay_passed']
            assert sha(cert['parent_checkpoint'])==cert['parent_sha256']
            assert Path(cert['parent_checkpoint']).parent==Path(cp['parent_directory'])
            check_map(cert['code_file_sha256']);check_map(cert['witness_file_sha256'])
            parent=json_read(cert['parent_checkpoint'])
            assert payload_sha(parent['frontier_files']+parent['terminal_files'])==cert['source_sha256']
            assert payload_sha(cp['frontier_files'])==cert['result_sha256']
            assert cert['remaining_regions']==cp['unresolved_active_boxes']
            assert cert['regions_with_strict_descent']==audit['regions_removed']
        elif fmt=='precision-upgrade-v1':
            cert=json_read(current/'precision_upgrade.json');assert cert['passed']
            assert sha(cert['parent_checkpoint'])==cert['parent_sha256']
            for item in cert['input_files']:assert sha(item['path'])==item['sha256']
            assert payload_sha(cp['frontier_files'])==cert['retained_sha256']
            assert cert['independent_MPS_lift_check_passed'] and cert['all_old_terminal_regions_reconsidered']
        elif fmt=='minimum-rotation-reduction-v1':
            cert=json_read(current/'minimum_rotation_check.json');assert cert['passed']
            assert sha(cert['parent_checkpoint'])==cert['parent_checkpoint_sha256']
            check_map(cert['input_file_sha256'])
            parent=json_read(cert['parent_checkpoint'])
            assert payload_sha(parent['frontier_files'])==cert['input_frontier_sha256']
            assert payload_sha(cp['frontier_files'])==cert['output_frontier_sha256']
            assert cert['independent_reference_all_rejections_passed']
        elif fmt=='exact-boundary-seed-v1':
            cert=json_read(current/'boundary_seed_certificate.json');assert cert['passed']
            assert sha(cert['parent_checkpoint'])==cert['parent_sha256']
            assert sha(cert['analytic_certificate'])==cert['analytic_certificate_sha256']
            check_map(cert['code_file_sha256']);check_map(cert['input_file_sha256'])
            assert payload_sha(cp['frontier_files'])==cert['frontier_sha256']
            independent=json_read(ROOT/'certified_spatial/replay_receipts'/f'{current.name}_independent_seed.json')
            assert independent['passed'] and independent['entire_root_partition_independently_checked_on_MPS']
        else:
            assert fmt=='legacy-full-history'
            assert audit['all_split_children_cover_parent'] and audit['root_ordinal_ranges_contiguous']
            assert sha(current/'primary_model.metal')==cp['primary_model_sha256']
            for p,h in audit['input_data_sha256'].items():assert sha(ROOT/p)==h
        nodes.append({'directory':str(current),'format':fmt,'checkpoint_sha256':sha(current/'checkpoint.json'),
            'audit_sha256':sha(current/'checkpoint_audit.json'),'passed':True})
        if time.monotonic()-last>10:print('Proof chain verified nodes',len(nodes),flush=True);last=time.monotonic()
        if not cp.get('parent_directory'):break
        current=Path(cp['parent_directory'])
    final=json_read(Path(nodes[0]['directory'])/'checkpoint.json')
    assert final['exact_boundary_remaining_catalogue_cleared']
    assert final['unresolved_active_boxes']==0 and final['unresolved_precision_boxes']==0
    result={'passed':True,'nodes_checked':len(nodes),'nodes':nodes,'all_code_input_parent_and_journal_hashes_match':True,
        'all_saved_GPU_replay_receipts_passed':True,'all_boundary_regions_disposed':True,
        'geometry_check_role':'None; file and metadata checks only. The separate GPU receipts supply numerical verification.',
        'elapsed_seconds':time.monotonic()-start}
    (ROOT/'certified_spatial/replay_receipts/proof_chain_integrity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'},indent=2),flush=True)

if __name__=='__main__':main()
