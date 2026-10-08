"""Package verified receipts without changing their mathematical scope."""
import json,html,hashlib,datetime
from pathlib import Path
from n3_branches_gpu import ROOT
from make_contact_report import exact_reference_display

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    dest=ROOT/'proof_n3';dest.mkdir(exist_ok=True)
    receipts=ROOT/'certified_spatial/replay_receipts'
    final_dir=Path('/Volumes/SquarePackingProof/SquarePacking/runs/compact_a4156676')
    final=read(final_dir/'checkpoint.json');final_audit=read(final_dir/'checkpoint_audit.json')
    chain=read(receipts/'proof_chain_integrity.json');legacy=read(receipts/'legacy_complete_reconstruction.json')
    catalogue=read(receipts/'catalogue_fresh_full_check.json');geometry=read(ROOT/'certified_spatial/geometric_semantics_certificate.json')
    seed=read(receipts/'boundary_seed_8e6474bd_independent_seed.json')
    for obj in [final_audit,chain,legacy,catalogue,geometry,seed]:assert obj['passed']
    assert final['offline_replay_passed'] and final['exact_boundary_remaining_catalogue_cleared']
    assert final['unresolved_active_boxes']==0 and final['unresolved_precision_boxes']==0
    assert chain['nodes'][0]['checkpoint_sha256']==sha(final_dir/'checkpoint.json')
    assert chain['nodes'][-1]['checkpoint_sha256']==legacy['root_checkpoint_sha256']
    assert legacy['checker_sha256']==sha(ROOT/'replay_legacy_tree_gpu.py')
    assert geometry['source_sha256']==sha(ROOT/'geometric_semantics_gpu.py')
    assert catalogue['checker_sha256']==sha(ROOT/'recheck_catalogue_gpu.py')
    assert (dest/'coverage-review.txt').is_file()
    paths=[final_dir/'checkpoint.json',final_dir/'checkpoint_audit.json',receipts/'proof_chain_integrity.json',
        receipts/'legacy_complete_reconstruction.json',receipts/'catalogue_fresh_full_check.json',
        receipts/'boundary_seed_8e6474bd_independent_seed.json',ROOT/'certified_spatial/geometric_semantics_certificate.json',
        dest/'coverage-review.txt',ROOT/'geometric_semantics_gpu.py',ROOT/'replay_legacy_tree_gpu.py',
        ROOT/'audit_proof_chain.py',ROOT/'recheck_catalogue_gpu.py',ROOT/'audit_boundary_seed_gpu.py',
        ROOT/'exact_boundary_runner_gpu.py',ROOT/'compact_boundary_search_gpu.py']
    certificate={'status':'complete_computational_proof_candidate','problem':'Three unit equilateral triangles in a square; disjoint interiors, boundary contact allowed',
        'claim':'s_min = sqrt(3)/2 + sqrt(6)/4','claimed_side_display':exact_reference_display(),
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'search_coverage_completed':True,'unresolved_active_regions':0,'unresolved_precision_regions':0,
        'contact_cases_independently_checked':catalogue['contact_cases_checked'],
        'replay_generations_checked':chain['nodes_checked'],'analytic_boundary_profiles':seed['analytic_covered_profiles'],
        'exact_geometry_identities_checked_on_both_GPUs':geometry['exact_polynomial_identities'],
        'legacy_boxes_freshly_reconstructed':legacy['boxes_reconstructed'],
        'legacy_physical_exclusions_freshly_rechecked':legacy['physical_exclusions_freshly_rechecked'],
        'legacy_splits_freshly_checked_on_MPS':legacy['splits_independently_rechecked_on_MPS'],
        'all_legacy_parent_queue_lineages_and_interrupt_carryovers_verified':True,
        'exact_feasible_construction_verified':True,'logical_coverage_argument_written':True,
        'cpu_numerical_fallback':False,'CPU_role':'Orchestration, metadata and file byte transport/hashing only',
        'independently_implemented_entire_spatial_geometric_model':False,
        'externally_reviewed_mathematical_argument':False,'formal_proof_assistant_checked':False,
        'global_optimality_proved':False,
        'trust_boundary':'Complete replayable computational certificate and written coverage/source-review argument. Independent arithmetic, catalogue, bisection and polynomial checks do not replace independent review of the common spatial geometric model and logical reductions.',
        'final_checkpoint':str(final_dir/'checkpoint.json'),'file_sha256':{str(p):sha(p) for p in paths}}
    (dest/'certificate.json').write_text(json.dumps(certificate,indent=2))
    replay_text='''GPU replay instructions (Apple Metal / MLX and PyTorch MPS)

Run from the SquarePacking project directory. Retained checkpoints must be
mounted at /Volumes/SquarePackingProof; code/input hashes enforce the recorded
versions. Do not edit files in accepted proof generations. CPU is used for
file I/O, byte hashes and orchestration; numerical geometry stays on GPU.

python3 triangle_square_research/geometric_semantics_gpu.py
python3 triangle_square_research/recheck_catalogue_gpu.py
python3 triangle_square_research/replay_legacy_tree_gpu.py
python3 triangle_square_research/audit_boundary_seed_gpu.py /Volumes/SquarePackingProof/SquarePacking/runs/boundary_seed_8e6474bd
python3 triangle_square_research/exact_boundary_runner_gpu.py --replay /Volumes/SquarePackingProof/SquarePacking/runs/compact_a4156676
python3 triangle_square_research/audit_proof_chain.py
python3 triangle_square_research/assemble_n3_certificate.py

The complete ordered checkpoint ancestry and per-generation replay receipts
are in proof_chain_integrity.json. Each generation also retains its frozen
controller/model sources, parent/input commitments and replay journal or
rational velocity witnesses. Its original scope is preserved.
'''
    (dest/'replay-instructions.txt').write_text(replay_text)
    (dest/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Three triangles: completed GPU certificate</title>
<style>:root{{color-scheme:light dark;--bg:light-dark(#faf9f6,#171916);--fg:light-dark(#252923,#e6e9e3);--soft:light-dark(#edf2ec,#253026);--line:light-dark(#cbd6ca,#486049);--muted:light-dark(#586456,#bcc8b8);--accent:light-dark(#265939,#afdabb)}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 system-ui,sans-serif}}main{{max-width:950px;margin:auto;padding:38px 24px 72px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px;margin-top:30px}}.tag{{color:var(--accent);font-weight:650}}.note{{border-left:4px solid var(--accent);padding:12px 18px;background:var(--soft);margin:24px 0}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.stat{{background:var(--soft);padding:18px;border-radius:10px}}.stat b{{display:block;font-size:28px}}.formula{{font:27px/1.7 Georgia,serif;overflow:auto}}a{{color:var(--accent)}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:12px 8px;border-bottom:1px solid var(--line)}}.muted{{color:var(--muted)}}@media(max-width:650px){{.stats{{grid-template-columns:1fr}}h1{{font-size:29px}}}}</style>
<main><p class="tag">GPU certificate complete · mathematical review pending</p><h1>Three triangles in a square</h1>
<p>The exhaustive minimum-profile computation has finished. Every retained region is excluded by a checked certificate or covered by an exact algebraic lower bound. The active and precision-limit queues are both empty.</p>
<div class="stats"><div class="stat"><b>0</b>unresolved regions</div><div class="stat"><b>{catalogue['contact_cases_checked']:,}</b>contact cases checked</div><div class="stat"><b>{chain['nodes_checked']}</b>verified generations</div></div>
<h2>Claimed exact optimum</h2><div class="formula">sₘᵢₙ = √3/2 + √6/4 ≈ {html.escape(certificate['claimed_side_display'])}</div>
<p>The exact construction has unit triangle edges, stays inside the square, and has disjoint interiors. Boundary contact is allowed. Its geometry was checked using exact polynomial coefficients in both GPU implementations.</p>
<div class="note"><b>Evidence boundary.</b> This is a complete computer-assisted proof candidate. The written coverage argument and the common geometric contractor model still need independent mathematical review. Different interval arithmetic, exact child-union checks and polynomial identities are valuable cross-checks; they are not a second independent implementation of the entire spatial geometric model. This report does not label the theorem independently established.</div>
<h2>What was verified</h2><table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>
<tr><td>Entire contact catalogue</td><td>{catalogue['contact_cases_checked']:,} cases matched an independent GPU algorithm</td></tr>
<tr><td>Original tree reconstructed</td><td>{legacy['boxes_reconstructed']:,} boxes; all parent queues and interrupted-run carryovers matched</td></tr>
<tr><td>Original physical exclusions replayed</td><td>{legacy['physical_exclusions_freshly_rechecked']:,} checked again with separate interval arithmetic</td></tr>
<tr><td>Original subdivision coverage</td><td>{legacy['splits_independently_rechecked_on_MPS']:,} exact child unions checked on MPS</td></tr>
<tr><td>Exact geometry and construction</td><td>{geometry['exact_polynomial_identities']} coefficient identities checked on both GPUs</td></tr>
<tr><td>Boundary family coverage</td><td>{seed['analytic_covered_profiles']:,} profiles covered by the exact algebraic lower bound</td></tr>
<tr><td>Final interval and motion certificates</td><td>All remaining regions resolved and replayed</td></tr>
<tr><td>Files and provenance</td><td>{chain['nodes_checked']} linked generations; all committed code, inputs, parents and journals matched</td></tr>
</tbody></table>
<h2>Review and reproduce</h2><p><a href="certificate.json">Certificate and file commitments</a> · <a href="coverage-review.txt">Complete coverage argument and trust boundary</a> · <a href="replay-instructions.txt">GPU replay instructions</a></p>
<p>The retained proof generations remain on the dedicated 1 TB volume on WD_Game. Numeric geometry runs on the GPU; the CPU handles orchestration, metadata, file transport and hashes.</p>
<p class="muted">The packing value is the existing n=3 construction listed on <a href="https://erich-friedman.github.io/packing/triinsqu/">Erich Friedman's Triangles in Squares page</a>, credited there to Friedman in 1996. This work concerns its optimality certificate.</p></main></html>''')
    status=ROOT/'wd_run_status.json';s=read(status)
    s.update(state='complete_computational_certificate',pid=None,worker_pid=None,
        updated_at=certificate['created_at'],unresolved_boxes=0,precision_boxes=0,contact_systems=0,
        last_verified_boundary_checkpoint=str(final_dir/'checkpoint.json'),exact_optimality_proved=False,
        computational_certificate_complete=True,target_text='s0 exact boundary',
        message='Complete GPU proof candidate: no unresolved regions; all replays, catalogue, ancestry and exact construction checks passed. Independent review of the shared geometric model and logical coverage remains.')
    status.write_text(json.dumps(s,indent=2))
    report=ROOT/'contacts-report.html';page=report.read_text()
    banner='<div class="note" id="final-certificate-banner"><b>The GPU proof candidate is complete: 0 unresolved regions.</b> All checkpoints, exact construction and coverage checks passed. Independent mathematical review remains. <a href="proof_n3/report.html">Open the complete certificate and review argument.</a></div>'
    if 'id="final-certificate-banner"' not in page:page=page.replace('<main>','<main>'+banner,1)
    page=page.replace('Other spatial contact systems remain to be certified.','The completed boundary certificate covers all other retained profiles; independent review of the geometric model remains.')
    page=page.replace('Any remaining boxes must be resolved with stronger valid constraints or further certified subdivision. A completed rational target still leaves its gap to s₀. A proof of the exact optimum also needs certified local bounds at all surviving equality or limiting families.',
        'The exact-boundary interval is now fully covered, including the singular equality family. Both unresolved queues are empty. The complete computation and its written coverage argument are available in the linked proof-candidate report; the remaining step is independent mathematical review.')
    report.write_text(page)
    print(json.dumps({k:v for k,v in certificate.items() if k!='file_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
