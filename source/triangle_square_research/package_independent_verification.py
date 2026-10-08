"""Package the independent local audits; preserve the original candidate."""
import ast,datetime,hashlib,json
from pathlib import Path
from independent_geometry_gpu import ROOT

DEST=ROOT/'proof_n3/independent_verification'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def imports(path):
    result=[]
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node,ast.Import):result.extend(a.name for a in node.names)
        if isinstance(node,ast.ImportFrom):result.append(node.module or '')
    return result
def literal(path,name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise ValueError(name)

def main():
    legacy=read(DEST/'legacy_independent_verification.json');later=read(DEST/'later_generations_independent.json')
    coverage=read(DEST/'coverage.json');boundary=read(DEST/'boundary_and_construction.json');domains=read(DEST/'continuous_domain_and_precision.json')
    calibration=read(DEST/'calibration.json');chain=read(ROOT/'certified_spatial/replay_receipts/proof_chain_integrity.json')
    for x in [legacy,later,coverage,boundary,domains,calibration,chain]:assert x['passed']
    core=ROOT/'independent_geometry_gpu.py';motion=ROOT/'independent_motion_gpu.py';adapter=ROOT/'verify_later_independent_gpu.py'
    assert legacy['verifier_sha256']==sha(core)==calibration['source_sha256']
    assert coverage['source_sha256']==sha(ROOT/'independent_coverage_gpu.py')
    assert boundary['source_sha256']==sha(ROOT/'independent_boundary_gpu.py')
    assert domains['source_sha256']==sha(ROOT/'independent_seed_coverage_gpu.py')
    expected=[n for n in chain['nodes'] if n['format'] in ['compact-deterministic-replay-v1','strict-descent-reduction-v1','minimum-rotation-reduction-v1']]
    by_directory={r['directory']:r for r in later['generations']}
    assert set(by_directory)=={n['directory'] for n in expected}
    for node in expected:
        r=by_directory[node['directory']]
        assert r['passed'] and r['checkpoint_sha256']==node['checkpoint_sha256']==sha(Path(node['directory'])/'checkpoint.json')
        assert r['verifier_sha256']==sha(core) and r['motion_verifier_sha256']==sha(motion) and r['adapter_sha256']==sha(adapter)
    # Review the metadata copier separately from geometric inference. All
    # archived physical kernels use this exact bounded-output routine.
    output_source=literal(ROOT/'certified_model_gpu.py','SOURCE').strip()
    assert 'for(uint j=0;j<23;j++)updated[23*row+j]=states[23*row+j];' in output_source
    assert 'updated[23*row+3+2*j]' in output_source and 'updated[23*row+4+2*j]' in output_source
    physical_sources=0
    for node in chain['nodes']:
        p=Path(node['directory'])/'primary_model.metal'
        if node['format'] in ['compact-deterministic-replay-v1','legacy-full-history']:
            if not p.exists():continue
            text=p.read_text();position=text.index('uint row=thread_position_in_grid.x')
            assert text[position:].strip()==output_source
            assert 'updated[' not in text[:position]
            physical_sources+=1
    modules=[core,motion,ROOT/'independent_coverage_gpu.py',ROOT/'independent_boundary_gpu.py']
    dependency_review={str(p):imports(p) for p in modules}
    for p,deps in dependency_review.items():
        assert not any(d.startswith(('certified_','contact_catalog_','spatial_minimum_','descent_','critical_family_','n3_branches_')) for d in deps),p
    final=read('/Volumes/SquarePackingProof/SquarePacking/runs/compact_a4156676/checkpoint.json')
    assert final['unresolved_active_boxes']==final['unresolved_precision_boxes']==0 and final['offline_replay_passed']
    generations=later['generations']
    physical=legacy['physical_claims']+sum(r.get('physical_claims',0) for r in generations)
    contractions=legacy['contraction_claims']+sum(r.get('contraction_claims',0) for r in generations)
    inference_rows=physical+contractions
    assert physical==final['excluded_boxes_independently_replayed'] and inference_rows==final['boxes_evaluated']
    evidence_files=[DEST/name for name in ['legacy_independent_verification.json','later_generations_independent.json','coverage.json',
        'boundary_and_construction.json','continuous_domain_and_precision.json','calibration.json','mathematical-review.txt']]
    evidence_files+=modules+[adapter,ROOT/'verify_legacy_independent_gpu.py',ROOT/'independent_seed_coverage_gpu.py',Path(__file__).resolve()]
    result={'status':'independently_checked_computer_assisted_certificate','problem':'Three unit equilateral triangles in a square; disjoint interiors, boundary contact allowed',
        'minimum_claim':'sqrt(3)/2 + sqrt(6)/4','side_display':'1.47839784',
        'all_three_requested_local_audits_complete':True,'coverage_audit_passed':True,'separate_GPU_geometry_and_motion_verifier_passed':True,
        'alternative_boundary_and_exact_construction_review_passed':True,'unverified_recorded_inference_claims':0,
        'contact_cases_checked':coverage['contact_cases_checked'],'continuous_geometry_inferences_checked':inference_rows,
        'physical_rejection_claims_checked':physical,'necessary_contraction_claims_checked':contractions,
        'rigid_motion_removals_checked':sum(r.get('rigid_motion_claims',0) for r in generations),
        'supplied_motion_removals_checked':sum(r.get('supplied_motion_claims',0) for r in generations),
        'exact_algebraic_field_identities_checked':boundary['exact_identities_checked'],'analytic_profiles_checked':boundary['analytic_profiles_checked'],
        'initial_continuous_roots_checked':domains['initial_continuous_roots_checked'],'precision_upgrade_regions_checked':domains['precision_upgrade_input_regions'],
        'later_recorded_generations_verified':len(generations),'linked_generations_in_proof_chain':chain['nodes_checked'],
        'physical_metadata_copy_routines_reviewed':physical_sources,'unresolved_active_regions':0,'unresolved_precision_regions':0,
        'standalone_verifiers_import_original_search_geometry':False,'dependency_review':dependency_review,
        'contact_scope':'Certificates apply to full proper contact profiles. Endpoint/zero-overlap limits are assigned their actual VV/VE/EE profile; additional contacts are not silently ignored.',
        'mathematical_reductions_separately_reviewed_locally':True,'external_peer_review_completed':False,'formal_proof_assistant_checked':False,
        'remaining_trust':'GPU/compiler integer semantics, file/hash/evidence bookkeeping, and the written mathematical lemmas. Independent implementations were written and checked in this local task, not by an external reviewer.',
        'cpu_numerical_fallback':False,'CPU_role':'Orchestration, metadata, code generation, file byte transport and hashes only',
        'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_sha256':{str(p):sha(p) for p in evidence_files}}
    (DEST/'certificate.json').write_text(json.dumps(result,indent=2))
    (DEST/'reproduce.txt').write_text('''Run from the SquarePacking project directory with WD_Game's proof volume mounted.
Numerical geometry and algebra use GPU only. Existing candidate evidence stays unchanged.

python3 triangle_square_research/check_independent_geometry_gpu.py
python3 triangle_square_research/verify_legacy_independent_gpu.py
python3 triangle_square_research/verify_later_independent_gpu.py
python3 triangle_square_research/independent_coverage_gpu.py
python3 triangle_square_research/independent_boundary_gpu.py
python3 triangle_square_research/independent_seed_coverage_gpu.py
python3 triangle_square_research/package_independent_verification.py

Archived producer shaders reconstruct committed claims. They are not the
standalone verifier's rejection predicates. Inspect dependency_review and
the core, motion, coverage and algebra source commitments in certificate.json.
''')
    (DEST/'report.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent n=3 certificate verification</title>
<style>:root{{color-scheme:light dark;--bg:light-dark(#faf9f6,#171916);--fg:light-dark(#252923,#e6e9e3);--soft:light-dark(#edf2ec,#253026);--accent:light-dark(#265939,#afdabb);--line:light-dark(#ccd6ca,#486049)}}body{{margin:0;background:var(--bg);color:var(--fg);font:17px/1.65 system-ui,sans-serif}}main{{max-width:950px;margin:auto;padding:36px 24px 72px}}h1{{font-size:34px;line-height:1.2}}h2{{font-size:23px;margin-top:30px}}a{{color:var(--accent)}}.tag{{color:var(--accent);font-weight:650}}.note{{padding:14px 18px;border-left:4px solid var(--accent);background:var(--soft)}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.stat{{padding:18px;background:var(--soft);border-radius:10px}}.stat b{{display:block;font-size:27px}}table{{width:100%;border-collapse:collapse}}td,th{{padding:12px 8px;border-bottom:1px solid var(--line);text-align:left}}.formula{{font:28px/1.7 Georgia,serif}}@media(max-width:650px){{.stats{{grid-template-columns:1fr}}h1{{font-size:29px}}}}</style>
<main><p class="tag">All three independent local audits passed</p><h1>Three triangles: independently checked certificate</h1>
<p>The separate GPU model has verified every recorded rejection, necessary tightening and motion removal. The coverage reductions, continuous root ranges, precision changes, boundary argument and exact construction also passed their separate checks.</p>
<div class="stats"><div class="stat"><b>0</b>unverified inference claims</div><div class="stat"><b>{inference_rows:,}</b>geometry inferences checked</div><div class="stat"><b>{coverage['contact_cases_checked']:,}</b>contact cases audited</div></div>
<h2>Minimum claim verified by the certificate</h2><div class="formula">sₘᵢₙ = √3/2 + √6/4 ≈ 1.47839784</div>
<table><thead><tr><th>Requested check</th><th>Result</th></tr></thead><tbody>
<tr><td>Every possible optimum represented</td><td>15-degree relation audit matched all contact, spatial and component memberships; continuous charts and all root/precision transitions checked.</td></tr>
<tr><td>Separate GPU geometry and motion verifier</td><td>{physical:,} rejection claims and {contractions:,} necessary contractions checked; all rigid and supplied-motion removals independently established.</td></tr>
<tr><td>Independent boundary argument and construction</td><td>{boundary['exact_identities_checked']} exact identities in a different algebraic-field representation; all {boundary['analytic_profiles_checked']:,} analytic profiles checked.</td></tr>
</tbody></table>
<h2>A scope clarification from the review</h2><p>A vertex–edge tag means contact in the edge's interior. Endpoint contact uses a vertex–vertex tag. Each certificate concerns a full active contact profile; a packing with extra contacts is assigned its actual profile. This is how the proof covers boundary and singular cases without treating an arbitrary requirement prefix as a complete packing description.</p>
<p class="note"><b>Verification status.</b> These are completed independent local implementation checks and a separately derived mathematical review. They are not external peer review or a formal proof-assistant derivation. The original candidate and its evidence have been preserved.</p>
<h2>Inspect and reproduce</h2><p><a href="certificate.json">Verified certificate and source commitments</a> · <a href="mathematical-review.txt">Independent mathematical review</a> · <a href="reproduce.txt">GPU reproduction instructions</a></p><p><a href="../report.html">Original candidate report</a></p></main></html>''')
    status=ROOT/'wd_run_status.json';s=read(status)
    s.update(state='independent_verification_complete',pid=None,worker_pid=None,updated_at=result['created_at'],
        independent_local_verification_complete=True,unresolved_boxes=0,precision_boxes=0,contact_systems=0,
        message='All three independent local audits passed: coverage, separate GPU geometry/motion checks, and alternative exact boundary/construction review. External peer review has not been performed.')
    status.write_text(json.dumps(s,indent=2))
    old_report=ROOT/'proof_n3/report.html';text=old_report.read_text()
    banner='<div class="note" id="independent-update"><b>Update: all three independent local audits have passed.</b> <a href="independent_verification/report.html">Open the independent verification report.</a> This page preserves the earlier candidate status.</div>'
    if 'id="independent-update"' not in text:text=text.replace('<main>','<main>'+banner,1)
    old_report.write_text(text)
    contacts=ROOT/'contacts-report.html';text=contacts.read_text()
    if 'id="independent-validation-update"' not in text:
        text=text.replace('<main>','<main><div class="note" id="independent-validation-update"><b>All three independent local audits passed.</b> <a href="proof_n3/independent_verification/report.html">Open the independently checked certificate.</a></div>',1)
    contacts.write_text(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ['file_sha256','dependency_review']},indent=2),flush=True)

if __name__=='__main__':main()
