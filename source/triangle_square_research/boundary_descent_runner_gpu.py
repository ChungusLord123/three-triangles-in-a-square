"""Strict descent certificates with exact-boundary rather than rational scope."""
import argparse,json
from pathlib import Path
import descent_witness_search_v2_gpu as solver
from n3_branches_gpu import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--output-base',required=True)
    a=p.parse_args();cp=json.loads((Path(a.parent)/'checkpoint.json').read_text())
    assert cp['exact_boundary_goal'] and cp['offline_replay_passed']
    directory=solver.reduce(a.parent,a.output_base);solver.replay(directory,publish=False)
    checkpoint=directory/'checkpoint.json';updated=json.loads(checkpoint.read_text())
    updated.update(complete_below_target_exclusion=False,certified_lower_bound=None,
        exact_boundary_remaining_catalogue_cleared=updated['all_minimum_candidate_regions_excluded'],
        scope='Exact-boundary shell after separate analytic family coverage. Strict-descent certificates exclude full minimum profiles; no rational exclusion above the exact optimum is claimed.',global_optimality_proved=False)
    checkpoint.write_text(json.dumps(updated,indent=2))
    (directory/'boundary_scope_receipt.json').write_text(json.dumps({'passed':True,
        'rational_claim_removed':True,'exact_boundary_goal':True,'independent_witness_replay_passed':True,
        'global_optimality_proved':False},indent=2))
    (ROOT/'certified_spatial/latest_boundary_run.json').write_text(json.dumps({'directory':str(directory)},indent=2))

if __name__=='__main__':main()
