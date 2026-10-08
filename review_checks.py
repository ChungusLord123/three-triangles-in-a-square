"""Portable lightweight review checks; this is not full proof replay."""
import argparse,json,sys,hashlib
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--coverage',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parent
    source=root/'source/triangle_square_research'
    sys.path.insert(0,str(source))
    import mlx.core as mx
    import torch
    assert mx.metal.is_available() and torch.backends.mps.is_available(),'Metal GPU and MPS are required; no CPU fallback is enabled.'
    mx.set_default_device(mx.gpu)
    import check_independent_geometry_gpu as calibration
    import independent_boundary_gpu as boundary
    (source/'proof_n3/independent_verification').mkdir(parents=True,exist_ok=True)
    calibration.main()
    checks_mlx=boundary.algebra(boundary.FieldPolynomials('mlx'))
    checks_mps=boundary.algebra(boundary.FieldPolynomials('mps'))
    assert checks_mlx==checks_mps
    output=root/'review_output';output.mkdir(exist_ok=True)
    result={'passed':True,'standalone_geometry_calibration_passed':True,'exact_field_identities_checked':len(checks_mlx),
        'both_MLX_and_MPS_passed':True,'full_numerical_certificate_replayed':False,
        'scope':'Lightweight sanity checks for code, exact construction and algebra. Full proof replay requires omitted archival data.',
        'cpu_numerical_fallback':False,'verifier_source_sha256':hashlib.sha256((source/'independent_geometry_gpu.py').read_bytes()).hexdigest()}
    (output/'lightweight_checks.json').write_text(json.dumps(result,indent=2))
    if a.coverage:
        import independent_coverage_gpu as coverage
        original=coverage.load
        def local(path):
            path=Path(path)
            if 'contact_catalog' in path.parts:path=source/'contact_catalog'/path.name
            elif 'certified_spatial' in path.parts:path=source/'certified_spatial'/path.name
            return original(path)
        coverage.load=local;coverage.main()
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
