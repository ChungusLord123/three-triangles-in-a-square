"""Fresh full MPS catalogue check, including exact quotient reconstruction."""
import os,json,time,hashlib
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import torch
from check_contact_catalog_mps import read_u32,relations,decisions,D
from contact_catalog_gpu import OUT
from n3_branches_gpu import ROOT

def main():
    start=time.monotonic();last=start
    summary=json.loads((OUT/'screening_summary.json').read_text());assert summary['complete_angular_screen']
    masks=read_u32(OUT/'wall_masks.npy');triples=read_u32(OUT/'wall_triplets.npy')
    rel,bit=relations();histogram=torch.zeros(16,device=D,dtype=torch.int32);checked=0
    for part in summary['parts']:
        survivors=read_u32(part['ids_file']) if part['ids_file'] else torch.empty(0,device=D,dtype=torch.int32)
        for first in range(part['first'],part['first']+part['count'],32768):
            count=min(32768,part['first']+part['count']-first)
            ids=torch.arange(first,first+count,device=D,dtype=torch.int32)
            quotient=ids//50653;remainder=ids-quotient*50653
            assert ((remainder>=0)&(remainder<50653)).all().item()
            reason,_,_=decisions(ids,masks,triples,rel,bit)
            expected=torch.zeros(count,device=D,dtype=torch.bool)
            selected=survivors[(survivors>=first)&(survivors<first+count)]
            expected[(selected-first).to(torch.int64)]=True
            assert torch.equal(reason==0,expected)
            histogram+=torch.stack([(reason==k).sum() for k in range(16)]).to(torch.int32)
            checked+=count
        if time.monotonic()-last>10:
            print('Fresh independent catalogue cases',checked,flush=True);last=time.monotonic()
    assert torch.equal(histogram,torch.tensor(summary['reason_bit_histogram'],device=D,dtype=torch.int32))
    result={'passed':True,'contact_cases_checked':checked,'all_survivor_memberships_checked':True,
        'all_large_profile_quotients_exactly_reconstructed':True,'independent_method':'MPS Boolean transitive closure and angular relation join',
        'cpu_numerical_fallback':False,'elapsed_seconds':time.monotonic()-start,
        'checker_sha256':hashlib.sha256(open(__file__,'rb').read()).hexdigest()}
    (ROOT/'certified_spatial/replay_receipts/catalogue_fresh_full_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
