"""Verify later recorded generations with a standalone geometric verifier.

Archived shaders are evidence producers, not trusted rejection predicates.
Their outputs must match committed journals. The independent model checks
every physical rejection, every contraction, and every claimed motion removal.
Existing producer modules are never imported or executed as Python code.
"""
import argparse,ast,hashlib,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from independent_geometry_gpu import ROOT,load_file,source_commitment
from independent_motion_gpu import MotionVerifier,commitment as motion_commitment
mx.set_default_device(mx.gpu)
DEST=ROOT/'proof_n3/independent_verification'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a):
    if a is None:return None
    mx.eval(a);return hashlib.sha256(np.asarray(a).tobytes(order='C')).hexdigest()
def cat(parts,cols=23):return mx.concatenate(parts,axis=0) if parts else mx.zeros((0,cols),dtype=mx.int32)
def take(rows,mask):
    count=mx.sum(mask.astype(mx.int32)).item()
    if not count:return None
    indices=mx.argsort(mx.where(mask,mx.arange(mask.size),2147483647))[:count]
    return rows[indices]
def read(path):return json.loads(Path(path).read_text())
def literal(path,name):
    for item in ast.parse(Path(path).read_text()).body:
        if isinstance(item,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in item.targets):return ast.literal_eval(item.value)
    raise ValueError(name)

CO=r'''
out[0]=roots[0];out[1]=roots[1];out[2]=roots[2];out[3]=roots[3];out[4]=UNIT;
out[5]=floor_div(fraction[0]*UNIT,fraction[1]);out[6]=ceil_div(fraction[0]*UNIT,fraction[1]);
out[7]=roots[6];out[8]=roots[7];out[9]=0;
'''

class Evidence:
    def __init__(self,directory,verifier):
        self.directory=Path(directory);self.v=verifier;self.cp=read(self.directory/'checkpoint.json')
        self.conf=read(self.directory/'config.json') if (self.directory/'config.json').exists() else {}
        self.masks=verifier.masks;self.triples=verifier.triples
        allowed=load_file(ROOT/'contact_catalog/wall_angles.npy')
        self.angle_masks=mx.sum(allowed.astype(mx.uint32)*(mx.array(1,dtype=mx.uint32)<<mx.arange(12,dtype=mx.uint32)),axis=1)
        constant=mx.fast.metal_kernel(name=f'independent_evidence_constants_{verifier.bits}',input_names=['roots','fraction'],output_names=['out'],header=verifier.header,source=CO)
        self.co=constant(inputs=[verifier.roots,mx.array([self.cp['target_numerator'],self.cp['target_denominator']],dtype=mx.int64)],
            grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(10,)],output_dtypes=[mx.int64])[0]
        if verifier.bits==24:self.co=self.co.astype(mx.int32)
        self.header=None;self.primary=None;self.minimum=None
        if (self.directory/'config.json').exists():
            text=(self.directory/'primary_model.metal').read_text();position=text.index('uint row=thread_position_in_grid.x')
            assert sha(self.directory/'primary_model.metal')==self.conf['primary_model_sha256']
            self.header=text[:position]
            assert 'constants_data[4]' not in self.header
            self.primary=mx.fast.metal_kernel(name='frozen_evidence_primary_'+self.conf['primary_model_sha256'][:10],
                input_names=['states','wmasks','triples','angle_masks','constants_data','control','amount'],
                output_names=['updated','reasons','split_variable'],header=self.header,source=text[position:])
            if any('minimum_classification_sha256' in r for r in self.records()):
                minimum_source=literal(ROOT/'spatial_minimum_rotation_gpu.py','SOURCE')
                self.minimum=mx.fast.metal_kernel(name='frozen_evidence_minimum_'+self.conf['primary_model_sha256'][:10],
                    input_names=['states','wmasks','triples','constants_data','amount'],output_names=['keep'],header=self.header,source=minimum_source)
    def records(self):return [json.loads(line) for line in (self.directory/'journal.jsonl').read_text().splitlines()]
    def produce(self,states):
        parts=[]
        for first in range(0,states.shape[0],4096):
            a=states[first:first+4096]
            result=self.primary(inputs=[a,self.masks,self.triples,self.angle_masks,self.co,mx.array([self.conf['rounds'],1],dtype=mx.uint32),mx.array([a.shape[0]],dtype=mx.uint32)],
                grid=(a.shape[0],1,1),threadgroup=(128,1,1),output_shapes=[a.shape,(a.shape[0],),(a.shape[0],)],output_dtypes=[mx.int32,mx.uint32,mx.int32])
            mx.eval(*result);parts.append(result)
        return tuple(mx.concatenate([p[j] for p in parts]) for j in range(3))
    def produce_minimum(self,states):
        parts=[]
        for first in range(0,states.shape[0],4096):
            a=states[first:first+4096]
            r=self.minimum(inputs=[a,self.masks,self.triples,self.co,mx.array([a.shape[0]],dtype=mx.uint32)],
                grid=(a.shape[0],1,1),threadgroup=(128,1,1),output_shapes=[(a.shape[0],)],output_dtypes=[mx.uint32])[0]
            mx.eval(r);parts.append(r)
        return mx.concatenate(parts)

def roots_for_shell(basis,evidence):
    ids=load_file(ROOT/'certified_spatial/spatial_candidates.npy').astype(mx.int32)
    lookup=load_file(ROOT/'certified_spatial/rotation_critical_lookup.npy')
    ids=take(ids[:,None],lookup[ids]>0)[:,0]
    ordinals=mx.arange(ids.size*8,dtype=mx.uint32)
    unit=mx.array(1,dtype=mx.int64)<<mx.array(evidence.v.bits,dtype=mx.int64)
    half_value=(evidence.co[6].astype(mx.int64)+1)>>1
    domain=mx.concatenate([mx.broadcast_to(mx.stack([-half_value,half_value])[None,:],(6,2)),
        mx.broadcast_to(mx.stack([-unit,unit])[None,:],(3,2)),
        mx.stack([mx.array(basis['side_lower_fixed'],dtype=mx.int64),evidence.co[6].astype(mx.int64)])[None,:]],axis=0).astype(mx.int32).reshape(20)
    meta=mx.stack([ids[ordinals>>3],(ordinals&7).astype(mx.int32),mx.zeros_like(ordinals).astype(mx.int32)],axis=1)
    return mx.concatenate([meta,mx.broadcast_to(domain[None,:],(ordinals.size,20))],axis=1)

def split(kept,dimension,bits,minimum):
    ds=mx.maximum(dimension,0)
    low=mx.take_along_axis(kept[:,3::2],ds[:,None],axis=1)[:,0].astype(mx.int64)
    high=mx.take_along_axis(kept[:,4::2],ds[:,None],axis=1)[:,0].astype(mx.int64)
    branch=(dimension>=0)&(high-low>minimum)&(kept[:,2]<(360 if bits==30 else 180))
    parents=take(kept,branch);terminal=take(kept,~branch)
    if parents is None:return None,terminal
    dims=take(dimension[:,None],branch)[:,0]
    lo=mx.take_along_axis(parents[:,3::2],dims[:,None],axis=1)[:,0].astype(mx.int64)
    hi=mx.take_along_axis(parents[:,4::2],dims[:,None],axis=1)[:,0].astype(mx.int64)
    mid=lo+((hi-lo)>>1);indices=mx.arange(parents.shape[0])
    left=parents.at[indices,4+2*dims].add((mid-hi).astype(mx.int32)).at[:,2].add(1)
    right=parents.at[indices,3+2*dims].add((mid-lo).astype(mx.int32)).at[:,2].add(1)
    return mx.stack([left,right],axis=1).reshape(-1,23),terminal

def require_motion(v,states,velocity=None):
    ok=v.motion_check(states,velocity)
    if mx.all(ok>0).item():return
    missing=take(states,ok==0)
    if missing is not None and mx.all(v.contract(missing,16)[1]!=0).item():return
    for name,a in [('states',states),('accepted',ok)]:mx.save(str(DEST/f'later_unverified_motion_{name}.npy'),a)
    if velocity is not None:mx.save(str(DEST/'later_unverified_motion_velocity.npy'),velocity)
    raise RuntimeError('Independent motion/infeasibility verifier did not certify every claimed removal.')

def compact(directory,v):
    start=time.monotonic();e=Evidence(directory,v);basis=read(e.directory/'basis.json')
    current=roots_for_shell(basis,e) if basis.get('kind')=='side_shell' else cat([load_file(a['path']) for a in basis['frontier']])
    lookup=load_file(ROOT/'certified_spatial/rotation_critical_lookup.npy')
    filtered=take(current,lookup[current[:,0]]>0);current=filtered if filtered is not None else cat([])
    records=e.records();next_parts=[];terminals=[];processed=0;checked=physical=contractions=minimum_removed=0;last=time.monotonic()
    for r in records:
        kind=r['kind']
        if kind=='rotation_filter':assert r['retained_rows']==current.shape[0] and r['retained_sha256']==digest(current)
        elif kind=='pass_start':
            assert r['rows']==current.shape[0] and r['sha256']==digest(current);next_parts=[];processed=0
        elif kind=='batch':
            assert r['first']==processed
            states=current[processed:processed+r['boxes']];assert digest(states)==r['input_sha256']
            claimed,reasons,dimensions=e.produce(states)
            assert digest(reasons)==r['reasons_sha256']
            accepted,refined,verdict=v.verify_transition(states,claimed,reasons,12)
            if not mx.all(accepted).item():
                for name,a in [('states',states),('claimed',claimed),('reasons',reasons),('refined',refined),('verdict',verdict),('accepted',accepted)]:mx.save(str(DEST/f'later_unverified_{name}.npy'),a)
                raise RuntimeError('Independent model left a contraction or rejection unverified at '+str(directory))
            rejected=(reasons!=0)&(reasons!=9);n=mx.sum(rejected.astype(mx.int32)).item();assert n==r['excluded']
            physical+=n;contractions+=mx.sum((reasons==0).astype(mx.int32)).item()
            kept=take(claimed,~rejected);dims=take(dimensions[:,None],~rejected)
            if kept is not None and 'minimum_classification_sha256' in r:
                classification=e.produce_minimum(kept);assert digest(classification)==r['minimum_classification_sha256']
                removed=take(kept,classification==0)
                if removed is not None:require_motion(v,removed);minimum_removed+=removed.shape[0]
                dims=take(dims,classification>0);kept=take(kept,classification>0)
            elif 'minimum_classification_sha256' in r:assert r['minimum_classification_sha256'] is None
            assert digest(kept)==r['contracted_sha256'] and digest(dims)==r['dimensions_sha256']
            children=terminal=None
            if kept is not None:children,terminal=split(kept,dims[:,0],v.bits,e.conf['minimum_split_width'])
            assert digest(children)==r['children_sha256'] and digest(terminal)==r['terminal_sha256']
            if children is not None:next_parts.append(children)
            if terminal is not None:terminals.append(terminal)
            processed+=r['boxes'];checked+=r['boxes']
            if time.monotonic()-last>10:print(directory.name,'independent rows',checked,flush=True);last=time.monotonic()
        elif kind=='pass_end':
            assert processed==r['processed_rows']
            if processed<current.shape[0]:next_parts.append(current[processed:])
            current=cat(next_parts);assert current.shape[0]==r['next_rows'] and digest(current)==r['next_sha256']
        elif kind=='final':assert current.shape[0]==r['frontier_rows'] and digest(current)==r['frontier_sha256']
        else:raise ValueError(kind)
    stored=cat([load_file(p) for p in e.cp['frontier_files']]);assert stored.shape==current.shape and mx.all(stored==current).item()
    term=cat(terminals)
    if term.shape[0]:assert mx.all(term==load_file(e.directory/'terminal.npy')).item()
    assert checked==e.cp['new_boxes_evaluated'] and physical==e.cp['new_interval_exclusions']
    assert minimum_removed==e.cp.get('new_minimum_rotation_regions_removed',0)
    return {'passed':True,'rows':checked,'physical_claims':physical,'contraction_claims':contractions,
        'rigid_motion_claims':minimum_removed,'all_journal_commitments_matched':True,'elapsed_seconds':time.monotonic()-start}

def nearest_header(parent):
    current=Path(parent)
    while True:
        if (current/'config.json').exists():
            text=(current/'primary_model.metal').read_text();return text[:text.index('uint row=thread_position_in_grid.x')]
        current=Path(read(current/'checkpoint.json')['parent_directory'])

def descent(directory,v):
    start=time.monotonic();cp=read(directory/'checkpoint.json');cert=read(directory/'descent_certificate.json');parent=read(cert['parent_checkpoint'])
    source=cat([load_file(p) for p in parent['frontier_files']+parent['terminal_files']]);assert digest(source)==cert['source_sha256']
    keys=load_file(directory/'prototype_keys.npy');velocities=load_file(directory/'velocities.npy');good=load_file(directory/'prototype_valid.npy')
    e=Evidence(directory,v);header=nearest_header(cp['parent_directory'])
    code=literal(directory/'snapshot_descent_coefficients_gpu.py','SOURCE')
    producer=mx.fast.metal_kernel(name='frozen_evidence_motion_coefficients',
        input_names=['states','wmasks','triples','constants_data','axes','amount'],output_names=['coefficients','bias','counts','chosen_axes','available'],header=header,source=code)
    kept=[];removed=0;last=time.monotonic()
    for first in range(0,source.shape[0],4096):
        states=source[first:first+4096];match=(states[:,0]*8+states[:,1])[:,None]==keys[None,:]
        assert mx.all(mx.any(match,axis=1)).item();idx=mx.argmax(match,axis=1)
        co,b,count,axes,available=producer(inputs=[states,v.masks,v.triples,e.co,mx.full((states.shape[0],3),-1,dtype=mx.int32),mx.array([states.shape[0]],dtype=mx.uint32)],
            grid=(states.shape[0],1,1),threadgroup=(64,1,1),output_shapes=[(states.shape[0],42,9,2),(states.shape[0],42),(states.shape[0],),(states.shape[0],3),(states.shape[0],)],
            output_dtypes=[mx.int64,mx.int64,mx.uint32,mx.int32,mx.uint32])
        accepted=mx.zeros((states.shape[0],),dtype=mx.bool_);independent=mx.zeros_like(accepted)
        enabled=mx.arange(42,dtype=mx.uint32)[None,:]<count[:,None]
        for j in range(4):
            velocity=velocities[idx,j]
            endpoint=mx.where(velocity[:,None,:]>=0,co[:,:,:,0],co[:,:,:,1])
            proved=good[idx,j]&(available>0)&mx.all((mx.sum(endpoint*velocity[:,None,:].astype(mx.int64),axis=2)>b*1024)|~enabled,axis=1)
            accepted|=proved
            # Candidate validity flags are producer metadata; the independent
            # verifier establishes the supplied velocity directly.
            independent|=v.motion_check(states,velocity)>0
        missing=take(states,accepted&~independent)
        if missing is not None:
            if not mx.all(v.contract(missing,16)[1]!=0).item():
                mx.save(str(DEST/'descent_unverified_states.npy'),missing)
                raise RuntimeError('Independent verifier left a supplied-motion removal unproved.')
        alive=take(states,~accepted);removed+=mx.sum(accepted.astype(mx.int32)).item()
        if alive is not None:kept.append(alive)
        if time.monotonic()-last>10:print(directory.name,'independent motion rows',first+states.shape[0],flush=True);last=time.monotonic()
    remaining=cat(kept);assert digest(remaining)==cert['result_sha256'] and removed==cert['regions_with_strict_descent']
    stored=cat([load_file(p) for p in cp['frontier_files']]);assert remaining.shape==stored.shape and mx.all(remaining==stored).item()
    return {'passed':True,'rows':source.shape[0],'supplied_motion_claims':removed,'remaining_regions':remaining.shape[0],
        'all_producer_removals_independently_proved':True,'elapsed_seconds':time.monotonic()-start}

def rotation_node(directory,v):
    cp=read(directory/'checkpoint.json');r=read(directory/'minimum_rotation_check.json');parent=read(r['parent_checkpoint'])
    states=cat([load_file(p) for p in parent['frontier_files']]);assert digest(states)==r['input_frontier_sha256']
    text=(directory/'primary_model.metal').read_text();position=text.index('uint row=thread_position_in_grid.x')
    e=Evidence(directory,v)
    producer=mx.fast.metal_kernel(name='frozen_evidence_rotation_node',header=text[:position],source=text[position:],
        input_names=['states','wmasks','triples','constants_data','amount'],output_names=['keep'])
    output=producer(inputs=[states,v.masks,v.triples,e.co,mx.array([states.shape[0]],dtype=mx.uint32)],
        grid=(states.shape[0],1,1),threadgroup=(128,1,1),output_shapes=[(states.shape[0],)],output_dtypes=[mx.uint32])[0]
    assert digest(output)==r['classification_sha256']
    removed=take(states,output==0)
    if removed is not None:require_motion(v,removed)
    remaining=take(states,output>0)
    if remaining is None:remaining=cat([])
    assert digest(remaining)==r['output_frontier_sha256']
    return {'passed':True,'rows':states.shape[0],'rigid_motion_claims':0 if removed is None else removed.shape[0]}

def main():
    p=argparse.ArgumentParser();p.add_argument('--node');args=p.parse_args();DEST.mkdir(exist_ok=True)
    chain=read(ROOT/'certified_spatial/replay_receipts/proof_chain_integrity.json')['nodes']
    tasks=[Path(args.node)] if args.node else [Path(x['directory']) for x in reversed(chain) if x['format'] in ['compact-deterministic-replay-v1','strict-descent-reduction-v1','minimum-rotation-reduction-v1']]
    verifier={24:MotionVerifier(24),30:MotionVerifier(30)};base_commit=source_commitment();motion_hash=motion_commitment();adapter=sha(__file__)
    done=[]
    for directory in tasks:
        dest=DEST/(directory.name+'_independent.json');cp=read(directory/'checkpoint.json')
        if dest.exists():
            r=read(dest)
            if r.get('passed') and r.get('verifier_sha256')==base_commit and r.get('motion_verifier_sha256')==motion_hash and r.get('adapter_sha256')==adapter and r.get('checkpoint_sha256')==sha(directory/'checkpoint.json'):
                done.append(r);print('Verified receipt reused',directory.name,flush=True);continue
        print('Verifying independent geometry generation',directory.name,flush=True)
        v=verifier[cp['fixed_point_bits']];fmt=cp['format']
        result=compact(directory,v) if fmt=='compact-deterministic-replay-v1' else descent(directory,v) if fmt=='strict-descent-reduction-v1' else rotation_node(directory,v)
        assert source_commitment()==base_commit and motion_commitment()==motion_hash
        result.update(directory=str(directory),checkpoint_sha256=sha(directory/'checkpoint.json'),verifier_sha256=base_commit,
            motion_verifier_sha256=motion_hash,adapter_sha256=adapter,cpu_numerical_fallback=False)
        dest.write_text(json.dumps(result,indent=2));done.append(result);print(json.dumps(result,indent=2),flush=True)
    if not args.node:(DEST/'later_generations_independent.json').write_text(json.dumps({'passed':True,'generations':done,'all_later_geometry_and_motion_claims_verified':True},indent=2))

if __name__=='__main__':main()
