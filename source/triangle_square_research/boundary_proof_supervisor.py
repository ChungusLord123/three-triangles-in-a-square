"""Bounded background exact-boundary work; no unsupported global proof claims."""
import argparse,json,os,plistlib,subprocess,sys,time,uuid,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--seconds',type=float,default=600)
    a=p.parse_args();allocation=json.loads((ROOT/'wd_allocation.json').read_text())
    base=Path(allocation['data_root'])/'runs';operations=base.parent/'operations'
    status=ROOT/'wd_run_status.json';state=json.loads(status.read_text());job=uuid.uuid4().hex[:12]
    start=time.monotonic();parent=Path(a.parent).resolve();stage=0
    def publish(**fields):
        state.update(fields,updated_at=now(),elapsed_seconds=time.monotonic()-start)
        write(status,state)
    def guard():
        for mount,uid in [(allocation['drive_mount'],allocation['drive_uuid']),(allocation['mountpoint'],allocation['proof_volume_uuid'])]:
            r=subprocess.run(['diskutil','info','-plist',mount],capture_output=True,check=True)
            info=plistlib.loads(r.stdout)
            if info.get('VolumeUUID')!=uid or info.get('MountPoint')!=mount or not info.get('WritableVolume'):
                raise RuntimeError('Dedicated proof drive is unavailable or has a different identity.')
            if mount==allocation['mountpoint']:
                state.update(volume_used_bytes=info.get('CapacityInUse'),volume_free_bytes=info.get('APFSContainerFree'))
    def execute(args,phase):
        log=operations/f'boundary_{job}_{stage:03d}_{phase}.log'
        with log.open('w') as out:
            child=subprocess.Popen([sys.executable,*args],cwd=ROOT.parent,stdout=out,stderr=subprocess.STDOUT)
            publish(state=phase,worker_pid=child.pid,worker_log=str(log))
            while child.poll() is None:
                time.sleep(2)
                if (base.parent/'BOUNDARY_STOP_REQUESTED').exists():
                    child.terminate();child.wait();raise RuntimeError('Boundary run stopped by request; verified checkpoints preserved.')
            if child.returncode:raise RuntimeError(f'{phase} failed; inspect {log}')
            publish(worker_pid=None)
    initial=json.loads((parent/'checkpoint.json').read_text())
    publish(job_id=job,pid=os.getpid(),started_at=now(),state='boundary_starting',time_budget_seconds=a.seconds,
        last_verified_boundary_checkpoint=str(parent/'checkpoint.json'),unresolved_boxes=initial['unresolved_active_boxes'],
        precision_boxes=initial['unresolved_precision_boxes'],contact_systems=initial['frontier_profiles'],
        analytic_covered_profiles=initial['analytic_covered_profiles'],parent_checkpoint=str(parent/'checkpoint.json'),
        exact_optimality_proved=False,target_text='exact optimum boundary',completed_target=False,
        message='Verified rational exclusion through 1.4783977. Continuing the final boundary interval with separate algebraic coverage.')
    try:
        guard();operations.mkdir(exist_ok=True)
        while time.monotonic()-start<a.seconds:
            guard();cp=json.loads((parent/'checkpoint.json').read_text())
            assert cp['offline_replay_passed'] and cp['exact_boundary_goal']
            if cp['unresolved_active_boxes']==0 and cp['unresolved_precision_boxes']==0:
                publish(state='boundary_catalogue_cleared',worker_pid=None,pid=None,
                    message='All remaining boundary catalogue regions cleared. Independent geometric-model and global coverage review still required.');return
            stage+=1
            remaining=cp['unresolved_active_boxes']+cp['unresolved_precision_boxes']
            if remaining<1000000 and not parent.name.startswith('descent_'):
                execute(['triangle_square_research/boundary_descent_runner_gpu.py','--parent',str(parent),'--output-base',str(base)],'boundary_descent')
                parent=Path(json.loads((ROOT/'certified_spatial/latest_boundary_run.json').read_text())['directory'])
            else:
                execute(['triangle_square_research/exact_boundary_runner_gpu.py','--parent',str(parent),'--output-base',str(base),'--seconds','60'],'boundary_search')
                candidate=Path(json.loads((base/'boundary_pending.json').read_text())['directory'])
                execute(['triangle_square_research/exact_boundary_runner_gpu.py','--replay',str(candidate)],'boundary_replay')
                parent=candidate
            cp=json.loads((parent/'checkpoint.json').read_text());assert cp['offline_replay_passed']
            publish(last_verified_boundary_checkpoint=str(parent/'checkpoint.json'),unresolved_boxes=cp['unresolved_active_boxes'],
                precision_boxes=cp['unresolved_precision_boxes'],contact_systems=cp['frontier_profiles'],
                analytic_covered_profiles=cp['analytic_covered_profiles'],generation=stage,
                message='Exact-boundary checkpoint independently replayed. Exact global optimality remains unproved.')
            if cp['unresolved_active_boxes']==0 and cp['unresolved_precision_boxes']>0 and parent.name.startswith('descent_'):
                publish(state='boundary_precision_limit',pid=None,worker_pid=None,
                    message='Boundary regions retained at the arithmetic limit; stronger exact certificates are needed.');return
        publish(state='boundary_checkpointed',pid=None,worker_pid=None,message='Bounded boundary run completed and verified checkpoints saved. Exact optimality remains unproved.')
    except Exception as exc:
        publish(state='boundary_stopped',pid=None,worker_pid=None,message=str(exc))
        raise

if __name__=='__main__':main()
