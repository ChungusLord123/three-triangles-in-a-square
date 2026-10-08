"""One bounded background GPU proof run, writing to the WD-backed 1 TB volume.

Host work is process scheduling, file I/O, disk metadata and status display.
Each numerical chunk gets independent GPU checks and a full offline replay.
Nothing in sources/ is touched. Earlier proof generations stay immutable.
"""
import os,json,time,subprocess,sys,uuid,fcntl,plistlib,shutil,argparse
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
ALLOCATION=json.loads((ROOT/'wd_allocation.json').read_text())
DATA=Path(ALLOCATION['data_root']);RUNS=DATA/'runs';OPS=DATA/'operations'
STATUS=ROOT/'wd_run_status.json'
STOP=DATA/'STOP_REQUESTED'

def now():return datetime.now(timezone.utc).isoformat()
def write_json(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def guard():
    if not Path(ALLOCATION['drive_mount']).is_mount() or not Path(ALLOCATION['mountpoint']).is_mount():
        raise RuntimeError('WD-backed proof volume is not mounted')
    current=plistlib.loads(subprocess.check_output(['diskutil','info','-plist',ALLOCATION['mountpoint']]))
    if current.get('VolumeUUID')!=ALLOCATION['proof_volume_uuid']:
        raise RuntimeError('The proof volume identity changed')
    if json.loads((DATA/'allocation.json').read_text())['drive_uuid']!=ALLOCATION['drive_uuid']:
        raise RuntimeError('WD allocation identity changed')
    usage=shutil.disk_usage(ALLOCATION['mountpoint'])
    if usage.total>ALLOCATION['budget_bytes']:
        raise RuntimeError('The proof volume exceeds the authorized 1 TB capacity')
    return usage

def main(retry_checkpoint=None):
    lock=(OPS/'runner.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads(STATUS.read_text()) if retry_checkpoint else None
    elapsed_before=max(0,time.time()-datetime.fromisoformat(previous['started_at']).timestamp()) if previous else 0
    started=time.monotonic()-elapsed_before;job_id=uuid.uuid4().hex[:12]
    parent=Path(json.loads((ROOT/'certified_spatial/latest_run.json').read_text())['directory']).resolve()
    state={'job_id':job_id,'pid':os.getpid(),'state':'starting','started_at':previous['started_at'] if previous else now(),
      'budget_bytes':ALLOCATION['budget_bytes'],'volume_size_bytes':ALLOCATION['volume_size_bytes'],
      'data_root':str(DATA),'chunk_seconds':120,'time_budget_seconds':3600,
      'gpu_only_geometry':True,'exact_optimality_proved':False,'generation':0,
      'target_text':previous['target_text'] if previous else '1.478397','parent_checkpoint':str(parent/'checkpoint.json')}
    if previous:
        write_json(OPS/f'{previous["job_id"]}_failed_status.json',previous)
        state['resumes_job_id']=previous['job_id']
    def publish(**fields):
        state.update(fields);state['updated_at']=now();state['elapsed_seconds']=time.monotonic()-started
        try:
            u=shutil.disk_usage(ALLOCATION['mountpoint'])
            state['volume_used_bytes']=u.used;state['volume_free_bytes']=u.free
        except OSError:pass
        write_json(STATUS,state)
        try:write_json(DATA/'job_status.json',state)
        except OSError:pass
    def execute(args,log_path,phase):
        env=dict(os.environ);env['PYTORCH_ENABLE_MPS_FALLBACK']='0';env['PYTHONUNBUFFERED']='1'
        with log_path.open('w') as log:
            child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=env)
            publish(state=phase,worker_pid=child.pid,worker_log=str(log_path))
            while child.poll() is None:
                time.sleep(2)
                if time.monotonic()-float(state.get('last_publish_monotonic',0))>10:
                    publish(last_publish_monotonic=time.monotonic())
            if child.returncode:raise RuntimeError(f'{phase} exited with code {child.returncode}; see {log_path}')
        publish(worker_pid=None)
    try:
        guard();publish(state='running')
        executor=ROOT/'wd_proof_executor.py'
        if retry_checkpoint:
            candidate=Path(retry_checkpoint).resolve()
            publish(message='Retrying GPU replay with 4096-row dispatches.',pending_checkpoint=str(candidate/'checkpoint.json'))
            execute([sys.executable,str(executor),'--replay',str(candidate)],OPS/f'{job_id}_replay_retry.log','replaying')
            cp=json.loads((candidate/'checkpoint.json').read_text());assert cp['offline_replay_passed']
            parent=candidate
            publish(last_verified_checkpoint=str(parent/'checkpoint.json'),unresolved_boxes=cp['unresolved_active_boxes'],
              precision_boxes=cp['unresolved_precision_boxes'],contact_systems=cp['frontier_profiles'],completed_target=cp['complete_below_target_exclusion'],message='GPU replay retry passed.')
            with (OPS/f'{job_id}_report.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'make_contact_report.py')],cwd=ROOT,stdout=log,stderr=log,check=True)
            if cp['complete_below_target_exclusion']:
                publish(state='rational_targets_completed',message='Rational targets cleared; exact optimality still needs an exact boundary certificate.');return
        # First accepted checkpoint is produced quickly; later chunks are 120s.
        cycle=0;shell=None
        while time.monotonic()-started<3600 and not STOP.exists():
            usage=guard()
            if usage.free<8*1024**3:
                publish(state='storage_budget',message='Stopped with room reserved for a complete checkpoint.');return
            old=json.loads((parent/'checkpoint.json').read_text())
            if old['unresolved_active_boxes']==0 and old['unresolved_precision_boxes']>0:
                publish(state='precision_limit',message='Retained regions need finer arithmetic or stronger bounds.');return
            cycle+=1;seconds=15 if cycle==1 else min(120,3600-(time.monotonic()-started))
            command=[sys.executable,str(executor),
              '--parent',str(parent),'--output-base',str(RUNS),'--seconds',str(seconds),
              '--max-frontier','12000000','--min-width','1']
            if shell:command+=['--shell-numerator',str(shell[0]),'--shell-denominator',str(shell[1])]
            publish(generation=cycle,target_text='1.4783977' if shell else state['target_text'])
            execute(command,OPS/f'{job_id}_{cycle:04d}_search.log','running')
            candidate=Path(json.loads((RUNS/'compact_pending.json').read_text())['directory'])
            publish(pending_checkpoint=str(candidate/'checkpoint.json'))
            execute([sys.executable,str(executor),'--replay',str(candidate)],
              OPS/f'{job_id}_{cycle:04d}_replay.log','replaying')
            cp=json.loads((candidate/'checkpoint.json').read_text())
            assert cp['offline_replay_passed'] and cp['cpu_numerical_fallback'] is False
            parent=candidate;shell=None
            publish(state='running',last_verified_checkpoint=str(parent/'checkpoint.json'),
              unresolved_boxes=cp['unresolved_active_boxes'],precision_boxes=cp['unresolved_precision_boxes'],
              contact_systems=cp['frontier_profiles'],boxes_evaluated=cp['boxes_evaluated'],
              completed_target=cp['complete_below_target_exclusion'])
            with (OPS/f'{job_id}_report.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'make_contact_report.py')],cwd=ROOT,stdout=log,stderr=log,check=True)
            if cp['complete_below_target_exclusion']:
                if cp['target_numerator']==1478397 and cp['target_denominator']==1000000:
                    shell=(14783977,10000000);publish(message='Current rational target cleared; starting the narrower side interval.');continue
                publish(state='rational_targets_completed',message='Rational targets cleared; exact optimality still needs an exact boundary certificate.');return
        publish(state='stopped' if STOP.exists() else 'time_budget',
          message='Checkpoint preserved. Initial one-hour run budget finished.' if not STOP.exists() else 'Stopped at a verified checkpoint.')
    except Exception as e:
        publish(state='failed',message=str(e));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--retry-checkpoint')
    args=parser.parse_args();main(args.retry_checkpoint)
