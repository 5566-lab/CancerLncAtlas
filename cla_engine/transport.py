"""149-managed SSH transport. Existing SSH alias only; no credentials, rental or cloud shutdown."""
from __future__ import annotations
import json,os,re,shlex,subprocess,tarfile,time,uuid
from pathlib import Path
from .common import atomic_json,read_json,sha,fail,status,lock,verify_manifest
from .runtime_guard import make_resume_tar,check_remote_status,validate_checkpoint_pointer,OptimizerProgressWatch
from .lineage import validate_prepared,verify_completion,verify_signed_file

TERMINAL={'GO_TO_FORMAL_EVALUATION','NO_GO_WITHIN_DIAGNOSTIC_BUDGET','PAUSED_BUDGET_OR_SIGNAL','ERROR','TRAINED_FORMAL_CANDIDATE'}

def execute(args,timeout=120,check=True):
    r=subprocess.run(args,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
    if check and r.returncode:fail('TRANSPORT_ERROR','Command failed',returncode=r.returncode,command=args[:3],stderr=r.stderr[-2500:])
    return r

class SSH:
    def __init__(self,alias):
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+',alias) or alias.startswith('-'):fail('SSH_ALIAS_INVALID','Use a configured SSH alias, not an arbitrary shell string')
        self.alias=alias;self.deadline=None;self.cleanup_deadline=None
    def limit(self,maximum,*,cleanup=False):
        deadline=self.cleanup_deadline if cleanup else self.deadline
        if cleanup and deadline is None:fail('CLEANUP_NOT_AUTHORIZED','A bounded recovery period is required')
        if deadline is None:return maximum
        remaining=deadline-time.time()
        if remaining<=0:fail('RECOVERY_BUDGET_EXHAUSTED' if cleanup else 'BUDGET_EXHAUSTED','Transport cannot extend the authorized deadline')
        return max(.1,min(maximum,remaining))
    def run(self,command,timeout=120,check=True,*,cleanup=False):
        if cleanup and not cleanup_command_allowed(command):fail('UNSAFE_CLEANUP_COMMAND','Recovery cannot start a worker, update leases or modify training inputs')
        return execute(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2',self.alias,command],self.limit(timeout,cleanup=cleanup),check)
    def push(self,local,remote):
        return execute(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',str(local),self.alias+':'+remote],timeout=self.limit(7200))
    def pull(self,remote,local,*,cleanup=False):
        Path(local).parent.mkdir(parents=True,exist_ok=True)
        return execute(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=12',self.alias+':'+remote,str(local)],timeout=self.limit(7200,cleanup=cleanup))
    def json(self,path,*,cleanup=False):
        r=self.run('cat '+shlex.quote(path),check=False,cleanup=cleanup)
        return json.loads(r.stdout) if r.returncode==0 else None

def cleanup_command_allowed(command):
    """Closed allowlist; no shell composition or training commands after deadline."""
    if any(c in command for c in (';', '\n', '&&', '||', '`', '$', '|', '>', '<')):return False
    try:tokens=shlex.split(command)
    except ValueError:return False
    if not tokens:return False
    if tokens[0] in {'cat','sha256sum'}:return len(tokens)==2 and tokens[1].startswith('/')
    if tokens[0]=='ps':return len(tokens)>=4 and tokens[1]=='-p' and tokens[2].isdigit()
    if tokens[0]=='kill':return len(tokens)==3 and tokens[1] in {'-TERM','-KILL'} and tokens[2].isdigit()
    if tokens[0]=='tar':
        return len(tokens)==7 and tokens[1]=='-cf' and tokens[3]=='-C' and tokens[2].startswith('/') and tokens[4].startswith('/') and tokens[5:]==['--','.']
    return False


def recover_after_deadline(remote,rr,out,mirror,prepared,attempt,pid,deadline,*,grace=300):
    """Bounded stop/read/pull only. Cannot renew a lease or execute optimizer work."""
    out=Path(out);mirror=Path(mirror);remote.cleanup_deadline=deadline+float(grace)
    if time.time()>=remote.cleanup_deadline:
        return status(out,'RECOVERY_WINDOW_EXPIRED',checkpoint_available=(mirror/'LATEST.json').exists(),automatic_release=False,
            next_action='RUN_EXPLICIT_RECOVER_ONLY_OR_RELEASE_AFTER_VERIFYING_PERSISTED_FILES')
    try:
        # Validate process identity before a PID-specific signal; never broad pkill.
        proc=remote.run('ps -p '+str(pid)+' -o stat= -o args=',check=False,cleanup=True)
        alive=proc.returncode==0 and rr+'/code/cla.py' in proc.stdout and 'worker' in proc.stdout and not proc.stdout.lstrip().startswith('Z')
        if alive:remote.run('kill -TERM '+str(pid),check=False,cleanup=True)
        latest_sha=None
        while time.time()<remote.cleanup_deadline:
            st=remote.json(rr+'/run/STATUS.json',cleanup=True)
            latest=remote.json(rr+'/run/LATEST.json',cleanup=True)
            if latest:validate_checkpoint_pointer(latest)
            if latest and latest.get('sha256')!=latest_sha:
                rel=Path(latest['file'])
                if rel.is_absolute() or '..' in rel.parts or len(rel.parts)!=2 or rel.parts[0]!='checkpoints' or rel.suffix!='.pt':
                    fail('CHECKPOINT_POINTER_UNSAFE',str(rel))
                dest=mirror/rel;dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial')
                remote.pull(rr+'/run/'+rel.as_posix(),tmp,cleanup=True)
                if sha(tmp)!=latest['sha256']:fail('CHECKPOINT_TRANSFER_HASH_MISMATCH','Recovery copy differs')
                os.replace(tmp,dest);atomic_json(dest.with_name(dest.name+'.ready.json'),latest)
                atomic_json(mirror/'LATEST.json',latest);atomic_json(out/'PERSISTED_CHECKPOINT.json',latest);latest_sha=latest['sha256']
            proc=remote.run('ps -p '+str(pid)+' -o stat= -o args=',check=False,cleanup=True)
            alive=proc.returncode==0 and rr+'/code/cla.py' in proc.stdout and 'worker' in proc.stdout and not proc.stdout.lstrip().startswith('Z')
            terminal=st and st.get('attempt_id')==attempt and st.get('state') in TERMINAL
            if terminal:
                # Archive may write a container on the instance, but cannot invoke training.
                command='tar -cf '+shlex.quote(rr+'/result_recovery.tar')+' -C '+shlex.quote(rr+'/run')+' -- .'
                remote.run(command,cleanup=True)
                result=out/'result_recovery.tar';remote.pull(rr+'/result_recovery.tar',result,cleanup=True)
                observed=remote.run('sha256sum '+shlex.quote(rr+'/result_recovery.tar'),cleanup=True).stdout.split()[0]
                if observed!=sha(result):fail('RESULT_TRANSFER_HASH_MISMATCH','Recovery archive differs')
                extract=out/('recovered_'+uuid.uuid4().hex);extract.mkdir()
                with tarfile.open(result) as archive:
                    for member in archive.getmembers():
                        path=(extract/member.name).resolve()
                        if not path.is_relative_to(extract.resolve()) or not (member.isfile() or member.isdir()):fail('UNSAFE_ARCHIVE','Recovery archive')
                    archive.extractall(extract)
                received=read_json(extract/'STATUS.json')
                if received.get('attempt_id')!=attempt or received.get('state')!=st['state']:fail('RESULT_ATTEMPT_MISMATCH','Recovery is for another attempt')
                if (extract/'TRAINING_RECEIPT.json').exists():verify_completion(prepared,extract)
                contract=extract/'RUN_CONTRACT.json'
                if contract.exists():
                    binding=read_json(contract);binding['controller_prepared_path']=str(prepared);atomic_json(contract,binding)
                old=out/('mirror_before_recovery_'+uuid.uuid4().hex)
                if mirror.exists():os.replace(mirror,old)
                os.replace(extract,mirror)
                return status(out,'INSTANCE_RELEASE_REQUIRED',worker_terminal_state=st['state'],results_on_149=str(mirror),
                    optimizer_budget_extended=False,automatic_release=False,recovery_completed=True)
            if not alive:
                return status(out,'WORKER_STOPPED_RECOVERY_PARTIAL',checkpoint_available=(mirror/'LATEST.json').exists(),
                    optimizer_budget_extended=False,automatic_release=False,requires_instance_release=True)
            time.sleep(min(2,max(.1,remote.cleanup_deadline-time.time())))
    except Exception as exc:
        return status(out,'RECOVERY_INCOMPLETE',error=str(exc),checkpoint_available=(mirror/'LATEST.json').exists(),
            optimizer_budget_extended=False,automatic_release=False,next_action='RECOVER_ONLY_NO_NEW_TRAINING')
    return status(out,'RECOVERY_WINDOW_EXPIRED',checkpoint_available=(mirror/'LATEST.json').exists(),automatic_release=False)


def recover_only(out,alias,seconds=300):
    """Explicit bounded recovery of an existing job; no instance creation or worker launch."""
    out=Path(out);b=read_json(out/'BUDGET.json');r=read_json(out/'REMOTE.json')
    if not 0<float(seconds)<=600:fail('RECOVERY_BUDGET_INVALID','0 < seconds <= 600')
    if time.time()<b['deadline']:fail('RECOVERY_BEFORE_DEADLINE','Use normal supervision while the original training budget is active')
    if r['alias']!=alias:fail('RECOVERY_HOST_MISMATCH','Recover from the recorded instance, not a replacement with no files')
    binding=read_json(out/'run'/'RUN_CONTRACT.json') if (out/'run'/'RUN_CONTRACT.json').exists() else {}
    prepared=b.get('prepared_path') or binding.get('controller_prepared_path')
    if not prepared:fail('RECOVERY_PREPARED_PATH_MISSING','Budget must bind a controller input directory')
    remote=SSH(alias);remote.deadline=b['deadline']
    with lock(out/'supervisor'):
        atomic_json(out/'RECOVERY_AUTHORIZATION.json',{'seconds':seconds,'authorized_unix':time.time(),'optimizer_budget_extended':False})
        return recover_after_deadline(remote,r['root'],out,out/'run',prepared,r['attempt_id'],r['pid'],time.time(),grace=seconds)


def make_tar(source,target):
    source=Path(source).resolve();target=Path(target).resolve()
    if target.is_relative_to(source):fail('ARCHIVE_RECURSION','Output archive must be outside input directory')
    with tarfile.open(target,'w') as tar:
        for p in sorted(Path(source).rglob('*')):
            if p.is_file() and not any(x in p.parts for x in ['__pycache__','.git','.pytest_cache']) and not p.name.endswith(('.pyc','.partial')) and '.tmp.' not in p.name:
                tar.add(p,arcname=str(p.relative_to(source)),recursive=False)

CODE_ARCHIVE_DIRS=('cla_engine','cc_hhgt','scripts','config')
CODE_ARCHIVE_SUFFIXES={'.py','.r','.sh','.yaml','.yml','.json'}
SOURCE_MANIFEST=Path('provenance/joint_mainline_20261101/SOURCE_MANIFEST.json')

def make_code_tar(package,target):
    """Archive the unified joint runtime with the same scope as ``code_digest``.

    The source manifest is carried beside the runtime when available so a
    transferred capsule remains bound to the reviewed mainline generation.
    Patient exports, caches and historical implementation snapshots are never
    inferred into the capsule.
    """
    package=Path(package).resolve();target=Path(target).resolve()
    if target.is_relative_to(package):fail('ARCHIVE_RECURSION','Output archive must be outside the source tree')
    root_entry=package/'cla.py'
    if not root_entry.is_file():fail('CODE_ARCHIVE_INCOMPLETE','Unified entry point is missing',path='cla.py')
    entries=[root_entry]
    for folder in CODE_ARCHIVE_DIRS:
        base=package/folder
        if not base.exists():continue
        for p in sorted(base.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix.lower() not in CODE_ARCHIVE_SUFFIXES:continue
            if p.is_symlink() or not p.resolve().is_relative_to(package):fail('CODE_ARCHIVE_SYMLINK','Source capsule cannot follow links',path=p.relative_to(package).as_posix())
            entries.append(p)
    source_manifest=package/SOURCE_MANIFEST
    if source_manifest.is_file():
        # Carry the complete reviewed set so package-check also works after
        # extraction. A manifest without its bound files is not a valid capsule.
        from .package import verify as verify_source, source_files
        verify_source(package)
        present=set(entries)
        for p in source_files(package).values():
            if p not in present:entries.append(p);present.add(p)
        entries.append(source_manifest)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(target,'w') as tar:
        for p in entries:tar.add(p,arcname=p.relative_to(package).as_posix(),recursive=False)

def push_lease(remote,local,path):
    remote.push(local,path+'.partial')
    remote.run('mv -f '+shlex.quote(path+'.partial')+' '+shlex.quote(path))

def supervise(package,prepared,out,alias,hours,remote_python=None,remote_root='/root/cla_full_pyg_r6_2',resume=False,authorization=None):
    """Bound total wallclock INCLUDING transfer/retries. Restart cannot refresh deadline.
    Resuming on a new instance reuses 149's verified checkpoint. Does not allocate
    replacement instances or claim that killing Python stops cloud billing.
    """
    package=Path(package).resolve();prepared=Path(prepared).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if not (0<hours<=96):fail('BUDGET_INVALID','Specify >0 and <=96 instance hours explicitly')
    manifest,spec,cfg,lineage=validate_prepared(prepared)
    from .controller import assert_controller
    assert_controller(cfg)
    if manifest['metadata']['profile']=='formal':
        if not authorization:fail('FORMAL_ADMISSION_REQUIRED','Formal supervision must use the signed queue, not a direct manual worker')
        plan_path=Path(authorization['plan_path']);verify_signed_file(plan_path,authorization['base_config'],'FORMAL_JOB_PLAN')
        plan=read_json(plan_path)
        if not any(Path(j['prepared']).resolve()==prepared and j['sha256']==sha(prepared/'MANIFEST.json') for j in plan['jobs']):fail('FORMAL_JOB_NOT_AUTHORIZED','Prepared job not in signed plan')
        from .queue import check_admission
        check_admission(authorization['base_config'],plan['diagnostic_run'])
    if (out/'run'/'TRAINING_RECEIPT.json').exists():
        receipt,_=verify_completion(prepared,out/'run')
        return status(out,'INSTANCE_RELEASE_REQUIRED',worker_terminal_state=receipt['state'],completed_run_reused=True,automatic_release=False)
    remote=SSH(alias);budgetfile=out/'BUDGET.json';job=str(uuid.uuid4())
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote_root):fail('REMOTE_ROOT_INVALID','Simple absolute remote path required')
    with lock(out/'supervisor'):
        if budgetfile.exists():
            b=read_json(budgetfile)
            if b['prepared_sha256']!=sha(prepared/'MANIFEST.json'):fail('BUDGET_INPUT_DRIFT','New input requires a new explicitly budgeted job')
            if not resume:fail('JOB_EXISTS','Use resume; budget will not reset')
            if b['deadline']<=time.time():
                prior=read_json(out/'REMOTE.json') if (out/'REMOTE.json').exists() else None
                if not prior:return status(out,'BUDGET_EXHAUSTED',automatic_release=False,next_action='NO_TRAINING_STARTED_VERIFY_INSTANCE_RELEASE')
                if prior['alias']!=alias:fail('RECOVERY_HOST_MISMATCH','Only recover from the recorded instance')
                remote.deadline=b['deadline']
                return recover_after_deadline(remote,prior['root'],out,out/'run',prepared,prior['attempt_id'],prior['pid'],b['deadline'],grace=cfg.get('supervision',{}).get('cleanup_seconds',300))
            job=b['job_id']
        else:
            b={'job_id':job,'start':time.time(),'deadline':time.time()+hours*3600,'authorized_hours':hours,'prepared_sha256':sha(prepared/'MANIFEST.json'),'prepared_path':str(prepared),'billing_stop_implemented':False,'cloud_resource_created':False};atomic_json(budgetfile,b)
        remote.deadline=b['deadline']
        rr=remote_root+'/'+job;previous=read_json(out/'REMOTE.json') if (out/'REMOTE.json').exists() else None
        if previous and previous['alias']!=alias and time.time()<b.get('lease_expires_unix',0)+10:
            fail('PREVIOUS_LEASE_STILL_VALID','Do not start a second GPU worker until prior controller lease expires',retry_after_unix=b['lease_expires_unix']+10)
        remote.run('mkdir -p '+shlex.quote(rr))
        # Existing GPU environment is required, including actual PyG and current encoder dependencies.
        if remote_python is None:
            r=remote.run("for p in /usr/local/miniconda3/envs/py312/bin/python /opt/conda/bin/python /usr/bin/python3; do if [ -x \"$p\" ] && \"$p\" -c 'import torch,numpy,scipy,sklearn,torch_geometric,pandas; assert torch.cuda.is_available()' >/dev/null 2>&1; then echo \"$p\"; break; fi; done")
            remote_python=r.stdout.strip().splitlines()[0] if r.stdout.strip() else ''
        if not remote_python:fail('GPU_ENVIRONMENT_MISSING','No suitable existing GPU Python; environment is not silently modified')
        mirror=out/'run';mirror.mkdir(exist_ok=True)
        live=False
        if previous and previous['alias']==alias and str(previous.get('pid','')).isdigit():
            pid=str(previous['pid'])
            cmd='ps -p '+pid+' -o args='
            r=remote.run(cmd,check=False)
            live=r.returncode==0 and rr+'/code/cla.py' in r.stdout and 'worker' in r.stdout
        if live:
            attempt=previous.get('attempt_id')
            if not attempt:fail('OLD_WORKER_ATTEMPT_UNBOUND','Cannot reattach a worker lacking current attempt identity')
            started_unix=previous.get('worker_launch_unix')
            if not isinstance(started_unix,(int,float)):fail('WORKER_LAUNCH_TIME_UNBOUND','Cannot reattach an old attempt without a true launch timestamp')
        if not live:
            attempt=uuid.uuid4().hex;transfer_started_unix=time.time()
            status(out,'TRANSFERRING',gpu_alias=alias,deadline=b['deadline'],billing_stop_implemented=False)
            code_tar=out/'code.tar';data_tar=out/'training_inputs.tar';make_code_tar(package,code_tar);make_tar(prepared,data_tar)
            for local,name in [(code_tar,'code.tar'),(data_tar,'inputs.tar')]:
                remote.push(local,rr+'/'+name);r=remote.run('sha256sum '+shlex.quote(rr+'/'+name))
                if r.stdout.split()[0]!=sha(local):fail('TRANSFER_HASH_MISMATCH',name)
            remote.run('mkdir -p '+shlex.quote(rr+'/code')+' '+shlex.quote(rr+'/inputs')+' '+shlex.quote(rr+'/run')+' && tar -xf '+shlex.quote(rr+'/code.tar')+' -C '+shlex.quote(rr+'/code')+' && tar -xf '+shlex.quote(rr+'/inputs.tar')+' -C '+shlex.quote(rr+'/inputs'))
            mirror=out/'run';mirror.mkdir(exist_ok=True)
            # Fresh directory for every worker attempt. Do not union old outputs into the new run.
            remote.run('if [ -d '+shlex.quote(rr+'/run')+' ]; then mv '+shlex.quote(rr+'/run')+' '+shlex.quote(rr+'/previous_run_'+attempt)+'; fi; mkdir -p '+shlex.quote(rr+'/run'))
            if (mirror/'LATEST.json').exists():
                restore=out/'resume.tar';make_resume_tar(mirror,restore);remote.push(restore,rr+'/resume.tar')
                if remote.run('sha256sum '+shlex.quote(rr+'/resume.tar')).stdout.split()[0]!=sha(restore):fail('RESTORE_TRANSFER_HASH_MISMATCH','Restore archive differs from persisted copy')
                remote.run('tar -xf '+shlex.quote(rr+'/resume.tar')+' -C '+shlex.quote(rr+'/run'))
            # Check the current PyG backend, context path and shared gradients
            # before any model optimizer starts.
            if time.time() >= b['deadline']:fail('BUDGET_EXHAUSTED','Transfers consumed the authorized budget; do not start GPU training')
            probe=' '.join(map(shlex.quote,[remote_python,rr+'/code/cla.py','backend-check','--output',rr+'/backend_check','--device','cuda']))
            checked=remote.run(probe,timeout=max(1,min(300,b['deadline']-time.time())),check=False)
            (out/'BACKEND_CHECK_REMOTE.log').write_text(checked.stdout+'\n'+checked.stderr,encoding='utf-8')
            if checked.returncode:fail('GPU_BACKEND_CHECK_FAILED','Current PyG context/gradient check failed; no training started',exit_code=checked.returncode)
            remote.pull(rr+'/backend_check/BACKEND_TEST.json',out/'BACKEND_TEST_GPU.json')
            if time.time() >= b['deadline']:fail('BUDGET_EXHAUSTED','Backend test consumed authorized budget; do not start training')
            # Start only after transfer SHA and actual backend checks.

            b['lease_expires_unix']=min(b['deadline'],time.time()+120);atomic_json(budgetfile,b);lease=out/'LEASE.json';atomic_json(lease,{'job_id':job,'attempt_id':attempt,'expires_unix':b['lease_expires_unix']});push_lease(remote,lease,rr+'/run/SUPERVISOR_LEASE.json')
            command='env '+shlex.quote('CLA_REMOTE_ATTEMPT_ID='+attempt)+' '+shlex.quote('CLA_ATTEMPT_ID='+attempt)+' '+' '.join(map(shlex.quote,[remote_python,rr+'/code/cla.py','worker','--prepared',rr+'/inputs','--output',rr+'/run','--device','cuda','--deadline',str(b['deadline'])]))
            shell='nohup '+command+' > '+shlex.quote(rr+'/worker.log')+' 2>&1 < /dev/null & echo $!'
            started_unix=time.time()  # after upload, verification, backend check and lease; before actual worker spawn
            started=remote.run(shell);atomic_json(out/'REMOTE.json',{'alias':alias,'root':rr,'python':remote_python,'pid':started.stdout.strip(),'job_id':job,'attempt_id':attempt,'started_unix':started_unix,'worker_launch_unix':started_unix,'transfer_started_unix':transfer_started_unix,'worker_launch_ack_unix':time.time()})
        else:
            status(out,'REATTACHED_EXISTING_WORKER',job_id=job,gpu_alias=alias)
        current_remote=read_json(out/'REMOTE.json');worker_pid=current_remote['pid']
        if not str(worker_pid).isdigit():fail('WORKER_PID_INVALID','SSH launch did not return a PID')
        monitor_state=read_json(out/'OPTIMIZER_MONITOR.json') if (out/'OPTIMIZER_MONITOR.json').exists() else None
        if monitor_state and monitor_state.get('attempt_id')!=attempt:monitor_state=None
        progress_watch=OptimizerProgressWatch(attempt,started_unix,
            first_step_timeout=cfg.get('supervision',{}).get('first_optimizer_step_timeout_seconds',1800),
            progress_timeout=cfg.get('supervision',{}).get('progress_timeout_seconds',1800),state=monitor_state)
        failures=0;mirrored=None
        while True:
            try:
                if time.time()>=b['deadline']:
                    return recover_after_deadline(remote,rr,out,mirror,prepared,attempt,worker_pid,b['deadline'],grace=cfg.get('supervision',{}).get('cleanup_seconds',300))
                b['lease_expires_unix']=min(b['deadline'],time.time()+120);atomic_json(budgetfile,b);lease=out/'LEASE.json';atomic_json(lease,{'job_id':job,'attempt_id':attempt,'expires_unix':b['lease_expires_unix']});push_lease(remote,lease,rr+'/run/SUPERVISOR_LEASE.json')
                st=remote.json(rr+'/run/STATUS.json');latest=remote.json(rr+'/run/LATEST.json')
                if latest:validate_checkpoint_pointer(latest)
                if latest and (mirrored is None or latest['sha256']!=mirrored):
                    destination=mirror/latest['file'];destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix('.partial');remote.pull(rr+'/run/'+latest['file'],tmp)
                    if sha(tmp)!=latest['sha256']:fail('CHECKPOINT_TRANSFER_HASH_MISMATCH','Do not acknowledge or release instance')
                    os.replace(tmp,destination);atomic_json(destination.with_name(destination.name+'.ready.json'),latest);atomic_json(mirror/'LATEST.json',latest);atomic_json(out/'PERSISTED_CHECKPOINT.json',latest);mirrored=latest['sha256']
                    ack=out/'ACK.json';atomic_json(ack,latest);remote.push(ack,rr+'/run/PERSISTED_ACK.json')
                    # Delete only generations older than an acknowledged checkpoint; never a concurrently created newer one.
                    cleanup="import pathlib,json;d=pathlib.Path("+repr(rr+'/run/checkpoints')+");rs=sorted([(json.loads(p.read_text()),p) for p in d.glob('*.ready.json') if json.loads(p.read_text())['next_step'] < "+str(int(latest['next_step']))+"],key=lambda x:(x[0]['next_step'],x[0]['created_unix']));[(d.joinpath(pathlib.Path(r['file']).name).unlink(missing_ok=True),p.unlink(missing_ok=True)) for r,p in rs[:-1]]"
                    remote.run(shlex.quote(remote_python)+' -c '+shlex.quote(cleanup))
                proc=remote.run('ps -p '+str(worker_pid)+' -o stat= -o args=',check=False)
                alive=proc.returncode==0 and rr+'/code/cla.py' in proc.stdout and 'worker' in proc.stdout and not proc.stdout.lstrip().startswith('Z')
                watch=check_remote_status(st,attempt,alive,time.time(),started_unix,
                    startup_timeout=cfg.get('supervision',{}).get('startup_timeout_seconds',120),
                    progress_timeout=cfg.get('supervision',{}).get('progress_timeout_seconds',1800))
                if watch in {'WORKER_EXITED_WITHOUT_TERMINAL','STALE_ATTEMPT_STATUS','STARTUP_TIMEOUT','WORKER_PROGRESS_STALLED'}:
                    if alive:remote.run('kill -TERM '+str(worker_pid),check=False)
                    return status(out,watch,checkpoint_available=(mirror/'LATEST.json').exists(),attempt_id=attempt,automatic_release=False,requires_instance_release=True)
                progress='TERMINAL' if watch=='TERMINAL' else progress_watch.observe(st,time.time())
                atomic_json(out/'OPTIMIZER_MONITOR.json',progress_watch.snapshot())
                if progress in {'WORKER_PROGRESS_INVALID','WORKER_OPTIMIZER_STEP_REGRESSED','WORKER_NO_OPTIMIZER_PROGRESS','WORKER_OPTIMIZER_PROGRESS_STALLED'}:
                    if alive:remote.run('kill -TERM '+str(worker_pid),check=False)
                    return status(out,progress,attempt_id=attempt,checkpoint_available=(mirror/'LATEST.json').exists(),automatic_release=False,requires_instance_release=True)
                failures=0
                status(out,'SUPERVISING',worker_state=st,optimizer_progress_state=progress,checkpoint_persisted_on_149=mirrored is not None,seconds_remaining=max(0,b['deadline']-time.time()),billing_stop_implemented=False)
                if watch=='TERMINAL':
                    remote.run('tar -cf '+shlex.quote(rr+'/result.tar')+' -C '+shlex.quote(rr+'/run')+' .');result=out/'result.tar';remote.pull(rr+'/result.tar',result)
                    if remote.run('sha256sum '+shlex.quote(rr+'/result.tar')).stdout.split()[0]!=sha(result):fail('RESULT_TRANSFER_HASH_MISMATCH','Do not release before results are verified')
                    # tar built by this job; still reject path escapes and symlinks.
                    extracted=out/('result_verified_'+attempt);extracted.mkdir(exist_ok=False)
                    with tarfile.open(result) as tar:
                        for member in tar.getmembers():
                            target=(extracted/member.name).resolve()
                            if not target.is_relative_to(extracted.resolve()) or not (member.isfile() or member.isdir()):fail('UNSAFE_ARCHIVE','Result contains unsafe member')
                        tar.extractall(extracted)
                    received=read_json(extracted/'STATUS.json')
                    if received.get('attempt_id')!=attempt or received.get('state')!=st['state']:fail('RESULT_ATTEMPT_MISMATCH','Remote result belongs to another worker')
                    if (extracted/'TRAINING_RECEIPT.json').exists():
                        verify_completion(prepared,extracted)
                    contract=extracted/'RUN_CONTRACT.json'
                    if contract.exists():
                        binding=read_json(contract);binding['controller_prepared_path']=str(prepared);atomic_json(contract,binding)
                    # Old mirror is retained separately; extraction never merges stale predictions.
                    old=out/('previous_mirror_'+attempt)
                    if mirror.exists():os.replace(mirror,old)
                    os.replace(extracted,mirror)
                    return status(out,'INSTANCE_RELEASE_REQUIRED',worker_terminal_state=st['state'],results_on_149=str(mirror),automatic_release=False,message='Training stopped and results copied. Explicitly release the cloud instance; stopping Python is not stopping billing.')
            except (subprocess.TimeoutExpired,ConnectionError,json.JSONDecodeError) as ex:
                failures+=1
                if failures>=3:return status(out,'PREEMPTED_OR_UNREACHABLE',checkpoint_available=(mirror/'LATEST.json').exists(),next_action='RESUME_SAME_JOB_ON_EXISTING_OR_REPLACEMENT_SSH_ALIAS_WITH_REMAINING_BUDGET',detail=str(ex))
            except Exception as ex:
                from .common import Stop
                if isinstance(ex,Stop) and ex.code=='BUDGET_EXHAUSTED':
                    return recover_after_deadline(remote,rr,out,mirror,prepared,attempt,worker_pid,b['deadline'],grace=cfg.get('supervision',{}).get('cleanup_seconds',300))
                if isinstance(ex,Stop) and ex.code=='TRANSPORT_ERROR':
                    failures+=1
                    if failures>=3:return status(out,'PREEMPTED_OR_UNREACHABLE',next_action='RESUME_WITH_REMAINING_BUDGET',checkpoint_available=(mirror/'LATEST.json').exists())
                else:raise
            time.sleep(15)
