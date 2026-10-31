"""Scientific execution is fixed to host149; no cloud launch is performed here."""
from pathlib import Path
import socket,datetime,uuid
from .common import atomic_json,fail

def enter_command(args,root):
    if args.command in {'package-check','status'}:
        return
    actual=socket.gethostname()
    if actual!='149':
        fail('WRONG_EXECUTION_HOST','Preparation, audits and training must execute on149',target_host='149',actual_host=actual)
    if args.command=='supervise':
        fail('REMOTE_TRAINING_DISABLED','SSH/off-host supervision is disabled; use the signed local run-queue on host 149')
    directory=Path(root)/'.command_receipts'
    atomic_json(directory/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex+'.json'),
        {'target_host':'149','actual_host':actual,'command':args.command,'operation':'COMMAND_PREFLIGHT','paid_compute_started':False,
         'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
