"""Memory-bounded private evidence fitting with separate train and validation factories."""
from __future__ import annotations
import math,random
from collections.abc import Callable,Iterable
import numpy as np
from .evidence_training import (PrivateHeadFit,build_fresh_private_eventset_head,model_parameter_sha256,
                                _require_torch,_batch_tensors,_loss_terms)


def evaluate_stream(model,factory,*,batch_size,device,allowed_targets=None):
    torch=_require_torch();model.eval();q_sum=q_n=d_sum=d_n=0
    with torch.no_grad():
        for packet in factory():
            examples=packet.examples if hasattr(packet,"examples") else packet
            for start in range(0,len(examples),batch_size):
                batch=_batch_tensors(examples,list(range(start,min(start+batch_size,len(examples)))),device)
                result=model(batch["events"],batch["mask"],batch["core"])
                qm=torch.isfinite(batch["confidence"]);dm=batch["direction"].ge(0)
                if qm.any():
                    q_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(result["confidence_logit"][qm],batch["confidence"][qm],reduction="sum").cpu());q_n+=int(qm.sum())
                if dm.any():
                    d_sum+=float(torch.nn.functional.cross_entropy(result["direction_logits"][dm],batch["direction"][dm],reduction="sum").cpu());d_n+=int(dm.sum())
    if q_n+d_n==0:raise RuntimeError("Evidence independent validation has no usable supervision")
    return (q_sum/q_n if q_n else 0.)+.5*(d_sum/d_n if d_n else 0.),{"confidence_n":q_n,"direction_n":d_n}


def _packet_digest(examples):
    import hashlib
    h=hashlib.sha256()
    for e in examples:
        h.update(str(e.key).encode());h.update(np.asarray(e.event_features).tobytes());h.update(np.asarray(e.core_features).tobytes())
        h.update(str((e.confidence_target,e.direction_target)).encode())
    return h.hexdigest()


def supervision_profile(factory):
    values={'confidence':set(),'direction':set()};counts={'confidence':0,'direction':0}
    for packet in factory():
        examples=packet.examples if hasattr(packet,'examples') else packet
        for e in examples:
            if math.isfinite(float(e.confidence_target)):
                if not 0<=float(e.confidence_target)<=1:raise ValueError('Evidence quality target outside [0,1]')
                values['confidence'].add(float(e.confidence_target));counts['confidence']+=1
            if int(e.direction_target)>=0:
                if int(e.direction_target) not in (0,1,2):raise ValueError('Evidence direction target outside registered classes')
                values['direction'].add(int(e.direction_target));counts['direction']+=1
    return {k:{'n':counts[k],'distinct':sorted(values[k]),'identifiable':len(values[k])>1} for k in values}


def fit_private_eventset_streaming(train_factory:Callable,validation_factory:Callable,*,
        event_feature_dim,core_feature_dim,seed,epochs=40,patience=6,batch_size=128,
        learning_rate=.002,hidden_dim=96,dropout=.2,device='cpu',
        checkpoint_root=None,resume=False,input_binding=None,deadline=None,
        checkpoint_steps=20,stop_requested=None,interrupt_after=None):
    """Original EventSet model with independent validation and batch-position resume.

    A resumed stream is reopened deterministically. Completed packets are skipped,
    not replayed through the optimizer. Changed packets/inputs fail closed.
    ``interrupt_after`` is exclusively a bounded CPU test hook.
    """
    import time
    from pathlib import Path
    from .resume_state_r4 import canonical_sha,save,load,TrainingPaused
    from .optimizer_guard_v2 import optimizer_step_with_global_delta_guard
    torch=_require_torch();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if min(epochs,patience,batch_size,checkpoint_steps)<=0:raise ValueError('Evidence training bounds must be positive')
    if checkpoint_root is not None and not isinstance(input_binding,dict):raise ValueError('Checkpointed evidence requires explicit input hashes')
    train_profile=supervision_profile(train_factory);validation_profile=supervision_profile(validation_factory)
    if not any(v['n'] for v in train_profile.values()):raise RuntimeError('No supervision in Evidence training; no model can be declared trained')
    allowed_targets={k:train_profile[k]['identifiable'] and validation_profile[k]['identifiable'] for k in train_profile}
    if not any(allowed_targets.values()):raise ValueError('No supervised validation with identifiable independent targets; do not spend GPU on one-class labels')
    model=build_fresh_private_eventset_head(event_feature_dim=event_feature_dim,core_feature_dim=core_feature_dim,
                seed=seed,hidden_dim=hidden_dim,dropout=dropout).to(device)
    initial=model_parameter_sha256(model);optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.002)
    environment={'torch':str(torch.__version__),'numpy':np.__version__,'device':str(device),'cuda_device_count':torch.cuda.device_count() if str(device).startswith('cuda') else 0}
    signature=canonical_sha({'format':'R4_EVENTSET_RESUME_V1','inputs':input_binding or {},'seed':seed,
        'epochs':epochs,'patience':patience,'batch_size':batch_size,'learning_rate':learning_rate,
        'event_feature_dim':event_feature_dim,'core_feature_dim':core_feature_dim,'hidden_dim':hidden_dim,'dropout':dropout,'environment':environment})
    rng=np.random.default_rng(seed);best=math.inf;best_state=None;stale=0;history=[];steps=0;confidence=False;direction=False
    start_epoch=1;resume_packet=0;resume_batch=0;resume_order=None;resume_digest=None
    loss_sum=0.;loss_n=0;examples_seen=0;peak_examples=0;peak_events=0
    if checkpoint_root is not None:
        root=Path(checkpoint_root)
        if (root/'LATEST.json').exists() and not resume:raise FileExistsError('Existing Evidence checkpoint requires --resume')
        old=load(root,signature) if resume else None
        if old:
            model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);initial=old['initial']
            best=old['best'];best_state=old['best_state'];stale=old['stale'];history=old['history'];steps=old['steps']
            confidence=old['confidence'];direction=old['direction'];start_epoch=old['epoch'];resume_packet=old['packet'];resume_batch=old['batch']
            resume_order=old['order'];resume_digest=old['packet_digest'];rng.bit_generator.state=old['generator_state']
            random.setstate(old['python_rng']);np.random.set_state(old['numpy_rng']);torch.set_rng_state(old['torch_rng'])
            if old['cuda_rng'] is not None and torch.cuda.is_available():torch.cuda.set_rng_state_all(old['cuda_rng'])
            loss_sum=old['loss_sum'];loss_n=old['loss_n'];examples_seen=old['examples_seen'];peak_examples=old['peak_examples'];peak_events=old['peak_events']
    # No evaluation/test factory is accepted. Validation used ONLY for model selection.
    _,validation_counts=evaluate_stream(model,validation_factory,batch_size=batch_size,device=device,allowed_targets=allowed_targets)
    def checkpoint(epoch,packet,batch,order,packet_digest):
        if checkpoint_root is None:return
        save(checkpoint_root,{'signature':signature,'format':'R4_EVENTSET_RESUME_V1','steps':steps,
            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'initial':initial,'best':best,'best_state':best_state,
            'stale':stale,'history':history,'confidence':confidence,'direction':direction,
            'epoch':epoch,'packet':packet,'batch':batch,'order':order,'packet_digest':packet_digest,
            'generator_state':rng.bit_generator.state,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'loss_sum':loss_sum,'loss_n':loss_n,'examples_seen':examples_seen,'peak_examples':peak_examples,'peak_events':peak_events})
    def paused():return (deadline is not None and time.time()>=deadline) or (stop_requested is not None and stop_requested())
    last_save=time.time()
    for epoch in range(start_epoch,epochs+1):
        if stale>=patience:break
        model.train()
        for packet_index,packet in enumerate(train_factory()):
            if epoch==start_epoch and packet_index<resume_packet:continue
            examples=packet.examples if hasattr(packet,'examples') else packet
            checksum=_packet_digest(examples)
            is_resume=epoch==start_epoch and packet_index==resume_packet and resume_order is not None
            if is_resume:
                if checksum!=resume_digest:raise RuntimeError('EVIDENCE_RESUME_PACKET_CHANGED')
                order=resume_order;batch_start=resume_batch
            else:
                order=rng.permutation(len(examples)).tolist();batch_start=0
                examples_seen+=len(examples);peak_examples=max(peak_examples,len(examples));peak_events=max(peak_events,sum(e.event_count for e in examples))
            for start in range(batch_start,len(order),batch_size):
                if paused():checkpoint(epoch,packet_index,start,order,checksum);raise TrainingPaused('EVIDENCE_PAUSED_BUDGET_OR_SIGNAL')
                batch=_batch_tensors(examples,order[start:start+batch_size],device)
                if not allowed_targets['confidence']:batch['confidence'].fill_(float('nan'))
                if not allowed_targets['direction']:batch['direction'].fill_(-1)
                confidence |= bool(torch.isfinite(batch['confidence']).any());direction |= bool(batch['direction'].ge(0).any())
                result=model(batch['events'],batch['mask'],batch['core']);terms=_loss_terms(torch,result,batch)
                if terms:
                    loss=torch.stack(terms).sum()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite evidence loss')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer_step_with_global_delta_guard(model,optimizer,torch=torch,objective_value=float(loss.detach().cpu()))
                    steps+=1;loss_sum+=float(loss.detach().cpu());loss_n+=1
                next_start=start+batch_size
                last_batch=next_start>=len(order)
                cursor=(epoch,packet_index+1,0,None,None) if last_batch else (epoch,packet_index,next_start,order,checksum)
                if steps%checkpoint_steps==0 or time.time()-last_save>=60:
                    checkpoint(*cursor);last_save=time.time()
                if interrupt_after is not None and steps>=interrupt_after:
                    checkpoint(*cursor);raise TrainingPaused('EVIDENCE_PAUSED_TEST_INTERRUPT')
            resume_order=None;resume_digest=None;resume_batch=0
        if loss_n==0:raise RuntimeError('No supervised evidence optimizer steps; cannot emit TRAINED/PASS')
        validation_loss,_=evaluate_stream(model,validation_factory,batch_size=batch_size,device=device,allowed_targets=allowed_targets)
        history.append({'epoch':epoch,'train_loss':loss_sum/loss_n,'validation_loss':validation_loss,
                        'examples_seen':examples_seen,'max_resident_examples':peak_examples,'max_resident_events':peak_events})
        if validation_loss<best-1e-6:
            best=validation_loss;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
        else:stale+=1
        loss_sum=0.;loss_n=0;examples_seen=0;peak_examples=0;peak_events=0
        checkpoint(epoch+1,0,0,None,None)
        if stale>=patience:break
    if best_state is None or steps==0:raise RuntimeError('No evidence checkpoint')
    model.load_state_dict(best_state);final=model_parameter_sha256(model)
    if final==initial:raise RuntimeError('Evidence parameters did not change')
    return PrivateHeadFit(model=model,initial_parameter_sha256=initial,final_parameter_sha256=final,
                         history=history,optimizer_steps=steps,confidence_supervision_available=confidence,
                         direction_supervision_available=direction)
