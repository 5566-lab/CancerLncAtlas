"""Resolution-four follow-up using unchanged train-only module ridge equations."""
from __future__ import annotations
import argparse,json,socket,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from scripts.run_network_validation import split_plan,check_receipt,reweight
from cc_hhgt.network_ridge import make_context,base_kernel,group_kernels,alignment_weights,weighted_path,extract_weighted_fit,ARMS,BLOCKS,LAMBDAS,POWERS
from cc_hhgt.graph_ridge import target_policy,ridge_path,extract_fit
from cc_hhgt.m0_covariates import complete_rows
from cc_hhgt.network_evidence import require_physical_network_receipt
FINE_MODES=("union","random_101","random_202","random_303","random_404","random_505")

def variant_name(mode):
    return "module_resolution4_"+(mode if mode=="union" else "union_"+mode)

def run(inputs,networks,output,cancer,phase):
    if socket.gethostname()!="149":raise RuntimeError("HOST_149_REQUIRED")
    src=Path(inputs)/cancer;net=Path(networks)/cancer;d=Path(output)/cancer/phase;d.mkdir(parents=True,exist_ok=True)
    code=[Path(__file__),ROOT/"scripts/run_network_validation.py",ROOT/"cc_hhgt/network_ridge.py",ROOT/"cc_hhgt/graph_ridge.py",ROOT/"cc_hhgt/m0_covariates.py",ROOT/"cc_hhgt/network_evidence.py"]
    preflight=dict(target_host="149",cancer=cancer,phase=phase,input_receipt=sha(src/"COMPLETE.json"),network_receipt=sha(net/"COMPLETE.json"),plan=sha(ROOT/"config/network_resolution4_20261106.json"),code={str(p.relative_to(ROOT)):sha(p) for p in code},preprocessing="Unchanged original training-only preprocessing, clinical complete cases and target-source exclusions",resolution=4,version=1)
    if (d/"LAUNCH_PREFLIGHT.json").exists() and json.loads((d/"LAUNCH_PREFLIGHT.json").read_text())!=preflight:raise ValueError("RUN_CONFIGURATION_DRIFT")
    dump(d/"LAUNCH_PREFLIGHT.json",preflight)
    if (d/"COMPLETE.json").exists():check_receipt(d);return
    check_receipt(src);check_receipt(net)
    require_physical_network_receipt(json.loads((src/"NETWORK_AUDIT.json").read_text()))
    require_physical_network_receipt(json.loads((net/"CORRECTED_RNA_NETWORK_AUDIT.json").read_text()))
    resolution_policy=json.loads((net/"PROVENANCE.json").read_text())
    if resolution_policy.get("resolution")!=4 or resolution_policy.get("fitted_modes")!=list(FINE_MODES):raise ValueError("RESOLUTION4_PARTITION_REQUIRED")
    frames={b:pd.read_parquet(src/(b+".parquet")) for b in ("RNA",)+BLOCKS};pids=frames["RNA"].index
    if not all(f.index.equals(pids) for f in frames.values()):raise ValueError("PATIENT_AXIS_DRIFT")
    blocks={b:f.to_numpy(float) for b,f in frames.items()};del frames
    features=pd.read_csv(src/"RNA_FEATURES.tsv",sep="\t");fmap=pd.read_parquet(net/"FEATURE_MAP.parquet");cov=pd.read_parquet(src/"COVARIATES.parquet").reindex(pids);yf=pd.read_parquet(src/"Y.parquet").reindex(pids);meta=pd.read_csv(src/"TARGETS.tsv",sep="\t")
    policies=[target_policy(r) for r in meta.to_dict("records")];y=yf[meta.target_id].to_numpy(float)
    plan=split_plan(json.loads((src/"SPLITS.json").read_text()),len(pids));sp=plan[phase];dump(d/"SPLIT.json",sp);outer=np.asarray(sp["train"],int);test=np.asarray(sp["test"],int);final=phase=="final"
    groups={}
    for j,p in enumerate(policies):
        valid=np.isfinite(y[:,j])&complete_rows(cov,p["use_purity"])
        groups.setdefault((p["use_purity"],p["allowed_blocks"],tuple(valid)),[]).append(j)
    print("INPUT_READY",cancer,phase,"patients",len(pids),"targets",len(meta),"groups",len(groups),flush=True)
    contexts={};models={};allpred=[];allaudit=[]
    def context(purity):
        if purity not in contexts:
            cc=[]
            for name,rows in [("outer",outer)]+[("inner"+str(i),np.asarray(s["train"],int)) for i,s in enumerate(sp["inner"])]:
                started=time.time();ctx=make_context(blocks,features,fmap,cov,rows,purity)
                for unused in ("_rna_x","_rna_index","_rna_types"):ctx.pop(unused,None)
                cc.append(ctx)
                print("CONTEXT",cancer,phase,purity,name,round(time.time()-started,2),flush=True)
            contexts[purity]=cc
            if final:joblib.dump({k:v for k,v in cc[0].items() if k in ["train","design","design_spec","supported","preprocess","scales"]},d/("PREPROCESS_purity"+str(int(purity))+".joblib"),compress=3)
        return contexts[purity]
    for gi,(key,jlist) in enumerate(groups.items()):
        purity,allowed,vv=key;valid=np.asarray(vv,bool);checkpoint=d/("group_"+str(gi)+".joblib")
        if checkpoint.exists():
            saved=joblib.load(checkpoint);allpred.extend(saved["predictions"]);allaudit.extend(saved["audit"]);models.update(saved["models"]);print("RESUME_GROUP",gi,flush=True);continue
        cc=context(purity);oc=cc[0];ics=cc[1:];ft=outer[valid[outer]];te=test[valid[test]&oc["supported"][test]]
        pairs=[]
        for s,ic in zip(sp["inner"],ics):
            tr=np.asarray(s["train"],int);va=np.asarray(s["validation"],int);pairs.append((tr[valid[tr]],va[valid[va]&ic["supported"][va]]))
        enough=len(ft)>=12 and (final or len(te)>0) and all(len(tr)>=12 and len(va)>=3 for tr,va in pairs)
        js=[];audit=[];predictions=[];gmodels={}
        for j in jlist:
            reason="INSUFFICIENT_SUPPORTED_NESTED_CV_OBSERVATIONS" if not enough else "CONSTANT_TRAINING_TARGET" if np.std(y[ft,j])<1e-12 or any(np.std(y[tr,j])<1e-12 for tr,va in pairs) else None
            if reason:
                for mode in FINE_MODES:
                    for arm in ARMS:audit.append(dict(target_id=meta.iloc[j].target_id,variant=variant_name(mode),arm=arm,status=reason,phase=phase,n_train=len(ft),n_test=len(te)))
            else:js.append(j)
        if js:
            yy=y[np.ix_(ft,js)];tids=meta.target_id.iloc[js].tolist();validation_count=sum(len(va) for tr,va in pairs);base_losses={}
            for arm in ARMS:
                loss=np.zeros((len(LAMBDAS),len(js)))
                for ic,(tr,va) in zip(ics,pairs):
                    pp,_=ridge_path(base_kernel(ic,arm,allowed),ic["design"],tr,va,y[np.ix_(tr,js)],LAMBDAS);loss+=np.sum((pp-y[np.ix_(va,js)][None])**2,axis=1)
                base_losses[arm]=loss
            for mode in FINE_MODES:
                variant=variant_name(mode)
                for arm in ARMS:
                    started=time.time();losses=np.zeros((len(POWERS),len(LAMBDAS),len(js)));losses[0]=base_losses[arm]
                    for ic,(tr,va) in zip(ics,pairs):
                        kg,mass,labels=group_kernels(ic,arm,allowed,mode);w1=alignment_weights(kg,mass,ic["design"],tr,y[np.ix_(tr,js)],1.)
                        for oi,power in enumerate(POWERS[1:],1):
                            pp,_=weighted_path(kg,reweight(w1,mass,power),ic["design"],tr,va,y[np.ix_(tr,js)],LAMBDAS);losses[oi]+=np.sum((pp-y[np.ix_(va,js)][None])**2,axis=1)
                    losses/=validation_count;best=losses.reshape(-1,len(js)).argmin(axis=0);choice=np.asarray(POWERS)[best//len(LAMBDAS)];lchoice=LAMBDAS[best%len(LAMBDAS)]
                    kg,mass,labels=group_kernels(oc,arm,allowed,mode);w1=alignment_weights(kg,mass,oc["design"],ft,yy,1.)
                    for power in POWERS:
                        take=np.flatnonzero(choice==power)
                        if not len(take):continue
                        if power:
                            weights=reweight(w1[:,take],mass,power);pp,states=weighted_path(kg,weights,oc["design"],ft,te,yy[:,take],LAMBDAS,return_fits=final)
                        else:pp,state=ridge_path(base_kernel(oc,arm,allowed),oc["design"],ft,te,yy[:,take],LAMBDAS)
                        for jj,pos in enumerate(take):
                            li=int(np.argmin(abs(LAMBDAS-lchoice[pos])));tid=tids[pos]
                            audit.append(dict(target_id=tid,variant=variant,arm=arm,status="FITTED",phase=phase,parameter=float(power),lam=float(lchoice[pos]),inner_mse=float(losses.reshape(-1,len(js))[best[pos],pos]),n_train=len(ft),n_test=len(te),n_inner_train_min=min(len(tr) for tr,va in pairs),n_inner_validation_min=min(len(va) for tr,va in pairs),inner_validations=len(pairs),purity_covariate_used=purity,allowed_blocks=";".join(allowed),excluded_blocks=";".join(policies[js[pos]]["excluded_blocks"]),role=policies[js[pos]]["role"]))
                            if final:
                                if power:fit=extract_weighted_fit(states[jj],lchoice[pos],weights[:,jj])
                                else:
                                    ss={k:(v[:,jj:jj+1] if k in ["uy","covbeta"] else v) for k,v in state.items()};fit=extract_fit(ss,lchoice[pos],yy[:,pos:pos+1]);fit["dual"]=fit["dual"].ravel();fit["nuisance"]=fit["nuisance"].ravel();fit["group_weights"]=mass
                                fit.update(variant=variant,kind="module",mode=mode,arm=arm,parameter=float(power),allowed_blocks=allowed,purity=purity,group_labels=labels,resolution=4);gmodels[(tid,variant,arm)]=fit
                            else:predictions.append(pd.DataFrame(dict(patient_id=pids[te],target_id=tid,variant=variant,arm=arm,phase=phase,repeat=sp["repeat"],fold=sp["fold"],observed=y[te,js[pos]],predicted=pp[li,:,jj],train_mean_baseline=float(yy[:,pos].mean()))))
                    print("MODEL_COMPLETE",cancer,phase,gi,variant,arm,"targets",len(js),"seconds",round(time.time()-started,2),flush=True)
        saved=dict(audit=audit,predictions=predictions,models=gmodels);tmp=checkpoint.with_suffix(".tmp");joblib.dump(saved,tmp,compress=3);tmp.replace(checkpoint);allaudit.extend(audit);allpred.extend(predictions);models.update(gmodels)
        print("GROUP_COMPLETE",cancer,phase,gi+1,"of",len(groups),flush=True)
    pd.DataFrame(allaudit).to_parquet(d/"AUDIT.parquet",index=False)
    pd.concat(allpred,ignore_index=True).to_parquet(d/"PREDICTIONS.parquet",index=False) if allpred else pd.DataFrame().to_parquet(d/"PREDICTIONS.parquet")
    if final:joblib.dump(models,d/"MODELS.joblib",compress=3)
    meta.to_csv(d/"TARGETS.tsv",sep="\t",index=False)
    dump(d/"COMPLETE.json",dict(target_host="149",status="COMPLETE",files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!="COMPLETE.json"}))
    print("PHASE_COMPLETE",cancer,phase,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("inputs","networks","output","cancer","phase"):p.add_argument("--"+name,required=True)
    run(**vars(p.parse_args()))
