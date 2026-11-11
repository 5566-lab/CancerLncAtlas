#!/usr/bin/env python3
from pathlib import Path
import argparse,json,socket,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.relation_catalog import demo_configurations
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-config',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    if socket.gethostname()!='149': raise RuntimeError('Run preparation on host 149')
    out=Path(a.out)
    if out.exists(): raise ValueError('Use a new configuration directory')
    out.mkdir(parents=True)
    configs=demo_configurations(json.loads(Path(a.base_config).read_text()),a.catalog)
    for cancer,config in configs.items():
        (out/(cancer+'.json')).write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'target_host':'149','cancers':list(configs),'one_cancer_per_configuration':True,'training_started':False}))
