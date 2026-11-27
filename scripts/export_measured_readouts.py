"""Export measured patient readouts on host149; does not train models."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.measured_readouts import materialize_measured_readouts

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bindings',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cancer',required=True,choices=['BRCA','LUAD','COAD'])
    parser.add_argument('--context',required=True,choices=['train_label_fold2','train_label_fold3','train_label_fold4','validation'])
    args=parser.parse_args()
    result=materialize_measured_readouts(args.bindings,args.output,cancer=args.cancer,context=args.context,progress=lambda s:print(s,flush=True))
    print('COMPLETE',result['biological_targets'],result['readout_columns'],flush=True)
