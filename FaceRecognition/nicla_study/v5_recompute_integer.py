"""Re-evaluate existing gray checkpoints with the actually MCU-matching backend."""
import os,json,hashlib,time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np,tensorflow as tf
from .v5_learning import assess
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'
def main():
    rows=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text())['rows']
    with np.load(S/'runs/host_reference/nonfinal_features.npz') as ar:gray=ar['gray96'];detected=ar['detected']
    ids=[i for i,r in enumerate(rows) if r['role'] in ['unknown_calibration','development','unknown_development']]
    selected=[rows[i] for i in ids];x=gray[ids].astype(np.float32)/255;det=detected[ids]
    roles=np.array([r['role'] for r in selected]);labels=np.array([int(r['label'][1:])-1 if r['known'] else 5 for r in selected]);results=[]
    for seed in range(53001,53011):
        for stage in ['frozen','finetuned']:
            path=S/'runs/learning_alpha_0.1'/str(seed)/stage/'int8.tflite'
            it=tf.lite.Interpreter(model_path=str(path),num_threads=2,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);it.allocate_tensors();inp=it.get_input_details()[0];output=it.get_output_details()[0];values=np.zeros((len(ids),6),np.float32)
            for i in np.flatnonzero(det):
                scale,zp=inp['quantization'];q=np.clip(np.rint(x[i:i+1]/scale)+zp,-128,127).astype(np.int8);it.set_tensor(inp['index'],q);it.invoke();y=it.get_tensor(output['index'])[0];scale,zp=output['quantization'];values[i]=(y.astype(np.float32)-zp)*scale
            metrics=assess(values,roles,labels,det);metrics.update(seed=seed,stage=stage,backend='TF2.10.1 BUILTIN_REF',model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evaluated_roles=sorted(set(roles)),source_detector_failures_retained=True,source_parent_checkpoint_retrained=False)
            np.savez_compressed(path.parent/'int8_reference_predictions.npz',values=values,parent_indices=np.array(ids))
            (path.parent/'INT8_REFERENCE_RESULT.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n');results.append(metrics)
            print(json.dumps({'seed':seed,'stage':stage,'known_top1_full':metrics['known_raw_top1_full'],'known_acceptance':metrics['known_correct_acceptance'],'unknown_false':metrics['unknown_false_identifications']}),flush=True)
    (S/'reports/M5/GRAY_REFERENCE_RECOMPUTE.json').write_text(json.dumps({'status':'recomputed','roots_per_stage':10,'results':results,'new_gradient_trainings':0,'historical_optimized_results_preserved':True},indent=2)+'\n')
if __name__=='__main__':main()
