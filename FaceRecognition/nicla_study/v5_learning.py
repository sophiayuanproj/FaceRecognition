"""Finite ten-root LFW-v5 learning experiment; never reads future-final data.

Gray box-crop classifiers are intentionally distinct from the aligned host
reference. Every training root runs balanced batches and an overfit control.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import tensorflow as tf

from .d_campaign import build_classifier, unfreeze_top_fraction

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / 'work/project-research-v5'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def softmax(x):
    y = np.exp(x-x.max(axis=1, keepdims=True))
    return y/y.sum(axis=1, keepdims=True)


def assess(logits, roles, labels, detected):
    p=softmax(logits)
    confidence=p[:,:5].max(axis=1)
    pred=p.argmax(axis=1)
    cal=(roles=='unknown_calibration') & detected
    # Match confidence includes OTHER rejection. Threshold fitted only on cal.
    values=confidence[cal & (pred<5)]
    threshold=float(np.nextafter(values.max(),np.float32(np.inf))) if len(values) else 0.0
    accept=detected & (pred<5) & (confidence>=threshold)
    known=roles=='development'; unknown=roles=='unknown_development'
    per=[]
    for k in range(5):
        m=known & (labels==k)
        per.append({'label':'K%02d'%(k+1),'n':int(m.sum()),
            'correct_accepted':int((m & accept & (pred==labels)).sum()),
            'coverage':float(accept[m].mean()),'top1_conditional':float((pred[m & detected]==k).mean()),
            'correct_acceptance':float((accept & (pred==labels))[m].mean())})
    false=int((accept & unknown).sum()); n=int(unknown.sum())
    return {'threshold':threshold,'known_n':int(known.sum()),'per_person':per,
        'known_correct_acceptance':float((accept & (pred==labels))[known].mean()),
        'known_raw_top1_full':float((detected & (pred==labels))[known].mean()),
        'known_coverage':float(accept[known].mean()),
        'unknown_false_identifications':false,'unknown_n':n,'unknown_fpir':false/n,
        'unknown_image_iid_zero_event_upper95':1-0.05**(1/n) if false==0 else None,
        'unknown_independent_identity_groups':30,
        'uncertainty_limit':'Images within an identity are dependent; image-iid upper bound is descriptive only. 0/30 identity-events has upper95=9.50%.',
        'useful_gate_pass':False,'physical_qualification':False}


def infer(model,x):
    return np.concatenate([model(x[i:i+64],training=False).numpy() for i in range(0,len(x),64)])


def export_and_score(model,folder,x,roles,labels,detected,train_idx):
    folder.mkdir(parents=True,exist_ok=True)
    model.save(str(folder/'model.h5'),include_optimizer=False)
    original=infer(model,x)
    converter=tf.lite.TFLiteConverter.from_keras_model(model)
    float_bytes=converter.convert(); (folder/'float.tflite').write_bytes(float_bytes)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    # Training only, deterministic balanced pool across six labels.
    rep=[]
    for k in range(6):
        rep.extend(train_idx[labels[train_idx]==k][:30].tolist())
    def representative():
        for i in rep:
            yield [x[i:i+1]]
    converter.representative_dataset=representative
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type=tf.int8
    converter.inference_output_type=tf.int8
    integer=converter.convert(); (folder/'int8.tflite').write_bytes(integer)
    results={}; outputs={}
    for name,payload in [('float',float_bytes),('int8',integer)]:
        it=tf.lite.Interpreter(model_content=payload,num_threads=2);it.allocate_tensors()
        inp=it.get_input_details()[0];out=it.get_output_details()[0]
        values=[]
        for image in x:
            if name=='int8':
                scale,zp=inp['quantization']
                raw=np.clip(np.rint(image/scale)+zp,-128,127).astype(np.int8)
            else:raw=image
            it.set_tensor(inp['index'],raw[None]);it.invoke()
            y=it.get_tensor(out['index'])[0]
            if name=='int8':
                scale,zp=out['quantization'];y=(y.astype(np.float32)-zp)*scale
            values.append(y)
        values=np.asarray(values)
        if not np.isfinite(values).all():raise ValueError('Nonfinite predictions')
        outputs[name]=values
        results[name]=assess(values,roles,labels,detected)
        results[name].update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
            input={'shape':inp['shape'].tolist(),'dtype':str(inp['dtype']),'quantization':list(inp['quantization'])},
            output={'shape':out['shape'].tolist(),'dtype':str(out['dtype']),'quantization':list(out['quantization'])},
            operators=sorted({v['op_name'] for v in it._get_ops_details()}))
    results['parity']={'native_float_tflite_max_abs':float(np.max(np.abs(original-outputs['float']))),
        'ptq_vs_float_top1_disagreements':int((outputs['int8'].argmax(1)!=outputs['float'].argmax(1)).sum()),
        'scope':'all nonfinal roles; read role-specific metrics before interpretation'}
    np.savez_compressed(folder/'predictions.npz',native=original,**outputs)
    dump(folder/'RESULT.json',results)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha',type=float,choices=[.1,.25],required=True)
    parser.add_argument('--first-root',type=int,default=53001)
    parser.add_argument('--roots',type=int,default=10)
    args=parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    ref=STUDY/'runs/host_reference'
    index=json.loads((ref/'nonfinal_index.json').read_text());rows=index['rows']
    if any('future_final' in r['role'] for r in rows):raise ValueError('Final role prohibited')
    cache=ref/'nonfinal_features.npz'
    if hashlib.sha256(cache.read_bytes()).hexdigest()!=index['cache_sha256']:raise ValueError('Cache changed')
    data=np.load(cache);detected=data['detected'];x=data['gray96'].astype(np.float32)/255
    roles=np.asarray([r['role'] for r in rows]);labels=np.asarray([int(r['label'][1:])-1 if r['known'] else 5 for r in rows])
    train=np.flatnonzero(np.isin(roles,['train','other_train']) & detected)
    stop=np.flatnonzero((roles=='stopping') & detected)
    pools=[train[labels[train]==k] for k in range(6)]
    if any(not len(v) for v in pools):raise ValueError('Empty training class')
    base=STUDY/'runs'/('learning_alpha_'+str(args.alpha))
    protocol={'version':'v5-learning-1','alpha':args.alpha,'seeds':list(range(args.first_root,args.first_root+args.roots)),
        'data_manifest_sha256':hashlib.sha256((STUDY/'data/manifests/lfw_roles.jsonl').read_bytes()).hexdigest(),
        'cache_sha256':index['cache_sha256'],'batch':24,'balanced_per_class':4,
        'max_head_epochs':20,'max_finetune_epochs':40,'steps_per_epoch':10,
        'minimum_epochs':15,'stopping_patience':8,'stopping':'known stopping cross entropy, no development selection',
        'optimizer':'Adam head1e-3 / fine1e-4','backbone_finetune_fraction':.3,
        'other_label':'200 identity-disjoint training identities, balanced as one class',
        'augmentation':'horizontal flip p=.5, multiplicative brightness .85..1.15, additive -.03...03; train only; not camera calibrated',
        'tensor':'gray96 /255; source YuNet square box crops; no alignment or camera parity assumed',
        'physical_ready':False,'future_final_used':False}
    if (base/'PROTOCOL.json').exists() and json.loads((base/'PROTOCOL.json').read_text())!=protocol:
        raise ValueError('Protocol changed; new experiment namespace required')
    dump(base/'PROTOCOL.json',protocol)
    for seed in protocol['seeds']:
        folder=base/str(seed)
        if (folder/'COMPLETE.json').exists():continue
        started=time.monotonic();tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        model,_,_=build_classifier(ROOT,args.alpha,True,6,seed,'lfw_v5')
        rng=np.random.default_rng(seed)
        def batches():
            chosen=np.concatenate([rng.choice(pool,4,replace=len(pool)<4) for pool in pools]);rng.shuffle(chosen)
            xb=x[chosen].copy()
            flips=rng.random(len(xb))<.5;xb[flips]=xb[flips,:,::-1,:]
            xb=np.clip(xb*rng.uniform(.85,1.15,(len(xb),1,1,1))+rng.uniform(-.03,.03,(len(xb),1,1,1)),0,1).astype(np.float32)
            return xb,labels[chosen]
        # Separate train-only overfit head diagnostic (then restore fresh model).
        tiny=np.concatenate([rng.choice(pool,5,replace=False) for pool in pools])
        model.compile(optimizer=tf.keras.optimizers.Adam(.003),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        losses=[]
        for step in range(150):
            loss=float(model.train_on_batch(x[tiny],labels[tiny]));losses.append(loss)
            if not np.isfinite(loss):raise ValueError('Nonfinite overfit loss')
        control={'n':len(tiny),'steps':150,'initial_loss':losses[0],'final_loss':losses[-1],
                 'accuracy':float((infer(model,x[tiny]).argmax(1)==labels[tiny]).mean()),
                 'balanced_counts':dict(Counter(map(str,labels[tiny])))}
        dump(folder/'OVERFIT.json',control)
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        model,_,_=build_classifier(ROOT,args.alpha,True,6,seed,'lfw_v5')
        histories=[];stage_results={}
        for stage,epochs,lr in [('frozen',20,.001),('finetuned',40,.0001)]:
            if stage=='finetuned':unfreeze_top_fraction(model,.3)
            model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
            best=float('inf');best_weights=None;patience=0
            for epoch in range(epochs):
                train_losses=[]
                for step in range(10):
                    xb,yb=batches();loss=float(model.train_on_batch(xb,yb))
                    if not np.isfinite(loss):raise ValueError('Nonfinite training loss')
                    train_losses.append(loss)
                logits=infer(model,x[stop]);val=float(tf.keras.losses.sparse_categorical_crossentropy(labels[stop],logits,from_logits=True).numpy().mean())
                row={'stage':stage,'epoch':epoch,'train_loss':float(np.mean(train_losses)),
                     'stopping_loss':val,'stopping_top1':float((logits.argmax(1)==labels[stop]).mean())}
                histories.append(row)
                if val<best-1e-5:best=val;best_weights=model.get_weights();patience=0
                else:patience+=1
                dump(folder/'CURVES.json',histories)
                if epoch>=14 and patience>=8:break
            model.set_weights(best_weights)
            stage_results[stage]=export_and_score(model,folder/stage,x,roles,labels,detected,train)
            print(json.dumps({'alpha':args.alpha,'seed':seed,'stage':stage,'overfit':control,
                'float':stage_results[stage]['float']['known_correct_acceptance'],
                'int8':stage_results[stage]['int8']['known_correct_acceptance'],
                'elapsed_s':time.monotonic()-started}),flush=True)
        dump(folder/'COMPLETE.json',{'seed':seed,'actual_gradient_training':True,'elapsed_s':time.monotonic()-started,
            'overfit':control,'stages':stage_results,'future_final_used':False})


if __name__=='__main__':main()
