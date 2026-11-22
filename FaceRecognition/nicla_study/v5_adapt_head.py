"""Ten-root face-aware adaptation of the actual64px encoder's final layers.

Frozen-prefix features are cached once. Gradient updates train the spatial
depthwise pooling, pointwise projection and BN affine parameters, using the
validated aligned112 teacher as a target for unaligned64 box inputs. No final
images are loaded. No new from-scratch face pretraining is claimed.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2,numpy as np,tensorflow as tf
from .v5_flatbuffer import fuse
tf.compat.v1.disable_eager_execution()
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'
FEATURE='MobileFaceNet/MobileFaceNet/Conv2d_7/add:0'
OUTPUT='MobileFaceNet/Logits/SpatialSqueeze:0'
CONFIG=tf.compat.v1.ConfigProto(intra_op_parallelism_threads=2,inter_op_parallelism_threads=1)

def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def norm(v):return v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8)
def graphdef(path):g=tf.compat.v1.GraphDef();g.ParseFromString(path.read_bytes());return g

def metrics(features,rows,gallery,threshold,detected):
    roles=np.array([r['role'] for r in rows]);labels=np.array([int(r['label'][1:])-1 if r['known'] else -1 for r in rows])
    scores=features@gallery.T;conf=scores.max(1);pred=scores.argmax(1);accept=detected & (conf>=threshold)
    known=roles=='development';unknown=roles=='unknown_development'
    return {'known_n':int(known.sum()),'known_correct_acceptance':float((accept & (pred==labels))[known].mean()),
        'known_coverage':float(accept[known].mean()),'known_raw_top1_full':float((detected & (pred==labels))[known].mean()),
        'wrong_known':int((known & accept & (pred!=labels)).sum()),'unknown_n':int(unknown.sum()),'unknown_false':int((unknown & accept).sum()),
        'unknown_identity_groups':30,'threshold':threshold,'physical_qualified':False,'future_final_used':False,
        'per_person':[{'label':'K%02d'%(k+1),'n':int((known & (labels==k)).sum()),
            'correct_accepted':int((known & (labels==k) & accept & (pred==k)).sum())} for k in range(5)]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--roots',type=int,default=10)
    p.add_argument('--size',type=int,choices=[48,56,64],default=64)
    p.add_argument('--wait-for-gray-campaign',action='store_true');a=p.parse_args()
    out=S/'runs'/('adapted_encoder%d_v3'%a.size);out.mkdir(exist_ok=True)
    spatial=(a.size+15)//16
    index=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text());rows=index['rows']
    if any('future_final' in r['role'] for r in rows):raise ValueError('Final prohibited')
    roles=np.array([r['role'] for r in rows]);labels=np.array([int(r['label'][1:])-1 if r['known'] else 5 for r in rows])
    detected=np.load(S/'runs/host_reference/nonfinal_features.npz')['detected']
    teacher=norm(np.load(S/'runs/face_encoder/float_aligned_rgb.npy'))
    with np.load(S/'runs/face_encoder/inputs_nonfinal.npz') as ar:box=ar['box']
    pixels=np.stack([cv2.resize(im,(a.size,a.size)) for im in box]);x=(pixels.astype(np.float32)-127.5)/128
    parent=S/'runs'/('operator_fused_encoder%d'%a.size)/'parent.pb';original=graphdef(parent)
    selected={n.name:tf.make_ndarray(n.attr['value'].tensor) for n in original.node
              if n.op=='Const' and n.name.startswith('MobileFaceNet/Logits/')
              and (n.name.endswith('/weights') or n.name.endswith('/depthwise_weights') or n.name.endswith('/beta') or n.name.endswith('/gamma'))}
    if len(selected)!=9:raise ValueError('Unexpected adaptation parameter set: '+str(selected.keys()))
    protocol={'schema':'face-aware-head-adaptation-v3','input_size':a.size,'fused_export':'33 native PRELU operations, guarded FlatBuffer rewrite; int8 recalibrated/evaluated separately',
        'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'teacher_features_sha256':hashlib.sha256((S/'runs/face_encoder/float_aligned_rgb.npy').read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256((S/'data/manifests/lfw_roles.jsonl').read_bytes()).hexdigest(),
        'seeds':list(range(53001,53001+a.roots)),'parameters':{k:list(v.shape) for k,v in selected.items()},
        'fixed_prefix':FEATURE,'input':'unaligned RGB%d box, (pixel-127.5)/128'%a.size,'target':'aligned112 pretrained face embedding128',
        'training':'balanced6labels x4; actual gradients on9 final-layer weight/affine tensors; frozenprefix',
        'loss':'cosine KD + .2 known teacher-prototype CE(scale12) + .1 batch cosine Gram MSE',
        'augmentation':'train latent features: dropout2% and Gaussian noise1% per-channel std; not camera calibrated',
        'streams':'SeedSequence(seed).spawn(3): sampling, latent augmentation, control/gallery',
        'optimizer':'Adam3e-4, global gradient clip5','max_epochs':60,'steps_per_epoch':10,
        'stopping':'known stopping teacher cosine/CE loss, minimum15epochs, patience10',
        'overfit_control':'150updates on30 balanced training samples, restore initialization before actualfit',
        'gallery':'first20 detected training source groups per knownID, chosen by frozen manifest order',
        'quantization':'full INT8,120 balanced known training-only box inputs; model-specific thresholds',
        'future_final_used':False,'new_training_roots':a.roots,'integer_backend':'TF2.10.1 BUILTIN_REF, exact stock FOMO/native matched on100 inputs'}
    if (out/'PROTOCOL.json').exists() and json.loads((out/'PROTOCOL.json').read_text())!=protocol:raise ValueError('Protocol mismatch')
    dump(out/'PROTOCOL.json',protocol)
    feat_path=out/'frozen_prefix_nonfinal.npy'
    if feat_path.exists():features=np.load(feat_path)
    else:
        g=graphdef(parent)
        for n in g.node:
            if n.name=='img_inputs':n.attr['shape'].shape.dim[0].size=-1
        gr=tf.Graph()
        with gr.as_default():tf.import_graph_def(g,name='')
        features=np.zeros((len(rows),spatial,spatial,512),np.float32)
        with tf.compat.v1.Session(graph=gr,config=CONFIG) as sess:
            ids=np.flatnonzero(detected)
            for start in range(0,len(ids),32):
                chosen=ids[start:start+32];features[chosen]=sess.run(gr.get_tensor_by_name(FEATURE),{gr.get_tensor_by_name('img_inputs:0'):x[chosen]})
        np.save(feat_path,features);print(json.dumps({'fixed_prefix_cached':len(features)}),flush=True)
    train=np.flatnonzero(np.isin(roles,['train','other_train']) & detected);stop=np.flatnonzero((roles=='stopping') & detected)
    pools=[train[labels[train]==k] for k in range(6)]
    teacher_gallery=norm(np.stack([teacher[train[labels[train]==k]].mean(0) for k in range(5)]))
    gallery_ids=np.concatenate([train[labels[train]==k][:20] for k in range(5)])
    eval_ids=np.unique(np.concatenate([gallery_ids,np.flatnonzero(np.isin(roles,['unknown_calibration','development','unknown_development']) & detected)]))
    rep=np.concatenate([train[labels[train]==k][:24] for k in range(5)])
    if a.wait_for_gray_campaign:
        deadline=time.monotonic()+7200
        while not (S/'runs/learning_alpha_0.1/53010/COMPLETE.json').exists():
            if time.monotonic()>deadline:raise RuntimeError('Training owner dependency did not complete within2h')
            print('Waiting for the single existing gradient campaign to finish',flush=True);time.sleep(30)
    # Never launch ten trainings before the same graph/operators have actually
    # loaded and invoked beside the camera. This file is derived from raw lab
    # blocks by the research owner, not a user-supplied optimistic switch.
    preflight=S/'reports/M5'/('operator_fused_encoder%d_PREFLIGHT.json'%a.size)
    if not preflight.exists():raise RuntimeError('Missing hardware resource preflight '+str(preflight))
    proof=json.loads(preflight.read_text())
    reference=S/'runs'/('operator_fused_encoder%d'%a.size)/'int8.tflite'
    if proof.get('model_sha256')!=hashlib.sha256(reference.read_bytes()).hexdigest() or proof.get('successful_camera_invocations',0)<10:
        raise RuntimeError('Incomplete/stale camera model preflight')
    for seed in protocol['seeds']:
        folder=out/str(seed)
        if (folder/'COMPLETE.json').exists():continue
        begun=time.monotonic();streams=np.random.SeedSequence(seed).spawn(3)
        sample_rng,aug_rng,control_rng=[np.random.default_rng(s) for s in streams]
        gr=tf.Graph()
        with gr.as_default():
            tf.compat.v1.set_random_seed(seed)
            fp=tf.compat.v1.placeholder(tf.float32,[None,spatial,spatial,512],name='cached_prefix')
            tp=tf.compat.v1.placeholder(tf.float32,[None,128],name='teacher_target');lp=tf.compat.v1.placeholder(tf.int32,[None],name='labels')
            variables={name:tf.Variable(value,name='adapt_%d'%i) for i,(name,value) in enumerate(selected.items())}
            mapping={FEATURE:fp};mapping.update({name+':0':var.read_value() for name,var in variables.items()})
            tf.import_graph_def(original,name='',input_map=mapping)
            raw=gr.get_tensor_by_name(OUTPUT);embedding=tf.math.l2_normalize(raw,axis=1)
            kd=tf.reduce_mean(1-tf.reduce_sum(embedding*tp,axis=1));known_mask=tf.cast(lp<5,tf.float32)
            ce=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.minimum(lp,4),logits=12*tf.matmul(embedding,tf.constant(teacher_gallery),transpose_b=True))
            ce=tf.reduce_sum(ce*known_mask)/tf.maximum(tf.reduce_sum(known_mask),1.)
            gram=tf.reduce_mean(tf.square(tf.matmul(embedding,embedding,transpose_b=True)-tf.matmul(tp,tp,transpose_b=True)))
            loss=kd+.2*ce+.1*gram
            opt=tf.compat.v1.train.AdamOptimizer(.0003)
            grads=tf.gradients(loss,list(variables.values()))
            if any(g is None for g in grads):raise ValueError('Disconnected adaptation gradient')
            finite=tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g)) for g in grads]))
            clipped,_=tf.clip_by_global_norm(grads,5.0);step=opt.apply_gradients(zip(clipped,variables.values()))
            init=tf.compat.v1.global_variables_initializer()
        with tf.compat.v1.Session(graph=gr,config=CONFIG) as sess:
            sess.run(init)
            tiny=np.concatenate([control_rng.choice(pool,5,replace=False) for pool in pools]);control=[]
            for _ in range(150):
                value,okay,_=sess.run([loss,finite,step],{fp:features[tiny],tp:teacher[tiny],lp:labels[tiny]})
                if not okay or not np.isfinite(value):raise ValueError('Nonfinite control gradient/loss')
                control.append(float(value))
            dump(folder/'OVERFIT.json',{'samples':30,'updates':150,'initial_loss':control[0],'final_loss':control[-1],'all_gradients_finite':True})
            sess.run(init);best=float('inf');best_values=None;wait=0;curves=[]
            channel_std=np.maximum(features[train].std(axis=(0,1,2)),1e-5)[None,None,None,:]
            for epoch in range(60):
                losses=[]
                for _ in range(10):
                    ids=np.concatenate([sample_rng.choice(pool,4,replace=False) for pool in pools]);sample_rng.shuffle(ids)
                    batch=features[ids].copy();batch*=((aug_rng.random(batch.shape)>.02)/.98).astype(np.float32)
                    batch+=aug_rng.normal(0,.01,batch.shape).astype(np.float32)*channel_std
                    value,okay,_=sess.run([loss,finite,step],{fp:batch,tp:teacher[ids],lp:labels[ids]})
                    if not okay or not np.isfinite(value):raise ValueError('Nonfinite training gradient/loss')
                    losses.append(float(value))
                val=float(sess.run(loss,{fp:features[stop],tp:teacher[stop],lp:labels[stop]}))
                curves.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'stopping_loss':val,'gradients_finite':True});dump(folder/'CURVES.json',curves)
                if val<best-1e-5:best=val;best_values=sess.run(variables);wait=0
                else:wait+=1
                if epoch>=14 and wait>=10:break
            adapted=graphdef(parent)
            for n in adapted.node:
                if n.name in best_values:n.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(best_values[n.name]))
            pb=folder/'adapted.pb';pb.write_bytes(adapted.SerializeToString())
        results={}
        for kind in ['float','int8']:
            conv=tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(str(pb),['img_inputs'],[OUTPUT[:-2]],input_shapes={'img_inputs':[1,a.size,a.size,3]})
            if kind=='int8':
                conv.optimizations=[tf.lite.Optimize.DEFAULT]
                def reps():
                    for i in rep:yield [x[i:i+1]]
                conv.representative_dataset=reps;conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type=tf.int8;conv.inference_output_type=tf.int8
            expanded=conv.convert();payload,fusion=fuse(expanded)
            (folder/(kind+'_expanded.tflite')).write_bytes(expanded);dump(folder/(kind+'_fusion.json'),fusion)
            path=folder/(kind+'.tflite');path.write_bytes(payload)
            it=tf.lite.Interpreter(model_content=payload,num_threads=2,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);it.allocate_tensors();inp=it.get_input_details()[0];output=it.get_output_details()[0]
            vectors=np.zeros((len(rows),128),np.float32)
            for i in eval_ids:
                raw=x[i:i+1]
                if kind=='int8':scale,zp=inp['quantization'];raw=np.clip(np.rint(raw/scale)+zp,-128,127).astype(np.int8)
                it.set_tensor(inp['index'],raw);it.invoke();v=it.get_tensor(output['index'])[0]
                if kind=='int8':scale,zp=output['quantization'];v=(v.astype(np.float32)-zp)*scale
                if not np.isfinite(v).all():raise ValueError('Nonfinite exported embedding')
                vectors[i]=v/max(float(np.linalg.norm(v)),1e-8)
            gallery=norm(np.stack([vectors[gallery_ids[labels[gallery_ids]==k]].mean(0) for k in range(5)]))
            scores=vectors@gallery.T;threshold=float(np.nextafter(scores[(roles=='unknown_calibration') & detected].max(),np.float32(np.inf)))
            result=metrics(vectors,rows,gallery,threshold,detected)
            result.update(bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                input_shape=inp['shape'].tolist(),input_quantization=list(inp['quantization']),
                output_shape=output['shape'].tolist(),output_quantization=list(output['quantization']),
                operators=sorted({v['op_name'] for v in it._get_ops_details()}),evaluated_export_inputs=len(eval_ids))
            np.save(folder/(kind+'_features.npy'),vectors);np.savez_compressed(folder/(kind+'_gallery.npz'),gallery=gallery,threshold=np.float32(threshold))
            dump(folder/(kind+'_RESULT.json'),result);results[kind]=result
            print(json.dumps({'seed':seed,'kind':kind,'known_acceptance':result['known_correct_acceptance'],'unknown_false':result['unknown_false'],
                'elapsed_s':time.monotonic()-begun}),flush=True)
        dump(folder/'COMPLETE.json',{'seed':seed,'actual_gradient_training':True,'epochs':len(curves),'updates':len(curves)*10,
            'overfit_updates_separate':150,'trainable_parameters':int(sum(v.size for v in selected.values())),
            'results':results,'elapsed_s':time.monotonic()-begun,'future_final_used':False})

if __name__=='__main__':main()
