"""Deterministic64px spatial reduction of the pinned pretrained face encoder.

Changes the input and global-depthwise pooling kernel, not just zeroing weights.
No gradient training is implied; validate independently before recommendation.
"""
import os,json,hashlib
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2,numpy as np,tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];STUDY=ROOT/'work/project-research-v5'

def main():
    out=STUDY/'runs/spatial_encoder64';out.mkdir(parents=True,exist_ok=True)
    parent=STUDY/'models/pretrained/mobilefacenet_tf/MobileFaceNet_9925_9680.pb'
    g=tf.compat.v1.GraphDef();g.ParseFromString(parent.read_bytes());changes=[]
    for n in g.node:
        if '_output_shapes' in n.attr:del n.attr['_output_shapes']
        if n.name=='img_inputs':
            n.attr['shape'].shape.dim[0].size=1;n.attr['shape'].shape.dim[1].size=64;n.attr['shape'].shape.dim[2].size=64
            changes.append({'node':n.name,'input_shape':[1,64,64,3]})
        if n.op=='Const' and 'value' in n.attr:
            v=tf.make_ndarray(n.attr['value'].tensor)
            if v.shape==(7,7,512,1):
                new=np.stack([cv2.resize(v[:,:,i,0],(4,4),interpolation=cv2.INTER_LINEAR) for i in range(512)],axis=-1)[:,:,:,None]*(49/16)
                n.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(new.astype(np.float32)))
                changes.append({'node':n.name,'parent_shape':[7,7,512,1],'new_shape':[4,4,512,1],
                    'method':'bilinear spatial interpolation, multiplied49/16 to preserve constant-response scale approximately'})
    if len(changes)!=2:raise ValueError('Unexpected graph surgery targets')
    graph=out/'spatial64.pb';graph.write_bytes(g.SerializeToString())
    index=json.loads((STUDY/'runs/host_reference/nonfinal_index.json').read_text());rows=index['rows']
    if any('future_final' in r['role'] for r in rows):raise ValueError('Final access')
    roles=np.array([r['role'] for r in rows]);labels=np.array([int(r['label'][1:])-1 if r['known'] else -1 for r in rows])
    with np.load(STUDY/'runs/face_encoder/inputs_nonfinal.npz') as archive:
        cache={'aligned':archive['aligned'],'box':archive['box']}
    detected=np.load(STUDY/'runs/host_reference/nonfinal_features.npz')['detected']
    train=np.flatnonzero((roles=='train') & detected);rep=np.concatenate([train[labels[train]==k][:24] for k in range(5)])
    protocol={'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'derived_graph_sha256':hashlib.sha256(graph.read_bytes()).hexdigest(),
        'changes':changes,'new_gradient_trainings':0,'dtype_contract':'RGB64 (pixel-127.5)/128',
        'representatives':'120 balanced known train-only aligned images resized64','future_final_used':False,
        'status':'resource/quality hypothesis; not equivalent to original checkpoint or qualified camera identity model'}
    (out/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for kind in ['float','int8']:
        p=out/(kind+'.tflite')
        if p.exists():continue
        conv=tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(str(graph),['img_inputs'],['MobileFaceNet/Logits/SpatialSqueeze'],input_shapes={'img_inputs':[1,64,64,3]})
        if kind=='int8':
            conv.optimizations=[tf.lite.Optimize.DEFAULT]
            def gen():
                for i in rep:yield [((cv2.resize(cache['aligned'][i],(64,64)).astype(np.float32)-127.5)/128)[None]]
            conv.representative_dataset=gen;conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type=tf.int8;conv.inference_output_type=tf.int8
        p.write_bytes(conv.convert());print(json.dumps({'export':kind,'bytes':p.stat().st_size}),flush=True)
    allresults=[]
    for kind in ['float','int8']:
        p=out/(kind+'.tflite');it=tf.lite.Interpreter(model_path=str(p),num_threads=2);it.allocate_tensors();inp=it.get_input_details()[0];output=it.get_output_details()[0]
        metadata={'input_shape':inp['shape'].tolist(),'input_quantization':list(inp['quantization']),
            'output_shape':output['shape'].tolist(),'output_quantization':list(output['quantization']),
            'operators':sorted({v['op_name'] for v in it._get_ops_details()}),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
        (out/(kind+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        for transform in ['aligned_rgb','box_rgb']:
            path=out/(kind+'_'+transform+'.npy')
            if path.exists():features=np.load(path)
            else:
                features=np.zeros((len(rows),128),np.float32)
                for i in np.flatnonzero(detected):
                    pix=cv2.resize(cache['aligned' if transform=='aligned_rgb' else 'box'][i],(64,64));x=(pix.astype(np.float32)-127.5)/128
                    if kind=='int8':scale,zp=inp['quantization'];x=np.clip(np.rint(x/scale)+zp,-128,127).astype(np.int8)
                    it.set_tensor(inp['index'],x[None]);it.invoke();v=it.get_tensor(output['index'])[0]
                    if kind=='int8':scale,zp=output['quantization'];v=(v.astype(np.float32)-zp)*scale
                    if not np.isfinite(v).all():raise ValueError('Nonfinite embedding')
                    features[i]=v/max(float(np.linalg.norm(v)),1e-8)
                np.save(path,features)
            for seed in range(53001,53011):
                rng=np.random.default_rng(seed);gallery=[]
                for k in range(5):
                    ids=train[labels[train]==k];g=features[rng.choice(ids,min(10,len(ids)),replace=False)].mean(0);gallery.append(g/max(float(np.linalg.norm(g)),1e-8))
                gallery=np.array(gallery);scores=features@gallery.T;conf=scores.max(1);pred=scores.argmax(1)
                threshold=float(np.nextafter(conf[(roles=='unknown_calibration') & detected].max(),np.float32(np.inf)));accept=detected & (conf>=threshold)
                known=roles=='development';unknown=roles=='unknown_development'
                allresults.append({'kind':kind,'transform':transform,'gallery_sampling_seed':seed,
                    'known_correct_acceptance':float((accept & (pred==labels))[known].mean()),'known_coverage':float(accept[known].mean()),
                    'known_top1_full':float((detected & (pred==labels))[known].mean()),'known_n':int(known.sum()),
                    'unknown_false':int((accept & unknown).sum()),'unknown_n':int(unknown.sum()),'threshold':threshold,
                    'training_roots':0,'physical_qualified':False})
                np.savez_compressed(out/('%s_%s_gallery_%d.npz'%(kind,transform,seed)),gallery=gallery,threshold=np.float32(threshold))
            (out/'ROOT_RESULTS.json').write_text(json.dumps(allresults,indent=2,allow_nan=False)+'\n')
            subset=[r for r in allresults if r['kind']==kind and r['transform']==transform]
            print(json.dumps({'kind':kind,'transform':transform,'known_accept_mean':float(np.mean([r['known_correct_acceptance'] for r in subset])),
                'unknown_false_mean':float(np.mean([r['unknown_false'] for r in subset]))}),flush=True)

if __name__=='__main__':main()
