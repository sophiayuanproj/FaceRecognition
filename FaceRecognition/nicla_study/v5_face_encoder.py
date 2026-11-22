"""Face-pretrained frozen-graph route, explicit alignment and PTQ diagnostics.

Uses the upstream sirius-ai/MobileFaceNet_TF frozen checkpoint. This is a fixed
pretrained network, not ten independent new gradient-trained encoders.
"""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2
import numpy as np
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[2]
STUDY=ROOT/'work/project-research-v5'

def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
    cv2.setNumThreads(2)
    out=STUDY/'runs/face_encoder';out.mkdir(parents=True,exist_ok=True)
    source=STUDY/'models/pretrained/mobilefacenet_tf/MobileFaceNet_9925_9680.pb'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='fb046e5f723a70020962c6772a08c3c915a443ca19aaade732c2b84eea613f09':raise ValueError('Source changed')
    index=json.loads((STUDY/'runs/host_reference/nonfinal_index.json').read_text());rows=index['rows']
    if any('future_final' in r['role'] for r in rows):raise ValueError('Final prohibited')
    old=np.load(STUDY/'runs/host_reference/nonfinal_features.npz');detected=old['detected'];rgb96=old['rgb96']
    roles=np.array([r['role'] for r in rows]);labels=np.array([int(r['label'][1:])-1 if r['known'] else -1 for r in rows])
    cache=out/'inputs_nonfinal.npz'
    if cache.exists():data=np.load(cache);aligned=data['aligned'];box=data['box']
    else:
        models=ROOT/'work/project-finish-v4/models/pretrained/sface'
        det=cv2.FaceDetectorYN.create(str(models/'face_detection_yunet_2023mar.onnx'),'',(250,250),.85,.3,5000)
        sf=cv2.FaceRecognizerSF.create(str(models/'face_recognition_sface_2021dec.onnx'),'')
        aligned=np.zeros((len(rows),112,112,3),np.uint8);box=np.zeros_like(aligned)
        for i,row in enumerate(rows):
            if not detected[i]:continue
            image=cv2.imread(str(ROOT/row['path']));h,w=image.shape[:2];det.setInputSize((w,h));_,faces=det.detect(image)
            if faces is None or len(faces)!=1:raise ValueError('Reference detections changed')
            aligned[i]=cv2.cvtColor(sf.alignCrop(image,faces[0]),cv2.COLOR_BGR2RGB)
            box[i]=cv2.resize(rgb96[i],(112,112))
        np.savez_compressed(cache,aligned=aligned,box=box)
    # Upstream data_process.py: RGB, (pixel-127.5)*0.0078125.
    train=np.flatnonzero((roles=='train') & detected)
    rep=np.concatenate([train[labels[train]==k][:24] for k in range(5)])
    protocol={'source_repository':'https://github.com/sirius-ai/MobileFaceNet_TF',
        'source_pb_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'input':'img_inputs:0, NHWC112 RGB, (pixel-127.5)/128',
        'output':'MobileFaceNet/Logits/SpatialSqueeze:0, 128 raw; host L2 after inference',
        'alignment':'SFace five-landmark alignCrop uses ArcFace112 template; box route separately labeled',
        'pretraining_exposure':'Exact source training corpus is not bound to this checkpoint by metadata; LFW exposure unresolved.',
        'source_license':'Apache2 source; training-image rights remain separate',
        'int8_representatives':'120 known train-only aligned faces, balanced24 per class',
        'fixed_network':True,'gallery_sampling_seeds':list(range(53001,53011)),
        'future_final_used':False,'source_camera':'LFW source, not Nicla recapture'}
    dump(out/'PROTOCOL.json',protocol)
    for kind in ['float','int8']:
        path=out/(kind+'.tflite')
        if path.exists():continue
        conv=tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(str(source),['img_inputs'],
            ['MobileFaceNet/Logits/SpatialSqueeze'],input_shapes={'img_inputs':[1,112,112,3]})
        if kind=='int8':
            conv.optimizations=[tf.lite.Optimize.DEFAULT]
            def representatives():
                for i in rep:yield [(aligned[i:i+1].astype(np.float32)-127.5)/128]
            conv.representative_dataset=representatives
            conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type=tf.int8;conv.inference_output_type=tf.int8
        path.write_bytes(conv.convert())
        print(json.dumps({'export':kind,'bytes':path.stat().st_size}),flush=True)
    results=[]
    for kind in ['float','int8']:
        path=out/(kind+'.tflite');it=tf.lite.Interpreter(model_path=str(path),num_threads=2);it.allocate_tensors()
        inp=it.get_input_details()[0];output=it.get_output_details()[0]
        metadata={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'input_shape':inp['shape'].tolist(),'input_quantization':list(inp['quantization']),
            'output_shape':output['shape'].tolist(),'output_quantization':list(output['quantization']),
            'operators':sorted({r['op_name'] for r in it._get_ops_details()})}
        dump(out/(kind+'_metadata.json'),metadata)
        for transform in ['aligned_rgb','box_rgb','box_gray']:
            features=np.zeros((len(rows),128),np.float32)
            cache_feat=out/(kind+'_'+transform+'.npy')
            if cache_feat.exists():features=np.load(cache_feat)
            else:
                for i in np.flatnonzero(detected):
                    pix=aligned[i] if transform=='aligned_rgb' else box[i]
                    if transform=='box_gray':pix=np.repeat(cv2.cvtColor(pix,cv2.COLOR_RGB2GRAY)[:,:,None],3,axis=2)
                    x=(pix.astype(np.float32)-127.5)/128
                    if kind=='int8':
                        scale,zp=inp['quantization'];x=np.clip(np.rint(x/scale)+zp,-128,127).astype(np.int8)
                    it.set_tensor(inp['index'],x[None]);it.invoke();v=it.get_tensor(output['index'])[0]
                    if kind=='int8':
                        scale,zp=output['quantization'];v=(v.astype(np.float32)-zp)*scale
                    if not np.isfinite(v).all():raise ValueError('Nonfinite embedding')
                    features[i]=v/max(float(np.linalg.norm(v)),1e-8)
                np.save(cache_feat,features)
            for seed in range(53001,53011):
                rng=np.random.default_rng(seed);gallery=[]
                for k in range(5):
                    ids=np.flatnonzero((roles=='train') & (labels==k) & detected)
                    chosen=rng.choice(ids,min(10,len(ids)),replace=False);g=features[chosen].mean(0);gallery.append(g/max(float(np.linalg.norm(g)),1e-8))
                gallery=np.array(gallery);scores=features@gallery.T;confidence=scores.max(1);pred=scores.argmax(1)
                threshold=float(np.nextafter(confidence[(roles=='unknown_calibration') & detected].max(),np.float32(np.inf)))
                accept=detected & (confidence>=threshold);known=roles=='development';unknown=roles=='unknown_development'
                row={'kind':kind,'transform':transform,'gallery_sampling_root':seed,'n_known':int(known.sum()),
                    'known_correct_acceptance':float((accept & (pred==labels))[known].mean()),
                    'known_raw_top1_full':float((detected & (pred==labels))[known].mean()),
                    'coverage':float(accept[known].mean()),'unknown_false':int((accept & unknown).sum()),'unknown_n':int(unknown.sum()),
                    'threshold':threshold,'unknown_identity_groups':30,'board_qualified':False}
                results.append(row)
                np.savez_compressed(out/('%s_%s_gallery_%d.npz'%(kind,transform,seed)),gallery=gallery,threshold=np.float32(threshold))
            dump(out/'ROOT_RESULTS.json',results)
            subset=[r for r in results if r['kind']==kind and r['transform']==transform]
            print(json.dumps({'kind':kind,'transform':transform,'roots':len(subset),
                'known_mean':float(np.mean([r['known_correct_acceptance'] for r in subset])),
                'unknown_mean':float(np.mean([r['unknown_false'] for r in subset]))}),flush=True)


if __name__=='__main__':main()
