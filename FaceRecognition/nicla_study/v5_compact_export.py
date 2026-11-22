"""Measured compact face-encoder export with guarded native PReLU operators."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2,numpy as np,tensorflow as tf
from .v5_flatbuffer import fuse
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,choices=[48,56,64],required=True);a=p.parse_args()
    out=S/'runs'/('operator_fused_encoder'+str(a.size));out.mkdir(exist_ok=True)
    parent=(S/'models/pretrained/mobilefacenet_tf/MobileFaceNet_9925_9680.pb') if a.size==48 else S/'runs/spatial_encoder64/spatial64.pb'
    g=tf.compat.v1.GraphDef();g.ParseFromString(parent.read_bytes())
    for n in g.node:
        if '_output_shapes' in n.attr:del n.attr['_output_shapes']
        if n.name=='img_inputs':n.attr['shape'].shape.dim[0].size=1;n.attr['shape'].shape.dim[1].size=a.size;n.attr['shape'].shape.dim[2].size=a.size
        if a.size==48 and n.op=='Const' and 'value' in n.attr:
            v=tf.make_ndarray(n.attr['value'].tensor)
            if v.shape==(7,7,512,1):
                new=np.stack([cv2.resize(v[:,:,i,0],(3,3)) for i in range(512)],axis=-1)[:,:,:,None]*(49/9)
                n.attr['value'].tensor.CopyFrom(tf.make_tensor_proto(new.astype(np.float32)))
    pb=out/'parent.pb';pb.write_bytes(g.SerializeToString())
    idx=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text());rows=idx['rows']
    with np.load(S/'runs/face_encoder/inputs_nonfinal.npz') as ar:box=ar['box']
    detected=np.load(S/'runs/host_reference/nonfinal_features.npz')['detected']
    rep=[]
    for label in ['K01','K02','K03','K04','K05']:rep.extend([i for i,r in enumerate(rows) if r['role']=='train' and r['label']==label and detected[i]][:24])
    for kind in ['float','int8']:
        if a.size==64:raw=(S/'runs/spatial_encoder64'/(kind+'.tflite')).read_bytes()
        else:
            conv=tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(str(pb),['img_inputs'],['MobileFaceNet/Logits/SpatialSqueeze'],input_shapes={'img_inputs':[1,a.size,a.size,3]})
            if kind=='int8':
                conv.optimizations=[tf.lite.Optimize.DEFAULT]
                def reps():
                    for i in rep:yield [((cv2.resize(box[i],(a.size,a.size)).astype(np.float32)-127.5)/128)[None]]
                conv.representative_dataset=reps;conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                conv.inference_input_type=tf.int8;conv.inference_output_type=tf.int8
            raw=conv.convert()
        (out/(kind+'_expanded.tflite')).write_bytes(raw)
        payload,report=fuse(raw);(out/(kind+'.tflite')).write_bytes(payload)
        it=tf.lite.Interpreter(model_content=payload,num_threads=2);it.allocate_tensors()
        operators=[v['op_name'] for v in it._get_ops_details()]
        report.update(operators=sorted(set(operators)),size=a.size,input_quantization=list(it.get_input_details()[0]['quantization']),
            output_quantization=list(it.get_output_details()[0]['quantization']))
        (out/(kind+'_metadata.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='changes'}),flush=True)
        if kind=='float':
            old=tf.lite.Interpreter(model_content=raw,num_threads=2);old.allocate_tensors();results=[]
            for seed in range(53001,53011):
                rng=np.random.default_rng(seed);maximum=0.;passed=True;cosine_min=1.
                for _ in range(3):
                    x=rng.uniform(-.99609375,.99609375,(1,a.size,a.size,3)).astype(np.float32)
                    ys=[]
                    for interpreter in [old,it]:
                        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],x);interpreter.invoke();ys.append(interpreter.get_tensor(interpreter.get_output_details()[0]['index']))
                    maximum=max(maximum,float(np.max(np.abs(ys[0]-ys[1]))));passed=passed and bool(np.allclose(ys[0],ys[1],rtol=1e-5,atol=1e-4))
                    cosine_min=min(cosine_min,float(np.sum(ys[0]*ys[1])/max(float(np.linalg.norm(ys[0])*np.linalg.norm(ys[1])),1e-12)))
                results.append({'seed':seed,'fixtures':3,'maximum_abs_error':maximum,'passed':passed,'minimum_cosine':cosine_min})
            (out/'FLOAT_FUSION_PARITY.json').write_text(json.dumps({'prospective_tolerance':{'atol':1e-4,'rtol':1e-5},'blocks':results,
                'status':'verified' if all(r['passed'] for r in results) else 'failed','physical_evidence':False},indent=2)+'\n')
            if not all(r['passed'] for r in results):raise ValueError('Float fusion changes function beyond tolerance')

if __name__=='__main__':main()
