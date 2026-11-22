"""Rewrite algebraically expanded PReLU into the TFLite-recognized pattern.

Original: relu(x)+alpha*(x-abs(x))*.5.
Equivalent: relu(x)-alpha*relu(-x). We preserve every trained alpha. This is
operator/activation-memory optimization, not removing filters or retraining.
"""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2,numpy as np,tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'

def fuse_prelu(graph):
    g=tf.compat.v1.GraphDef();g.CopyFrom(graph);nodes={n.name:n for n in g.node};changes=[]
    for node in list(g.node):
        if node.op not in ['Add','AddV2'] or len(node.input)!=2:continue
        left=nodes[node.input[0].split(':')[0]]
        prefix=node.name.rsplit('/',1)[0]
        if left.op!='Relu':continue
        source=left.input[0]
        # tf.variable_scope and tf.name_scope have different prefixes here.
        # Follow the exact algebraic inputs rather than guessing alpha's path.
        right=nodes[node.input[1].split(':')[0]]
        if right.op!='Mul':continue
        inner=nodes[right.input[0].split(':')[0]];half=nodes[right.input[1].split(':')[0]]
        if inner.op!='Mul' or half.op!='Const' or float(tf.make_ndarray(half.attr['value'].tensor))!=.5:continue
        alpha=nodes[inner.input[0].split(':')[0]]
        while alpha.op=='Identity':alpha=nodes[alpha.input[0].split(':')[0]]
        sub=nodes[inner.input[1].split(':')[0]]
        if alpha.op!='Const' or not alpha.name.endswith('/prelu_alphas') or sub.op!='Sub':continue
        absolute=nodes[sub.input[1].split(':')[0]]
        if sub.input[0]!=source or absolute.op!='Abs' or absolute.input[0]!=source:raise ValueError('PReLU identity not established')
        av=tf.make_ndarray(alpha.attr['value'].tensor)
        def op(name,kind,inputs):
            n=g.node.add();n.name=prefix+'/v5_'+name;n.op=kind;n.input.extend(inputs);n.attr['T'].type=tf.float32.as_datatype_enum;return n.name
        neg=op('negative_input','Neg',[source]);relu=op('negative_relu','Relu',[neg])
        cn=g.node.add();cn.name=prefix+'/v5_negative_alpha';cn.op='Const';cn.attr['dtype'].type=tf.float32.as_datatype_enum
        cn.attr['value'].tensor.CopyFrom(tf.make_tensor_proto((-av).astype(np.float32)))
        mul=op('negative_branch','Mul',[cn.name,relu]);node.input[1]=mul
        changes.append({'output':node.name,'alpha':alpha.name,'channels':int(av.size)})
    if len(changes)<20:raise ValueError('Too few matched PReLU structures')
    return g,changes

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--size',type=int,choices=[56,64,112],required=True);a=p.parse_args()
    out=S/'runs'/('fused_encoder'+str(a.size));out.mkdir(exist_ok=True)
    parent=(S/'models/pretrained/mobilefacenet_tf/MobileFaceNet_9925_9680.pb') if a.size==112 else S/'runs/spatial_encoder64/spatial64.pb'
    original=tf.compat.v1.GraphDef();original.ParseFromString(parent.read_bytes())
    for n in original.node:
        if n.name=='img_inputs':
            n.attr['shape'].shape.dim[0].size=1;n.attr['shape'].shape.dim[1].size=a.size;n.attr['shape'].shape.dim[2].size=a.size
    graph,changes=fuse_prelu(original);pb=out/'fused.pb';pb.write_bytes(graph.SerializeToString())
    index=json.loads((S/'runs/host_reference/nonfinal_index.json').read_text());rows=index['rows']
    detected=np.load(S/'runs/host_reference/nonfinal_features.npz')['detected']
    with np.load(S/'runs/face_encoder/inputs_nonfinal.npz') as ar:box=ar['box']
    train=np.array([i for i,r in enumerate(rows) if r['role']=='train' and detected[i]])
    chosen=[]
    for label in ['K01','K02','K03','K04','K05']:chosen.extend([i for i in train if rows[i]['label']==label][:24])
    for kind in ['float','int8']:
        path=out/(kind+'.tflite')
        conv=tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(str(pb),['img_inputs'],['MobileFaceNet/Logits/SpatialSqueeze'],input_shapes={'img_inputs':[1,a.size,a.size,3]})
        if kind=='int8':
            conv.optimizations=[tf.lite.Optimize.DEFAULT]
            def reps():
                for i in chosen:yield [((cv2.resize(box[i],(a.size,a.size)).astype(np.float32)-127.5)/128)[None]]
            conv.representative_dataset=reps;conv.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            conv.inference_input_type=tf.int8;conv.inference_output_type=tf.int8
        payload=conv.convert();path.write_bytes(payload)
        it=tf.lite.Interpreter(model_content=payload,num_threads=2);it.allocate_tensors()
        operators=[v['op_name'] for v in it._get_ops_details()]
        metadata={'kind':kind,'size':a.size,'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
            'operators':sorted(set(operators)),'prelu_operators':operators.count('PRELU'),'rewritten_structures':len(changes),
            'input_quantization':list(it.get_input_details()[0]['quantization']),
            'output_quantization':list(it.get_output_details()[0]['quantization']),
            'parent_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'changes':changes,
            'status':'must verify function/parity and actual MCU arena; no automatic recognition qualification','future_final_used':False}
        (out/(kind+'_metadata.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({k:v for k,v in metadata.items() if k!='changes'}),flush=True)
        if metadata['prelu_operators']!=len(changes):raise ValueError('Converter did not fuse all intended PReLU operations')

if __name__=='__main__':main()
