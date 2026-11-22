"""Tiny operator controls distinguish PRELU support/shape from arena failure."""
import os,json
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import numpy as np,tensorflow as tf,flatbuffers
from tensorflow.lite.python import schema_py_generated as schema
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'
def main():
    out=S/'staging/prelu_controls';out.mkdir(exist_ok=True)
    x=tf.keras.Input((4,4,3));y=tf.keras.layers.PReLU(alpha_initializer=tf.keras.initializers.Constant(.25),shared_axes=[1,2])(x);model=tf.keras.Model(x,y)
    converter=tf.lite.TFLiteConverter.from_keras_model(model);converter.optimizations=[tf.lite.Optimize.DEFAULT]
    rng=np.random.default_rng(55101)
    def rep():
        for i in range(20):yield [rng.uniform(-1,1,(1,4,4,3)).astype(np.float32)]
    converter.representative_dataset=rep;converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8];converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    data=converter.convert();(out/'prelu_3d.tflite').write_bytes(data)
    m=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(data,0));g=m.subgraphs[0];op=next(o for o in g.operators if m.operatorCodes[o.opcodeIndex].builtinCode==schema.BuiltinOperator.PRELU)
    a=g.tensors[int(op.inputs[1])];original_shape=a.shape.tolist();a.shape=np.array([3],np.int32)
    if a.shapeSignature is not None:a.shapeSignature=np.array([3],np.int32)
    b=flatbuffers.Builder(0);v=m.Pack(b);b.Finish(v,file_identifier=b'TFL3');(out/'prelu_1d.tflite').write_bytes(bytes(b.Output()))
    results=[]
    for path in out.glob('*.tflite'):
        it=tf.lite.Interpreter(model_path=str(path),experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);it.allocate_tensors();inp=it.get_input_details()[0];output=it.get_output_details()[0]
        for seed in range(10):
            raw=((np.arange(48)*13+seed*23)%256).astype(np.uint8).view(np.int8).reshape(1,4,4,3);it.set_tensor(inp['index'],raw);it.invoke();expected=it.get_tensor(output['index']);results.append({'model':path.name,'seed':seed,'input_hex':raw.tobytes().hex(),'output_hex':expected.tobytes().hex()})
    (S/'reports/M5/PRELU_CONTROL_HOST.json').write_text(json.dumps({'control':'fixed arithmetic, no gradient training','keras_alpha_shape':original_shape,'cases':results},indent=2)+'\n');print('Exported native PRELU arithmetic controls')
if __name__=='__main__':main()
