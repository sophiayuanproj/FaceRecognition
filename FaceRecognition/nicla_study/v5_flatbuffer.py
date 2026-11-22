"""Guarded physical operator fusion for expanded upstream PReLU graphs.

Float semantics are algebraically equivalent. INT8 rounding changes are a new
candidate and MUST be recalibrated/evaluated; they are never called bit parity.
"""
import hashlib,json
from pathlib import Path
import flatbuffers,numpy as np
from tensorflow.lite.python import schema_py_generated as schema

def fuse(payload):
    m=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(payload,0))
    if len(m.subgraphs)!=1:raise ValueError('One subgraph only')
    sg=m.subgraphs[0];initial_tensor_count=len(sg.tensors);ops=sg.operators;producer={int(t):i for i,o in enumerate(ops) for t in o.outputs}
    consumers={}
    for i,o in enumerate(ops):
        for t in o.inputs:consumers.setdefault(int(t),[]).append(i)
    def code(i):return m.operatorCodes[ops[i].opcodeIndex].builtinCode
    def scalar(t):
        tensor=sg.tensors[t];buffer=m.buffers[tensor.buffer]
        if buffer.data is None or len(buffer.data)==0:return None
        if tensor.type==schema.TensorType.FLOAT32:arr=np.frombuffer(buffer.data.tobytes(),np.float32)
        elif tensor.type==schema.TensorType.INT8:
            arr=np.frombuffer(buffer.data.tobytes(),np.int8).astype(np.float32)
            q=tensor.quantization
            if q is None or len(q.scale)!=1:return None
            arr=(arr-q.zeroPoint[0])*q.scale[0]
        else:return None
        return float(arr[0]) if arr.size==1 else None
    remove=set();replace={};changes=[]
    B=schema.BuiltinOperator
    for i,op in enumerate(ops):
        if code(i)!=B.ADD or len(op.inputs)!=2:continue
        found=None
        for pos,neg in [(int(op.inputs[0]),int(op.inputs[1])),(int(op.inputs[1]),int(op.inputs[0]))]:
            pi,ni=producer.get(pos),producer.get(neg)
            if pi is None or ni is None or code(pi)!=B.RELU or code(ni)!=B.MUL:continue
            halfop=ops[ni];half=None;inner_tensor=None
            for a,b in [(int(halfop.inputs[0]),int(halfop.inputs[1])),(int(halfop.inputs[1]),int(halfop.inputs[0]))]:
                val=scalar(b)
                if val is not None and abs(val-.5)<1e-6:inner_tensor=a;half=b;break
            ai=producer.get(inner_tensor)
            if half is None or ai is None or code(ai)!=B.MUL:continue
            alphaop=ops[ai];subi=None;alpha=None
            for a,b in [(int(alphaop.inputs[0]),int(alphaop.inputs[1])),(int(alphaop.inputs[1]),int(alphaop.inputs[0]))]:
                si=producer.get(a)
                if si is not None and code(si)==B.SUB and b not in producer:subi=si;alpha=b;break
            if subi is None:continue
            sub=ops[subi];x=int(ops[pi].inputs[0]);absi=producer.get(int(sub.inputs[1]))
            if int(sub.inputs[0])!=x or absi is None or code(absi)!=B.ABS or int(ops[absi].inputs[0])!=x:continue
            chain=[absi,pi,subi,ai,ni]
            expected={absi:subi,pi:i,subi:ai,ai:ni,ni:i}
            if any(consumers.get(int(ops[k].outputs[0]))!=[v] for k,v in expected.items()):raise ValueError('Shared intermediate would be destroyed')
            if any(k in remove for k in chain):raise ValueError('Overlapping fusion chain')
            found=(x,alpha,chain);break
        if found is None:continue
        x,alpha,chain=found;new=schema.OperatorT();new.inputs=np.array([x,alpha],np.int32);new.outputs=op.outputs.copy()
        new.opcodeIndex=len(m.operatorCodes);new.builtinOptionsType=schema.BuiltinOptions.NONE
        replace[i]=new;remove.update(chain);changes.append({'original_add_index':i,'input_tensor':x,'alpha_tensor':alpha,'removed_op_indices':chain})
    if len(changes)<20:raise ValueError('Insufficient exact expanded-PReLU matches')
    op_code=schema.OperatorCodeT();op_code.builtinCode=B.PRELU;op_code.deprecatedBuiltinCode=B.PRELU
    # This runtime's PRELU registration is version1, including INT8. Version2
    # is not registered in the pinned TF2.10 interpreter and was rejected.
    op_code.version=1
    m.operatorCodes.append(op_code)
    sg.operators=[replace.get(i,o) for i,o in enumerate(ops) if i not in remove]
    live=set(map(int,sg.inputs))|set(map(int,sg.outputs))
    for op in sg.operators:live.update(int(v) for v in list(op.inputs)+list(op.outputs) if int(v)>=0)
    ordered=sorted(live);mapping={old:new for new,old in enumerate(ordered)}
    for op in sg.operators:
        op.inputs=np.array([mapping[int(v)] if int(v)>=0 else -1 for v in op.inputs],np.int32)
        op.outputs=np.array([mapping[int(v)] for v in op.outputs],np.int32)
    sg.inputs=np.array([mapping[int(v)] for v in sg.inputs],np.int32);sg.outputs=np.array([mapping[int(v)] for v in sg.outputs],np.int32)
    sg.tensors=[sg.tensors[i] for i in ordered]
    if m.signatureDefs:
        for signature in m.signatureDefs:
            if signature.subgraphIndex!=0:raise ValueError('Unexpected signature graph')
            for item in list(signature.inputs)+list(signature.outputs):item.tensorIndex=mapping[item.tensorIndex]
    builder=flatbuffers.Builder(0);offset=m.Pack(builder);builder.Finish(offset,file_identifier=b'TFL3');output=bytes(builder.Output())
    report={'schema':'v5-guarded-flatbuffer-prelu-fusion-1','parent_sha256':hashlib.sha256(payload).hexdigest(),
        'derived_sha256':hashlib.sha256(output).hexdigest(),'parent_bytes':len(payload),'derived_bytes':len(output),
        'fused_prelu_count':len(changes),'removed_operator_count':len(remove),'dead_tensor_count':initial_tensor_count-len(ordered),
        'changes':changes,'integer_semantics':'new rounding path; must recalibrate and evaluate this exact candidate'}
    return output,report

def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    payload,report=fuse(a.input.read_bytes());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(payload)
    a.output.with_suffix('.fusion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='changes'}))

if __name__=='__main__':main()
