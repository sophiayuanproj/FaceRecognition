"""Exact-pixel optical development analysis; source-calibrated host reference."""
import os,ast,hashlib,json
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import cv2,numpy as np,tensorflow as tf
ROOT=Path(__file__).resolve().parents[2];S=ROOT/'work/project-research-v5'

def rgb565(raw):
    pixels=np.frombuffer(raw,dtype='<u2').reshape(240,240).astype(np.uint32)
    r=(pixels>>11)&31;g=(pixels>>5)&63;b=pixels&31
    return np.stack([(r<<3)|(r>>2),(g<<2)|(g>>4),(b<<3)|(b>>2)],axis=-1).astype(np.uint8)

def main():
    cv2.setNumThreads(2)
    session=ROOT/(S/'runs/optical/LATEST_SESSION.txt').read_text().strip();receipts=json.loads((session/'RECEIPTS.json').read_text())
    if len(receipts)!=100:raise ValueError('Expected completed100-frame study')
    out=session/'analysis';out.mkdir(exist_ok=True)
    ref=S/'runs/host_reference';index=json.loads((ref/'nonfinal_index.json').read_text());source_rows=index['rows']
    with np.load(ref/'nonfinal_features.npz') as ar:features=ar['features'];detected=ar['detected'];source_boxes=ar['boxes']
    gallery=[]
    for k in ['K01','K02','K03','K04','K05']:
        ids=[i for i,r in enumerate(source_rows) if r['role']=='train' and r['label']==k and detected[i]];v=features[ids].mean(0);gallery.append(v/np.linalg.norm(v))
    gallery=np.array(gallery);reference=[r for r in json.loads((ref/'ROOT_RESULTS.json').read_text()) if r['gallery_templates_per_class']=='all' and r['seed']==53001][0];threshold=reference['threshold']
    modeldir=ROOT/'work/project-finish-v4/models/pretrained/sface'
    detector=cv2.FaceDetectorYN.create(str(modeldir/'face_detection_yunet_2023mar.onnx'),'',(240,240),.85,.3,5000)
    encoder=cv2.FaceRecognizerSF.create(str(modeldir/'face_recognition_sface_2021dec.onnx'),'')
    fomo_model=S/'private/board-backup-initial/rom/fomo_face_detection.tflite';it=tf.lite.Interpreter(model_path=str(fomo_model),num_threads=2,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF);it.allocate_tensors();inp=it.get_input_details()[0];output=it.get_output_details()[0]
    records=[];embeddings=[];source_map={r['record_id']:i for i,r in enumerate(source_rows)}
    for entry in receipts:
        folder=ROOT/entry['directory'];meta=entry['metadata'];request=entry['request']
        for file in entry['files']:
            payload=(folder/file['name']).read_bytes()
            if len(payload)!=file['bytes'] or hashlib.sha256(payload).hexdigest()!=file['sha256']:raise ValueError('Capture hash mismatch')
        rgb=rgb565((folder/'camera_rgb565.bin').read_bytes());bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        _,faces=detector.detect(bgr);count=0 if faces is None else len(faces)
        row={'capture_id':request['capture_id'],'block':request['block'],'truth':request['expected_truth'],
            'source_id':request['source'].get('record_id'),'condition':request['display'],'camera_face_count_reference':count,
            'accepted':False,'predicted':None,'confidence':None,'face_box':None,'eye_separation_px':None,
            'gray_mean':float(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).mean()),'near_saturated_fraction':float((rgb.max(2)>=250).mean()),
            'camera_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}
        vector=np.zeros(128,np.float32)
        if count==1:
            face=faces[0];vector=np.asarray(encoder.feature(encoder.alignCrop(bgr,face)),np.float32).ravel();vector/=max(float(np.linalg.norm(vector)),1e-8)
            scores=vector@gallery.T;k=int(scores.argmax());row.update(predicted='K%02d'%(k+1),confidence=float(scores[k]),accepted=bool(scores[k]>=threshold),
                face_box=face[:4].astype(float).tolist(),eye_separation_px=float(np.linalg.norm(face[4:6]-face[6:8])))
        row['fomo_clusters']=sum(len(channel) for channel in ast.literal_eval(meta['fomo'])) if meta['fomo']!='()' else 0
        row['haar_boxes']=len(meta['haar_boxes_160'])
        raw=np.frombuffer((folder/'fomo_input_i8.bin').read_bytes(),np.int8).reshape(inp['shape']);it.set_tensor(inp['index'],raw);it.invoke();host=it.get_tensor(output['index']).ravel().astype(np.int16)
        board=np.frombuffer((folder/'fomo_output_i8.bin').read_bytes(),np.int8).astype(np.int16);delta=np.abs(host-board)
        row['fomo_kernel_max_lsb']=int(delta.max());row['fomo_kernel_differing_values']=int((delta!=0).sum());row['fomo_kernel_mae_lsb']=float(delta.mean())
        row['fomo_kernel_within_prospective3lsb']=bool(delta.max()<=3)
        row['fomo_predict_us']=meta['fomo_predict_us'];row['haar_predict_us']=meta['haar_predict_us'];row['capture_us']=meta['capture_us']
        row['fomo_multiple_cluster_ambiguity']=row['fomo_clusters']>1
        # Generated calibration and blank frames remain independent negatives.
        records.append(row);embeddings.append(vector)
        cv2.imwrite(str(out/(request['capture_id']+'.png')),bgr)
    known=[r for r in records if r['truth'] in ['K01','K02','K03','K04','K05']];unknown=[r for r in records if r['truth']=='OTHER'];blank=[r for r in records if r['truth']=='NO_FACE']
    per=[]
    for label in ['K01','K02','K03','K04','K05']:
        rs=[r for r in known if r['truth']==label];per.append({'label':label,'n':len(rs),'correct_accepted':sum(r['accepted'] and r['predicted']==label for r in rs),'accepted':sum(r['accepted'] for r in rs),'reference_detected_single':sum(r['camera_face_count_reference']==1 for r in rs)})
    summary={'schema':'v5-optical-development-analysis-2','host_quantized_backend':'TF2.10.1 BUILTIN_REF; optimized-backend disagreement retained separately','captured_frames':len(records),'physical_blocks':len({r['block'] for r in records}),
        'physical_environments':1,'known_presentations':len(known),'known_source_groups':len({r['source_id'] for r in known}),
        'unknown_presentations':len(unknown),'unknown_identity_groups':30,'source_calibrated_host_threshold':threshold,
        'known_correct_acceptance':sum(r['accepted'] and r['predicted']==r['truth'] for r in known)/len(known),
        'known_coverage':sum(r['accepted'] for r in known)/len(known),'known_wrong_accepts':sum(r['accepted'] and r['predicted']!=r['truth'] for r in known),
        'unknown_false_identifications':sum(r['accepted'] for r in unknown),'unknown_fpir':sum(r['accepted'] for r in unknown)/len(unknown),
        'zero_event_upper95_for30_unknown_groups':1-.05**(1/30),'per_person':per,
        'detector_positive_presentation_triggers':{name:sum(r[key]>0 for r in known+unknown) for name,key in [('fomo','fomo_clusters'),('haar','haar_boxes')]},
        'blank_false_triggers':{name:sum(r[key]>0 for r in blank) for name,key in [('fomo','fomo_clusters'),('haar','haar_boxes')]},
        'fomo_multiple_cluster_presentations':sum(r['fomo_multiple_cluster_ambiguity'] for r in known+unknown),
        'stage_latency_ms':{key:{'median':float(np.median([r[key]/1000 for r in records])),'p95':float(np.percentile([r[key]/1000 for r in records],95))} for key in ['capture_us','fomo_predict_us','haar_predict_us']},
        'fomo_exact_input_kernel':{'inputs':100,'max_lsb':max(r['fomo_kernel_max_lsb'] for r in records),'all_within3lsb':all(r['fomo_kernel_within_prospective3lsb'] for r in records),
            'model_sha256':hashlib.sha256(fomo_model.read_bytes()).hexdigest()},
        'scope':'Actual optical development; host SFace diagnostic, not MCU identification. FOMO/Haar run on MCU. No power/response-to-arrival claim.',
        'limits':['Same source images recur across blocks; sources/session are dependent.','Positive presentation trigger is not manually annotated bounding-box recall.','Some LFW portraits contain background faces; multi-face reference rejects them.','Stage times exclude fixture transmission; not full user-response latency.','Unknown sample supports at best9.5% zero-event identity upper bound, not1%.','Pretraining exposure unresolved.'],
        'future_final_used':False,'ready_for_final_build':False}
    (out/'RECORDS.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n');np.save(out/'host_reference_features.npy',np.array(embeddings))
    (out/'RESULT.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
