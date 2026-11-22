"""Export a synthetic input and host reference output for an immutable v3 embedding model."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def run(workspace: Path) -> dict:
    import numpy as np
    import tensorflow as tf
    source=(workspace/"work/research-complete-v3/project/work/research-next/model-system/runs/training-v3/custom_base/seed-20261022/exports/embedding_full_int8.tflite").resolve()
    source_manifest=(workspace/"work/research-complete-v3/project/work/research-next/model-system/runs/training-v3/custom_base/seed-20261022/run_manifest.json").resolve()
    evaluation=(workspace/"work/research-complete-v3/project/work/research-next/model-system/runs/training-v3/custom_base/seed-20261022/exports/development-export-evaluation.json").resolve()
    if not all(p.is_file() for p in (source,source_manifest,evaluation)):
        raise FileNotFoundError("required immutable v3 model/receipts are missing")
    out_dir=workspace/"work/project-finish-v4/models/host_reference"
    out_dir.mkdir(parents=True,exist_ok=True)
    model=out_dir/"v3_custom_base_seed-20261022_embedding_full_int8.tflite"
    shutil.copy2(source,model)
    pixels=np.fromiter(((index*37+11)%256 for index in range(96*96)),dtype=np.uint8,count=96*96).reshape((1,96,96,1))
    pixels_path=out_dir/"synthetic_gray96_rowmajor_u8.bin"
    pixels_path.write_bytes(pixels.tobytes())
    interpreter=tf.lite.Interpreter(model_path=str(model),num_threads=1)
    interpreter.allocate_tensors()
    input_detail=interpreter.get_input_details()[0]
    input_dtype=input_detail["dtype"]
    normalized=pixels.astype(np.float32)/255.0
    if np.issubdtype(input_dtype,np.floating):
        tensor=normalized.astype(input_dtype)
        conversion="gray8 / 255 -> float input"
    else:
        scale,zero=input_detail["quantization"]
        if not scale:
            raise ValueError("integer model input has no quantization scale")
        q=np.rint(normalized/scale+zero)
        limits=np.iinfo(input_dtype)
        tensor=np.clip(q,limits.min,limits.max).astype(input_dtype)
        conversion="gray8 / 255, then round(real/scale + zero_point) and clip"
    interpreter.set_tensor(input_detail["index"],tensor)
    interpreter.invoke()
    output_detail=interpreter.get_output_details()[0]
    output_raw=interpreter.get_tensor(output_detail["index"])
    if output_raw.size != 64:
        raise ValueError("expected a 64-dimensional v3 embedding output; got %s"%list(output_raw.shape))
    out_scale,out_zero=output_detail["quantization"]
    if np.issubdtype(output_raw.dtype,np.integer) and out_scale:
        output_values=((output_raw.astype(np.float32)-out_zero)*out_scale).reshape(-1).tolist()
        raw_values=output_raw.reshape(-1).astype(int).tolist()
    else:
        output_values=output_raw.reshape(-1).astype(float).tolist()
        raw_values=output_raw.reshape(-1).tolist()
    output_bytes=output_raw.tobytes()
    reference={
      "schema":"host-reference-fixture-1","created_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
      "purpose":"synthetic 96x96 input and host output for pipeline byte/parity checks only",
      "data_role":"synthetic_fixture","model_type":"embedding","embedding_dimension":64,
      "model_path":"work/project-finish-v4/models/host_reference/"+model.name,
      "model_sha256":sha256(model),"source_v3_model_path":source.relative_to(workspace).as_posix(),
      "source_v3_model_sha256":sha256(source),"source_run_manifest_sha256":sha256(source_manifest),
      "source_export_evaluation_sha256":sha256(evaluation),"source_protocol_id":"model-system-2026-10-23-v3",
      "source_root_seed":20261022,"input":{"width":96,"height":96,"pixel_format":"gray8",
        "layout":"row_major_interleaved","color_order":"GRAY","normalization":conversion,
        "bytes_path":"work/project-finish-v4/models/host_reference/"+pixels_path.name,
        "bytes_sha256":sha256(pixels_path),"bytes":pixels_path.stat().st_size},
      "input_tensor":{"shape":[int(x) for x in input_detail["shape"]],"dtype":str(input_dtype),
        "quantization_scale":float(input_detail["quantization"][0]),"zero_point":int(input_detail["quantization"][1])},
      "output_tensor":{"shape":[int(x) for x in output_detail["shape"]],"dtype":str(output_raw.dtype),
        "quantization_scale":float(out_scale),"zero_point":int(out_zero)},
      "output_raw_values":raw_values,"output_dequantized_values":output_values,
      "output_bytes_sha256":hashlib.sha256(output_bytes).hexdigest(),
      "warning":"This historical public-data embedding is not the five-person classifier/gallery, not a selected deployment model, and not a Nicla result."
    }
    reference_path=out_dir/"v3_embedding_reference_fixture.json"
    reference_path.write_text(json.dumps(reference,indent=2)+"\n",encoding="utf-8")
    return {"model_bytes":model.stat().st_size,"model_sha256":reference["model_sha256"],
            "input_sha256":reference["input"]["bytes_sha256"],"output_sha256":reference["output_bytes_sha256"],
            "output_dimension":len(output_values),"reference_path":str(reference_path),
            "tensorflow":tf.__version__}


if __name__=="__main__":
    root=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[2]
    print(json.dumps(run(root),indent=2))
