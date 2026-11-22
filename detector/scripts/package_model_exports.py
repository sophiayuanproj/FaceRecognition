"""Bundle deployable TFLite exports and audit metadata; omit training-only H5 weights."""
from __future__ import annotations
import argparse,zipfile
from pathlib import Path


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model-dir',type=Path,required=True);ap.add_argument('--stock-dir',type=Path,required=True);ap.add_argument('--audit-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 files=[]
 for arm in ('cheap_sepconv_16','mobilenetv2_alpha_0p1'):
  for root in sorted((a.model_dir/arm).glob('seed-*')):
   for name in ('detector_float.tflite','detector_full_int8.tflite','labels.txt','root_summary.json','history.json'):
    p=root/name
    if not p.is_file():raise FileNotFoundError(p)
    files.append((p,p.relative_to(a.model_dir)))
 for p in (a.model_dir/'training_method.json',a.model_dir/'ten_root_summary.json',a.stock_dir/'fomo_face_detection-stock.tflite',a.stock_dir/'labels.txt',a.stock_dir/'model_audit.json',a.audit_dir/'root_export_audit.json'):
  if p.is_file(): files.append((p,Path('metadata')/p.name if p.parent==a.audit_dir else p.relative_to(a.model_dir.parent)))
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  z.writestr('README.txt','Job C detector deployment exports. Includes float and full-INT8 TFLite artifacts, per-root labels/histories/summaries, fixed stock reference, ten-root training summary, training method, and a complete tensor/operator/hash audit. The .weights.h5 files are intentionally excluded: they are training checkpoints, not deployable models. Raw data is not included.\n')
  for src,rel in files:z.write(src,rel.as_posix())
 print(f'Wrote {a.out} with {len(files)} artifacts plus README.')
if __name__=='__main__':main()
