"""Prepare a checkerboard shot and bind its Nicla geometry to the video layout."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout',type=Path,required=True)
    parser.add_argument('--scenes',type=Path)
    parser.add_argument('--captures',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    layout=json.loads(args.layout.read_text())
    fields=('size_vmin','opacity','background','x_pct','y_pct','rotation_deg')
    if args.scenes:
        scene=['optical-calibration-start','calibration',*(layout[k] for k in fields)]
        args.scenes.write_text(json.dumps([scene])+'\n')
    if not args.captures:
        return
    if not args.output:
        parser.error('--captures requires --output')
    spec=importlib.util.spec_from_file_location('measure_distance',
        HERE.parents[1]/'bodycam-distance/measure_distance_trial.py')
    measure=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measure)
    measure.cv2.ocl.setUseOpenCL(False)
    rows=[json.loads(line) for line in (args.captures/'captures.jsonl').read_text().splitlines()]
    if len(rows)<2:
        raise ValueError('Collect at least two frame-linked checkerboard shots.')
    fits=[]
    for row in rows:
        if (row.get('ack_status')!='completed' or not row['payload_checks']['frame']
                or any(row['display'].get(k)!=layout[k] for k in fields)):
            raise ValueError('Checkerboard frame or layout did not match.')
        path=args.captures/row['jpeg']
        matrix,error=measure.fit_calibration(path)
        if error>2:
            raise ValueError(f'Checkerboard residual was {error:.2f} pixels')
        fits.append((matrix,error,hashlib.sha256(path.read_bytes()).hexdigest()))
    matrix=fits[0][0]
    boxes=[measure.project(fit[0],(0,0,800,800)) for fit in fits]
    drift=max(abs(a-b) for box in boxes for a,b in zip(boxes[0],box))
    if drift>3:
        raise ValueError(f'The display moved {drift:.2f} camera pixels during calibration.')
    x,y,w,h=boxes[0]
    if x<0 or y<0 or x+w>240 or y+h>240:
        raise ValueError('Some of the calibrated video square falls outside the Nicla frame.')
    result={'layout':layout,'video_px_to_nicla_px':matrix.tolist(),
        'viewport':rows[0]['render']['viewport'],'video_bounds_nicla_px':boxes[0],
        'maximum_drift_px':round(drift,3),'max_corner_residual_px':round(max(f[1] for f in fits),3),
        'boot_id':rows[0]['boot_id'],'checkerboard_frame_ids':[r['frame_id'] for r in rows],
        'checkerboard_jpeg_sha256':[f[2] for f in fits]}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Calibrated visible video bounds',boxes[0])


if __name__=='__main__':
    main()
