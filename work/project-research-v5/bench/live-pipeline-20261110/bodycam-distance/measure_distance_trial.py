"""Project WIDER FACE labels into actual Nicla pixels using a matched checkerboard."""
import argparse
import json
from pathlib import Path
from statistics import median

import cv2
import numpy as np
from PIL import Image, ImageDraw


def fit_calibration(path):
    image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    found,corners = cv2.findChessboardCornersSB(image,(7,7),flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise ValueError(f'Checkerboard inner corners not found in {path}')
    grid = corners.reshape(7,7,2)
    delta = grid[0,-1]-grid[0,0]
    if abs(delta[1]) > abs(delta[0]):
        grid = grid.transpose(1,0,2)
    if grid[0,-1,0] < grid[0,0,0]:
        grid = grid[:,::-1]
    if grid[-1,0,1] < grid[0,0,1]:
        grid = grid[::-1]
    src = np.array([(x,y) for y in range(100,800,100)
                    for x in range(100,800,100)],np.float32)
    dest = grid.reshape(-1,2)
    transform,_ = cv2.findHomography(src,dest)
    projected = cv2.perspectiveTransform(src[None],transform)[0]
    return transform, float(np.linalg.norm(projected-dest,axis=1).max())


def project(transform,box):
    x,y,w,h=box
    pts=np.array([[[x,y],[x+w,y],[x+w,y+h],[x,y+h]]],np.float32)
    mapped=cv2.perspectiveTransform(pts,transform)[0]
    a,b=mapped.min(axis=0)
    c,d=mapped.max(axis=0)
    return [round(float(v),2) for v in (a,b,c-a,d-b)]


def intersection(left,right):
    x,y,w,h=left
    a,b,c,d=right
    return max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))


def has_detector_box(row):
    return bool((row.get('detection') or {}).get('regions_xywh'))


def full(row):
    return (row.get('detection') or {}).get('status')=='face' and (
        row.get('crop') or {}).get('status')=='valid' and (
        row.get('identity') or {}).get('status')=='ran' and all(
        row.get('payload_checks',{}).get(key) for key in ('frame','detector_input','identity_input'))


def analyze(directory):
    rows=[json.loads(line) for line in (directory/'captures.jsonl').read_text().splitlines()]
    calibration=[r for r in rows if r['scene'].startswith('optical-calibration') and r.get('jpeg')]
    if len(calibration)<2:
        raise ValueError('Need checkerboard captures before and after the scene sequence.')
    first,first_error=fit_calibration(directory/calibration[0]['jpeg'])
    last,last_error=fit_calibration(directory/calibration[-1]['jpeg'])
    virtual_grid=(0,0,800,800)
    frame_delta=max(abs(a-b) for a,b in zip(project(first,virtual_grid),project(last,virtual_grid)))
    if frame_delta>4:
        raise ValueError(f'Display geometry changed by {frame_delta:.1f} camera pixels')
    blocks={}
    for row in rows:
        blocks.setdefault(row['scene'],[]).append(row)
        if row.get('face_box_xywh_display_px'):
            label=project(first,row['face_box_xywh_display_px'])
            row['projected_face_xywh_sensor_px']=label
            boxes=(row.get('detection') or {}).get('regions_xywh') or []
            def iou(region):
                overlap=intersection(label,region)
                return overlap/(label[2]*label[3]+region[2]*region[3]-overlap)
            row['best_face_box_iou']=max((iou(box['rect_px']) for box in boxes),default=0)
            crop=(row.get('crop') or {}).get('rect_px')
            row['face_coverage_in_crop']=(intersection(label,crop)/(label[2]*label[3]) if crop else 0)
            row['matched_pipeline']=full(row) and row['best_face_box_iou']>=.3 and row['face_coverage_in_crop']>=.8
    summary={'scope':'single fixed screen-to-camera setup; projected face sizes, not physical distances',
        'checkerboard_max_residual_px':[round(first_error,2),round(last_error,2)],
        'end_to_end_projection_change_px':round(frame_delta,2),
        'complete_shots':sum(r.get('ack_status')=='completed' and r.get('payload_checks',{}).get('frame') for r in rows),
        'shots':len(rows),'blocks':{}}
    print(directory.name, 'projection drift',round(frame_delta,2),'residual',round(first_error,2),round(last_error,2))
    cols=4
    sheet=Image.new('RGB',(cols*252,((len(blocks)+cols-1)//cols)*284),'#eaeaea')
    draw=ImageDraw.Draw(sheet)
    for i,(name,group) in enumerate(blocks.items()):
        rep=next((r for r in group if r.get('matched_pipeline')),group[0])
        x,y=i%cols*252,i//cols*284
        expected=rep.get('projected_face_xywh_sensor_px')
        info={'shots':len(group),
            'expected_face_wh_px':expected[2:] if expected else None,
            'face_boxes':sum(has_detector_box(r) for r in group),
            'full_pipeline':sum(full(r) for r in group),
            'matched_pipeline':sum(bool(r.get('matched_pipeline')) for r in group),
            'no_face':sum((r.get('decision') or {}).get('status')=='NO_FACE' for r in group),
            'frame_ids':[r.get('frame_id') for r in group]}
        complete_times=[r['timings_ms']['total'] for r in group if full(r)]
        if complete_times:
            info['median_full_ms']=median(complete_times)
        summary['blocks'][name]=info
        print(name,info['expected_face_wh_px'],info['face_boxes'],info['full_pipeline'],
              info['matched_pipeline'],'/',len(group),'no_face',info['no_face'])
        if rep.get('jpeg'):
            image=Image.open(directory/rep['jpeg']).convert('RGB')
            overlay=ImageDraw.Draw(image)
            def outline(rect, color, stroke):
                a,b,w,h=rect
                overlay.rectangle((a,b,a+w,b+h),outline=color,width=stroke)
            if expected:
                outline(expected,'#ffae53',2)
            for box in (rep.get('detection') or {}).get('regions_xywh') or []:
                outline(box['rect_px'],'#46ffc4',1)
            sheet.paste(image,(x+5,y+5))
        draw.text((x+5,y+248),name,fill='#222')
        measure=(f'{expected[2]:.0f}x{expected[3]:.0f} px' if expected else 'control')
        draw.text((x+5,y+263),f"{measure} · {info['matched_pipeline']}/{len(group)} matched",fill='#222')
    (directory/'distance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (directory/'distance_review_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    sheet.save(directory/'distance_contact_sheet.jpg',quality=93)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',type=Path,nargs='+')
    for run in parser.parse_args().runs:
        analyze(run)
