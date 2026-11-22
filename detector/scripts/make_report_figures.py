"""Create development-only detector/crop/resource comparison figures."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--comparison',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True);ap.add_argument('--outdir',type=Path,required=True);a=ap.parse_args()
 d=json.loads(a.comparison.read_text()); audit=json.loads(a.audit.read_text()); a.outdir.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,'axes.labelsize':10})
 colors={'cheap_sepconv_16':'#0b7a75','mobilenetv2_alpha_0p1':'#9d3c63'}
 fig,ax=plt.subplots(figsize=(8.5,5.5),layout='constrained')
 for arm in colors:
  rows=[r for r in d['learned_roots'] if r['arm']==arm]
  xs=[r['exports']['full_int8']['selected_metrics']['false_trigger_scene_rate']*100 for r in rows]
  ys=[r['exports']['full_int8']['selected_metrics']['f1'] for r in rows]
  ax.scatter(xs,ys,s=46,alpha=.78,color=colors[arm],label=f'{arm} (10 INT8 roots)',edgecolors='white',linewidths=.45)
  agg=d['learned_arm_aggregates'][arm]['full_int8']['metrics_median']
  ax.scatter([agg['false_trigger_scene_rate']*100],[agg['f1']],s=155,marker='D',color=colors[arm],edgecolors='black',linewidths=.8,zorder=5)
  ax.annotate(f'median F1 {agg["f1"]:.3f}',(agg['false_trigger_scene_rate']*100,agg['f1']),xytext=(8,8),textcoords='offset points',color=colors[arm],weight='bold')
 stock=d['fixed_stock_at_0p4']
 ax.scatter([stock['false_trigger_scene_rate']*100],[stock['f1']],marker='*',s=220,color='#3056a5',edgecolors='black',linewidths=.7,label='Stock RGB FOMO, fixed threshold 0.40',zorder=7)
 ax.annotate(f'stock F1 {stock["f1"]:.3f}\n{stock["false_trigger_scene_rate"]*100:.0f}% false-trigger scenes',
   (stock['false_trigger_scene_rate']*100,stock['f1']),xytext=(10,12),textcoords='offset points',color='#233f81',weight='bold')
 ax.axvline(5,color='#b43a3a',linestyle='--',linewidth=1.5,label='5% false-trigger limit')
 ax.set(xlabel='False-trigger scenes among 100 development PASS negatives (%)',ylabel='Face-centroid F1',title='Development detection trade-off (thresholds selected per root)')
 ax.set_xlim(-1,88);ax.set_ylim(-.02,.40);ax.grid(alpha=.22);ax.legend(loc='upper left',frameon=True,fontsize=8)
 ax.text(.99,.02,'No learned root met the 5% limit. Compact roots are plotted at fallback threshold 0.70.',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#444')
 fig.savefig(a.outdir/'development_detection_tradeoff.png',dpi=180);plt.close(fig)

 stock=d['fixed_stock_at_0p4']; root=next(r for r in d['learned_roots'] if r['arm']=='cheap_sepconv_16' and r['root_seed']==20261102)
 c=root['exports']['full_int8']['selected_metrics']['crop_methods'];s=stock['crop_methods']
 sides=[80,96,112,128]
 fig,ax=plt.subplots(figsize=(8.5,5.5),layout='constrained')
 ax.plot(sides,[s[str(x)]['annotated_face_box_coverage_fraction']*100 for x in sides],'-o',color='#3056a5',label='Stock: conditional face-box coverage')
 ax.plot(sides,[s[str(x)]['end_to_end_annotated_face_coverage_fraction']*100 for x in sides],'--o',color='#3056a5',alpha=.78,label='Stock: coverage across all labeled centers')
 ax.plot(sides,[c[str(x)]['annotated_face_box_coverage_fraction']*100 for x in sides],'-s',color='#0b7a75',label='Compact INT8 seed 20261102: conditional coverage')
 ax.plot(sides,[c[str(x)]['end_to_end_annotated_face_coverage_fraction']*100 for x in sides],'--s',color='#0b7a75',alpha=.78,label='Compact INT8: coverage across all labeled centers')
 # Exact legacy crop geometry is separately shown at the common 96 px side.
 ax.scatter([96],[s['legacy_current_fixed_96']['annotated_face_box_coverage_fraction']*100],marker='D',s=75,color='#3056a5',edgecolors='black',zorder=5,label='Stock legacy-current fixed 96 px')
 ax.axvline(96,color='#555',linestyle=':',linewidth=1)
 ax.set(xlabel='Native grayscale square crop side in camera-window pixels',ylabel='Annotated face-box coverage (%)',title='Development crop coverage by fixed crop size')
 ax.set_xticks(sides);ax.grid(alpha=.22);ax.legend(fontsize=8,loc='upper right')
 ax.text(.99,.02,'Every valid crop is resized to 96×96 gray8. D-conditioned identity quality was unavailable.',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#444')
 fig.savefig(a.outdir/'development_crop_coverage.png',dpi=180);plt.close(fig)

 models=[{'label':'Stock RGB FOMO','audit':audit['stock_reference'],'host_ms':d['stock_reference']['audit']['host_interpreter_ms_per_scene'],'f1':stock['f1'],'false':stock['false_trigger_scene_rate'],'arm':None},
         {'label':'MobileNetV2 α=0.1','audit':None,'f1':d['learned_arm_aggregates']['mobilenetv2_alpha_0p1']['full_int8']['metrics_median']['f1'],'false':d['learned_arm_aggregates']['mobilenetv2_alpha_0p1']['full_int8']['metrics_median']['false_trigger_scene_rate'],'arm':'mobilenetv2_alpha_0p1'},
         {'label':'Compact SepConv-16','audit':None,'f1':d['learned_arm_aggregates']['cheap_sepconv_16']['full_int8']['metrics_median']['f1'],'false':d['learned_arm_aggregates']['cheap_sepconv_16']['full_int8']['metrics_median']['false_trigger_scene_rate'],'arm':'cheap_sepconv_16'}]
 for item in models:
  if item['audit'] is None:
   roots=[r for r in audit['roots'] if r['arm']==item['arm']]
   item['audit']=roots[0]['full_int8']
   dev_roots=[r for r in d['learned_roots'] if r['arm']==item['arm']]
   item['host_ms']=float(np.median([r['exports']['full_int8']['audit']['host_interpreter_ms_per_scene'] for r in dev_roots]))
 fig,ax=plt.subplots(figsize=(8.5,5.5),layout='constrained')
 sizes=[m['audit']['bytes']/1024 for m in models]
 cols=['#3056a5','#9d3c63','#0b7a75']; bars=ax.bar([m['label'] for m in models],sizes,color=cols,width=.55)
 ax.set_ylabel('Full-INT8 model file size (KiB)');ax.set_title('Artifact size, development quality, and host timing')
 ax.grid(axis='y',alpha=.2)
 for bar,m in zip(bars,models):
  ops=len(m['audit']['operators']); ms=m['host_ms']
  ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.35,
    f'{bar.get_height():.1f} KiB\nF1 {m["f1"]:.3f} · false triggers {m["false"]*100:.0f}%\n{ops} op types · host {ms:.2f} ms/scene',ha='center',va='bottom',fontsize=8)
 ax.set_ylim(0,max(sizes)*1.55)
 ax.text(.99,.02,'Host TFLite timings only; no Nicla timing/arena was measured for trained models.',transform=ax.transAxes,ha='right',va='bottom',fontsize=8,color='#444')
 fig.savefig(a.outdir/'development_resource_quality.png',dpi=180);plt.close(fig)
 print(json.dumps({'figures':[str(a.outdir/n) for n in ('development_detection_tradeoff.png','development_crop_coverage.png','development_resource_quality.png')]},indent=2))
if __name__=='__main__':main()
