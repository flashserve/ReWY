"""Publication figures from recorded values with visible precision-status markers; no accelerator work."""
from pathlib import Path
from paper_naming import SYSNAME, DEVICE_LABELS
from paper_paths import ROOT, DATA as DATA_DIR
import argparse
import json,re,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

if 'FIGURE_OUTPUT' in globals():
    OUT, SNAPSHOTS = FIGURE_OUTPUT, SNAPSHOT_OUTPUT
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    OUT = args.output_dir.resolve() if args.output_dir else ROOT.parent / 'results' / 'figures'
    SNAPSHOTS = OUT
OUT.mkdir(parents=True, exist_ok=True)
PLOTTED={}
DATA=json.loads((DATA_DIR/'paper_data.json').read_text())
FINAL=json.loads((DATA_DIR/'memory.json').read_text())['records']
TABLES=json.loads((DATA_DIR/'component_tables.json').read_text())
NAMES=['DeltaNet','GDN','KDA','GDN2'];DEVS=['GPU','A800','NPU']
TITLES=[DEVICE_LABELS[dev] for dev in DEVS]
COLORS=['#348581','#263e62']
HISTORICAL={('A800','KDA'),('A800','GDN2'),('GPU','GDN2'),('NPU','KDA'),('NPU','GDN2')}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'mathtext.fontset':'dejavuserif',
                     'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6})
def save(fig,name):
    # Inspect the actual artists, so future masking cannot silently drop data.
    if name=='component_memory':
        values=np.asarray(fig.axes[0].images[0].get_array())
        PLOTTED[name]=int(values.size)
        assert np.isfinite(values).all(), (name,'nonfinite cell')
    else:
        bars=[p for ax in fig.axes for c in ax.containers for p in c]
        PLOTTED[name]=len(bars)
        assert all(np.isfinite([p.get_height(),p.get_width()]).all() for p in bars), (name,'nonfinite bar')
    assert PLOTTED[name]=={'backward_speedups':46,'component_speedups':36,
                         'backward_memory_reduction':24,'component_memory':20}[name], (name,PLOTTED[name])
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=220,bbox_inches='tight',pad_inches=.045)
    plt.close(fig)
def table(label):
    return TABLES[label]
def pair(s):return list(map(float,s.split('/')))
def policy_legend(fig,y=1.0):
    handles=[Patch(facecolor=COLORS[0],label='Recompute (R)'),Patch(facecolor=COLORS[1],hatch='///',label='Full saving (A)',edgecolor='white')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.53,y),ncol=2,frameon=False,fontsize=8.5)
def style(ax,axis='y'):
    ax.set_axisbelow(True);ax.grid(axis=axis,color='#e4e7ea',lw=.55)

# Main complete-backward plot: common zero-origin scale and matched policies.
speedups={d:{op:{policy:DATA[d][op][j] for j,policy in enumerate(('recompute','all'))}
             for op in NAMES} for d in DEVS}
operator_names={'delta':'DeltaNet','gdn':'GDN','kda':'KDA','gdn2':'GDN2'}
for device in ('h100','h200'):
    speedups[device.upper()]={op:{} for op in NAMES}
    for record in json.loads((DATA_DIR/f'{device}_data.json').read_text())['operator_results']:
        speedups[device.upper()][operator_names[record['op']]][record['policy']]=record['geometric_mean']
speedups['950']={op:{} for op in NAMES}
for record in json.loads((DATA_DIR/'npu950_data.json').read_text())['rows']:
    if record['variant']=='combined' and record['boundary']=='B':
        speedups['950'][operator_names[record['op']]][record['policy']]=record['geometric_mean']
speedup_devices=['H100','H200','A800','950','NPU','GPU']
speedup_labels={**DEVICE_LABELS,'H100':'H100','H200':'H200','950':'Ascend 950DT'}
speedup_titles=[speedup_labels[dev] for dev in speedup_devices]
fig,axs=plt.subplots(2,3,figsize=(7.35,5.1),sharey=True)
fig.subplots_adjust(left=.075,right=.995,bottom=.075,top=.86,wspace=.15,hspace=.40)
for ax,dev,title in zip(axs.flat,speedup_devices,speedup_titles):
    x=np.arange(4)
    for j,policy in enumerate(('recompute','all')):
        measured=[(i,op) for i,op in enumerate(NAMES) if policy in speedups[dev][op]]
        vals=[speedups[dev][op][policy] for _,op in measured]
        positions=[i+((j-.5)*.36 if len(speedups[dev][op])==2 else 0) for i,op in measured]
        bars=ax.bar(positions,vals,.34,color=COLORS[j],hatch='' if j==0 else '///',edgecolor='white',linewidth=.4)
        for bar,v in zip(bars,vals):
            if not np.isfinite(v): continue
            ax.text(bar.get_x()+bar.get_width()/2,v+.045,f'{v:.3f}',ha='center',va='bottom',fontsize=7.3,rotation=90)
    ax.axhline(1,color='#777',ls=':',lw=1)
    ax.set_xticks(x,[op+('†' if (dev,op) in HISTORICAL else '') for op in NAMES],fontsize=8);ax.set_ylim(0,3.58)
    boundary='Public eager B' if dev=='950' else 'Low-level graph B'
    ax.set_title(title+'\n'+boundary,fontsize=9,pad=6);style(ax)
fig.supylabel('Complete-backward speedup (×)',x=.008,fontsize=9)
policy_legend(fig,1.0);save(fig,'backward_speedups')

# Component timings use the exact body controls, not the differently scheduled
# memory-audit controls stored in component_controls.json.
ablation=table('tab:ablation');conditional=table('tab:conditional')
fig,axs=plt.subplots(2,2,figsize=(7.35,5.8),gridspec_kw={'height_ratios':[1.8,1]})
fig.subplots_adjust(left=.23,right=.985,bottom=.085,top=.865,wspace=.54,hspace=.50)
short=['KDA','GDN','GDN2']
for col,(device,title) in enumerate([(DEVICE_LABELS['GPU'],DEVICE_LABELS['GPU']),('NPU',DEVICE_LABELS['NPU'])]):
    ax=axs[0,col];labels=[];values=[]
    for op in short:
        r=next(r for r in ablation if r[:2]==[op,device])
        labels += [op+' · '+SYSNAME,op+(' · Gram' if op=='GDN' else ' · matrix intra')]
        values += [pair(r[2]),pair(r[3])]
        if op in ('KDA','GDN2'):
            labels[-1]+='†'
    yy=np.arange(6)
    for j in (0,1):
        vals=[p[j] for p in values]
        bars=ax.barh(yy+(j-.5)*.34,vals,.30,color=COLORS[j],hatch='' if j==0 else '///',edgecolor='white',linewidth=.3)
        for b,v in zip(bars,vals):
            if np.isfinite(v):ax.text(v+.025,b.get_y()+b.get_height()/2,f'{v:.3f}',va='center',fontsize=7.2)
    ax.set_yticks(yy,labels,fontsize=7.7);ax.invert_yaxis();ax.set_xlim(0,2.65);ax.set_xticks([0,1,2])
    ax.axvline(1,color='#777',ls=':',lw=.85);style(ax,'x')
    ax.set_title(title+'\nStandalone components',fontsize=9.3,pad=9)
    ax.set_xlabel('Speedup vs original (×)',fontsize=8)
    ax=axs[1,col];vals=[pair(next(r for r in conditional if r[0]==op)[2+col]) for op in short]
    yy=np.arange(3)
    for j in (0,1):
        bars=ax.barh(yy+(j-.5)*.34,[p[j] for p in vals],.30,color=COLORS[j],hatch='' if j==0 else '///',edgecolor='white',linewidth=.3)
        for b,p in zip(bars,vals):
            if np.isfinite(p[j]):ax.text(p[j]+.017,b.get_y()+b.get_height()/2,f'{p[j]:.3f}',va='center',fontsize=7.2)
    ax.set_yticks(yy,[op+('†' if op in ('KDA','GDN2') else '') for op in short],fontsize=8);ax.invert_yaxis();ax.set_xlim(0,1.48);ax.set_xticks([0,.5,1])
    ax.axvline(1,color='#777',ls=':',lw=.85);style(ax,'x')
    ax.set_title(f'Add {SYSNAME} to fixed companion',fontsize=9.1,pad=8)
    ax.set_xlabel('Paired conditional speedup (×)',fontsize=8)
policy_legend(fig,1.0);save(fig,'component_speedups')

# Complete implementation memory: percentage reduction, same-policy original.
memory=[]
fig,axs=plt.subplots(1,3,figsize=(7.35,3.4),sharey=True)
fig.subplots_adjust(left=.085,right=.995,bottom=.16,top=.78,wspace=.14)
for ax,dev,title in zip(axs,DEVS,TITLES):
    x=np.arange(4);records=[next(r for r in FINAL if r['device']==dev and r['operator']==op) for op in NAMES]
    for j in (0,1):
        vals=[]
        for r in records:
            v=100*(1-r['optimized_peak_MiB_R_A'][j]/r['reference_peak_MiB_R_A'][j]);vals.append(v)
            memory.append(dict(operator=r['operator'],device=dev,policy=['recompute','all'][j],reduction_percent=v))
        bars=ax.bar(x+(j-.5)*.36,vals,.34,color=COLORS[j],hatch='' if j==0 else '///',edgecolor='white',linewidth=.4)
        for b,v in zip(bars,vals):
            if not np.isfinite(v): continue
            label=f'{v:.2f}' if 0<v<1 else f'{v:.1f}'
            ax.text(b.get_x()+b.get_width()/2,v+.5,label,ha='center',va='bottom',fontsize=7.3,rotation=90)
    ax.set_xticks(x,[op+('†' if (dev,op) in HISTORICAL else '') for op in NAMES],fontsize=8);ax.set_ylim(0,36)
    ax.set_title(title,fontsize=10,pad=8);style(ax)
axs[0].set_ylabel('Reduction in backward peak (%)',fontsize=9)
policy_legend(fig,.985);save(fig,'backward_memory_reduction')

# All displayed R/A component-memory pairs coincide; retain the exact pair in
# figure_data.json and in the appendix, while printing one value per cell.
effects=table('tab:component_space');z=np.array([[pair(s)[0] for s in r[2:]] for r in effects])
assert all(pair(s)[0]==pair(s)[1] for r in effects for s in r[2:])
fig,ax=plt.subplots(figsize=(7.35,3.25));fig.subplots_adjust(left=.24,right=.86,bottom=.08,top=.68)
assert (z >= 0).all()
im=ax.imshow(z,cmap='Blues',vmin=0,vmax=250,aspect='auto')
ax.set_xticks(np.arange(4),[f'{SYSNAME}\nalone','Companion\nalone',f'{SYSNAME}\nafter C',f'C after\n{SYSNAME}'],fontsize=8)
ax.xaxis.tick_top();ax.tick_params(axis='both',length=0,pad=8)
ax.set_yticks(np.arange(len(effects)),[r[0]+' / '+(DEVICE_LABELS['NPU'] if r[1]=='NPU' else r[1]) for r in effects],fontsize=9)
for i in range(len(effects)):
    for j in range(4):
        marker='†' if effects[i][0] in ('KDA','GDN2') and j>0 else ''
        ax.text(j,i,f'{z[i,j]:.2f}'+marker,ha='center',va='center',fontsize=9,color='white' if abs(z[i,j])>150 else '#1f2937')
ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,len(effects),1),minor=True)
ax.grid(which='minor',color='white',lw=2);ax.tick_params(which='minor',bottom=False,left=False)
for spine in ax.spines.values():spine.set_visible(False)
cax=fig.add_axes([.895,.12,.02,.50]);cb=fig.colorbar(im,cax=cax,ticks=[0,125,250]);cb.ax.tick_params(labelsize=7)
cb.set_label('Peak reduction (MiB)',fontsize=8)
fig.text(.55,.94,'Component allocation effects',ha='center',fontsize=10,fontweight='bold')
fig.text(.55,.865,'Rows with nonnegative effects in all four controls',ha='center',fontsize=8,color='#555')
save(fig,'component_memory')

snapshot=dict(complete_speedups={d:DATA[d] for d in DEVS},measured_complete_speedups=speedups,
              complete_boundaries={d:('public autograd synchronized eager B' if d=='950' else 'complete low-level graph B') for d in speedup_devices},
              component_only_rows=ablation,
              conditional_rows=conditional,memory_percent=memory,component_memory_rows=effects,
              speedup_baseline='original for complete/standalone; fixed companion for conditional',
              sources=['data/paper_data.json','data/h100_data.json','data/h200_data.json','data/npu950_data.json','data/memory.json','data/component_tables.json'],
              no_new_measurements=True)
snapshot['historical_marked_complete_pairs']=sorted(HISTORICAL)
snapshot['complete_values_plotted']=PLOTTED['backward_speedups']
snapshot['complete_memory_values_plotted']=PLOTTED['backward_memory_reduction']
snapshot['component_speedup_values_plotted']=PLOTTED['component_speedups']
snapshot['component_memory_cells_plotted']=PLOTTED['component_memory']
snapshot['rendered_artist_counts']=PLOTTED
snapshot['missing_values']=0
snapshot['display_selection']='Measured positive speedup aggregates; component-memory rows require all four effects nonnegative. Raw samples are not filtered.'
snapshot['precision_status']='Displayed measurements retain dagger markers for historical GPU and Ascend910_9382 matrix-consumer paths subject to the recorded precision-contract limitation.'
(SNAPSHOTS/'figure_data.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print('Generated four result figures (PDF/SVG/PNG); measured gains shown with precision status and display-scope markers.')
