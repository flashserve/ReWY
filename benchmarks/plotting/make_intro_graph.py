"""Figure 1: the actual local backward DAG, before and after residual WY."""
from pathlib import Path
from paper_naming import SYSNAME
from paper_paths import ROOT, DATA
import argparse
import sys
import json

if 'FIGURE_OUTPUT' in globals():
    OUT, SNAPSHOTS = FIGURE_OUTPUT, SNAPSHOT_OUTPUT
else:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    OUT = args.output_dir.resolve() if args.output_dir else ROOT.parent / 'results' / 'figures'
    SNAPSHOTS = OUT
OUT.mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as PlotPath

plt.rcParams.update({'font.family':'DejaVu Serif','mathtext.fontset':'dejavuserif','pdf.fonttype':42})
NAVY, TEAL, ORANGE, GRAY = '#273d60', '#247b72', '#ad593b', '#4b5663'
fig, ax = plt.subplots(figsize=(7.7,4.2))
fig.subplots_adjust(left=.006,right=.994,bottom=.012,top=.99)
ax.set_xlim(0,17);ax.set_ylim(0,9.1);ax.axis('off')

nodes = {}
edges = []
def node(key,x,y,w,h,label,kind='product',gemms=0):
    colors={'product':('#f0f3f7',NAVY),'removed':('#fff0e8',ORANGE),
            'output':('#eaf5f1',TEAL),'shared':('#d6eee7',TEAL),'input':('#ffffff',GRAY)}
    face,stroke=colors[kind]
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.04,rounding_size=.08',
                               fc=face,ec=stroke,lw=1.1,zorder=3))
    ax.text(x,y,label,ha='center',va='center',fontsize=9.5 if kind=='shared' else 9,
            color=stroke,linespacing=1.2,zorder=4)
    nodes[key]=dict(x=x,y=y,w=w,h=h,label=label,kind=kind,gemms=gemms)

def port(key,where):
    n=nodes[key];x,y,w,h=[n[k] for k in ('x','y','w','h')]
    return {'top':(x,y+h/2+.04),'bottom':(x,y-h/2-.04),
            'left':(x-w/2-.04,y),'right':(x+w/2+.04,y)}[where]

def edge(source,target,start='bottom',end='top',via=None,color=GRAY,dashed=False):
    points=[port(source,start)]+(via or [])+[port(target,end)]
    # A single path keeps bends continuous and the head at the target port.
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle='-|>',mutation_scale=15,
                                shrinkA=0,shrinkB=0,color=color,lw=1.45,
                                linestyle='--' if dashed else '-',zorder=2))
    edges.append([source,target])

ax.text(.18,8.72,'(a) Expanded WY: 7 products',fontsize=10,weight='bold',color=NAVY)
ax.text(10.05,8.72,f'(b) {SYSNAME}: 3 products',fontsize=10,weight='bold',color=TEAL)
ax.text(.18,8.24,r'Forward operands: $A, E, Z, S$',fontsize=8,color=GRAY)
ax.text(10.05,8.24,r'Reuse forward correction $R$',fontsize=8,color=GRAY)
ax.plot([9.6,9.6],[1.0,8.85],color='#d4dce2',lw=1)

# Seven explicit matrix products, plus a sum. The orange intermediate values
# are precisely those removed by differentiating the residual map as a whole.
node('b_r',5.0,7.55,1.1,.52,r'$\bar R$',kind='input')
node('b_w',1.65,6.35,2.8,.74,r'$\bar W=-\bar R S^\top$',kind='removed',gemms=1)
node('b_rz',5.0,6.35,2.25,.74,r'$T_Z=\bar R Z^\top$',gemms=1)
node('b_z',8.15,6.35,2.45,.74,r'$\bar Z=A^\top\bar R$',kind='output',gemms=1)
node('b_e',1.65,4.78,2.85,.74,r'$\bar E_{\rm dir}=A^\top\bar W$',kind='output',gemms=1)
node('b_we',4.95,4.78,2.5,.74,r'$T_E=\bar W E^\top$',gemms=1)
node('b_a',7.85,3.50,2.75,.74,r'$\bar A=T_Z+T_E$',kind='removed')
node('b_left',5.15,2.16,2.15,.74,r'$J=A^\top\bar A$',gemms=1)
node('b_d',8.0,.99,2.95,.74,r'$D=-\mathcal{L}_{<}(JA^\top)$',kind='output',gemms=1)

edge('b_r','b_w',start='left',via=[(1.65,7.55)])
edge('b_r','b_rz')
edge('b_r','b_z',start='right',via=[(8.15,7.55)])
edge('b_w','b_e')
edge('b_w','b_we',start='right',end='left',via=[(3.35,6.35),(3.35,4.78)])
edge('b_rz','b_a',start='right',end='top',via=[(6.45,6.35),(6.45,4.10),(7.85,4.10)])
edge('b_we','b_a',start='bottom',end='left',via=[(4.95,3.50)])
edge('b_a','b_left',start='bottom',end='right',via=[(7.85,2.16)])
edge('b_left','b_d',start='bottom',end='left',via=[(5.15,.99)])

ax.text(.3,3.40,'Eliminated adjoints',fontsize=8.5,color=ORANGE,weight='bold')
ax.text(.3,2.96,r'$\bar W$ and $\bar A$',fontsize=8.5,color=GRAY)
ax.text(.3,2.48,'Inverse pullback:',fontsize=8.5,color=NAVY)
ax.text(.3,2.04,r'$\bar A\;\longrightarrow\; A^\top\bar A\;\longrightarrow\;D$',fontsize=8.5,color=NAVY)

# Three products. P is shared by all three required gradients. Zbar is an
# alias of P, not an additional multiplication. R is a saved forward value.
node('r_r',13.25,7.55,1.1,.52,r'$\bar R$',kind='input')
node('r_p',13.25,6.35,2.8,.86,r'$P=A^\top\bar R$',kind='shared',gemms=1)
node('r_z',11.25,4.80,2.0,.74,r'$\bar Z=P$',kind='output')
node('r_e',15.30,4.80,2.9,.74,r'$\bar E_{\rm dir}=-PS^\top$',kind='output',gemms=1)
node('r_d',13.25,2.65,3.45,.86,r'$D=-\mathcal{L}_{<}(PR^\top)$',kind='output',gemms=1)
node('r_forward',10.45,2.65,.8,.52,r'$R$',kind='input')
edge('r_r','r_p',color=TEAL)
edge('r_p','r_z',start='left',via=[(11.25,6.35)],color=TEAL)
edge('r_p','r_e',start='right',via=[(15.30,6.35)],color=TEAL)
edge('r_p','r_d',color=TEAL)
edge('r_forward','r_d',start='right',end='left',color=TEAL,dashed=True)
ax.text(11.25,4.15,'Alias; no product',ha='center',fontsize=8,color=TEAL)
ax.text(10.60,1.94,'Forward value',ha='center',fontsize=8,color=GRAY)
ax.text(13.25,1.37,'One shared sensitivity',ha='center',fontsize=8.5,color=TEAL,weight='bold')

ax.text(.2,.12,r'Same outputs: $\bar Z,\bar E_{\rm dir},D$.  Common downstream work: $DY$, $D^\top E$, and the state adjoint.',
        fontsize=8.5,color=GRAY)

for prefix,n in [('b_',7),('r_',3)]:
    assert sum(v['gemms'] for k,v in nodes.items() if k.startswith(prefix))==n
for ext in ('pdf','svg','png'):
    fig.savefig(OUT/f'intro_overview.{ext}',dpi=220,bbox_inches='tight',pad_inches=.025)
plt.close(fig)
(SNAPSHOTS/'intro_graph_data.json').write_text(json.dumps(dict(nodes=nodes,edges=edges,
    expanded_products=7,residual_products=3,unchanged_downstream=['D Y','D^T E','state adjoint']),indent=2)+'\n')
print('Generated Figure 1 as an explicit dependency DAG.')
