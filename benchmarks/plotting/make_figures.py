"""Regenerate manuscript figures from recorded aggregates; no device benchmark."""
from pathlib import Path
from paper_naming import SYSNAME, DEVICE_LABELS
from paper_paths import ROOT, DATA as DATA_DIR
import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.path import Path as PlotPath

plt.rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 9,
    "mathtext.fontset": "dejavuserif", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
})
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
OUT = args.output_dir.resolve() if args.output_dir else ROOT.parent / "results" / "figures"
SNAPSHOTS = OUT
OUT.mkdir(parents=True, exist_ok=True)
DATA = json.loads((DATA_DIR / "paper_data.json").read_text(encoding="utf-8"))

fig, axs = plt.subplots(1, 3, figsize=(9.2, 3.15), sharey=True)
colors = ["#4d9290", "#273d60"]
names = ["DeltaNet", "GDN", "KDA", "GDN2"]
for ax, dev in zip(axs, ["GPU","A800","NPU"]):
    x = np.arange(4)
    for j, label in enumerate(["Recompute","Full saving"]):
        vals = [DATA[dev][n][j] for n in names]
        bars = ax.bar(x+(j-.5)*.35, vals, .33, color=colors[j],
                      edgecolor="white", linewidth=.4, label=label,
                      hatch="" if j==0 else "///")
        for b,v in zip(bars,vals):
            ax.text(b.get_x()+b.get_width()/2,v+.045,f"{v:.4f}",
                    ha="center",va="bottom",fontsize=7.0,rotation=90)
    ax.axhline(1,color="#696969",ls=":",lw=.8)
    ax.set_xticks(x,names,fontsize=8)
    ax.set_ylim(0,3.45)
    ax.set_title(DEVICE_LABELS[dev],fontsize=10,fontweight="bold")
    ax.set_axisbelow(True)
    ax.grid(axis="y",color="#e5e7eb",lw=.5)
axs[0].set_ylabel("Complete-backward speedup")
axs[0].legend(loc="upper left",frameon=False,fontsize=7.5)
fig.tight_layout(pad=.7,w_pad=1.6)
fig.savefig(OUT/"backward_speedups.pdf",bbox_inches="tight")
fig.savefig(OUT/"backward_speedups.png",dpi=180,bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(3.78,3.75))
fig.subplots_adjust(left=.02,right=.98,bottom=.025,top=.98)
ax.set_xlim(0,10); ax.set_ylim(0,9); ax.axis("off")
def box(x,y,w,h,text,face="#eef2f5",color="#273d60",fs=9.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.05,rounding_size=0.08",
                              fc=face,ec=color,lw=1.0,zorder=3))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color="#1f2937",zorder=4)
def arrow(points,color="#4b5663"):
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=14,
                                shrinkA=0,shrinkB=0,lw=1.35,color=color,zorder=2))
ax.text(.05,8.65,"Expanded WY",fontsize=10,fontweight="bold",color="#273d60")
ax.text(9.95,8.65,"7 products",ha="right",fontsize=10,fontweight="bold",color="#273d60")
box(.05,7.30,4.25,.92,r"$\bar W=-\bar R S^\top$"+"\n1 product")
box(5.15,7.07,4.55,1.15,r"$\bar A=\bar RZ^\top+\bar WE^\top$"+"\n2 products")
box(.05,5.67,4.25,.92,r"$\bar Z=A^\top\bar R$"+"\n1 product")
box(5.15,5.67,4.55,.92,r"$\bar E_{\rm dir}=A^\top\bar W$"+"\n1 product")
box(1.4,4.18,7.2,.92,r"$D=-\mathcal{L}_{<}(A^\top\bar A A^\top)$"+"\n2 products")
arrow([(4.35,7.76),(5.10,7.76)])
arrow([(2.175,7.25),(2.175,6.95),(4.60,6.95),(4.60,6.13),(5.10,6.13)])
# Route the inverse adjoint outside both boxes, with a long leftward final leg.
arrow([(9.75,7.645),(9.95,7.645),(9.95,4.64),(8.65,4.64)])
ax.plot([0,10],[3.86,3.86],color="#d4d9df",lw=.7)
ax.text(.05,3.46,f"{SYSNAME} pullback",fontsize=10,fontweight="bold",color="#276d69")
ax.text(9.95,3.46,"3 products",ha="right",fontsize=10,fontweight="bold",color="#276d69")
box(2.4,2.25,5.2,1.03,r"$P=A^\top\bar R$"+"\n"+r"1 product; $\bar Z=P$",
    face="#e9f4f2",color="#458881")
box(.05,.68,4.7,.96,r"$D=-\mathcal{L}_{<}(PR^\top)$"+"\n1 product",
    face="#e9f4f2",color="#458881")
box(5.25,.68,4.7,.96,r"$\bar E_{\rm dir}=-PS^\top$"+"\n1 product",
    face="#e9f4f2",color="#458881")
arrow([(3.6,2.20),(2.4,1.69)],"#247b72")
arrow([(6.4,2.20),(7.6,1.69)],"#247b72")
ax.text(5,.05,r"Excluded: $DY$, $D^\top E$, state adjoint.",
        ha="center",fontsize=9,color="#4b5563")
fig.savefig(OUT/"dataflow.pdf",bbox_inches="tight")
fig.savefig(OUT/"dataflow.png",dpi=180,bbox_inches="tight")
plt.close(fig)

# Figure 1 uses explicit operations and dependencies, not matrix-count icons.
import runpy
runpy.run_path(str(Path(__file__).with_name('make_intro_graph.py')), run_name='__main__',
               init_globals={'FIGURE_OUTPUT': OUT, 'SNAPSHOT_OUTPUT': SNAPSHOTS})

# The result panels read the curated measurements and exact-value appendix.
import runpy
runpy.run_path(str(Path(__file__).with_name('make_result_figures.py')), run_name='__main__',
               init_globals={'FIGURE_OUTPUT': OUT, 'SNAPSHOT_OUTPUT': SNAPSHOTS})
