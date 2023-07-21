from matplotlib import pyplot as plt
import pandas as pd

def graph_baseline_ecs_pdf():
    """Plot Baseline posterior from Sherwood's paper."""

    df = pd.read_json("sherwood_ecs.json").reindex()
    pdf = df["pdf"]
    x = df["ECS"]
    cdf = df["cdf"]
    
    ac = "#36494f"
    ac2 = "orange"
    with plt.rc_context(
            {'axes.edgecolor': ac, 'axes.labelcolor': ac, 'xtick.color': ac, 'ytick.color': ac, 'figure.facecolor':
                'white'}):

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3), dpi=300)
        fig.subplots_adjust(bottom=.15)
        l1 = ax.plot(x, pdf, color="k", label="Densité de probabilité")
    
    with plt.rc_context(
            {'axes.labelcolor': ac, 'xtick.color': ac, 'ytick.color': ac2, }):
        ax2 = ax.twinx()
        l2 = ax2.plot(x, 1-cdf, color=ac2, label="Probabilité de dépassement", clip_on=False)
    
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    #ax.legend(lns, labs, frameon=False)

    ax.set_xlim([0,8])
    ax.set_ylim([0, .8])
    ax2.set_xlim([0, 8])
    ax2.set_ylim([0, 1])
        
    for axi in [ax, ax2]:
        for key, spine in axi.spines.items():
            if key in ["top"]:
                spine.set_visible(False)
                
    ax.set_xlabel("Sensibilité climatique effective (K)")
    ax.set_ylabel("Densité de probabilité (K$^{-1}$)")
    ax2.set_ylabel("Probabilité de dépassement")
    
    return fig
    