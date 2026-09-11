"""Illustrate the two fixed input transforms; no learned or test-set quantities."""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    z=np.linspace(-4,4,801)
    clipped=np.clip(z,-3,3)
    centers=np.linspace(-3,3,16)
    gaussian=np.exp(-.5*((clipped[None,:]-centers[:,None])/.4)**2)
    repeat=np.repeat(((clipped+3)/6)[None,:],16,axis=0)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for column,(name,values) in enumerate((('Gaussian tuning',gaussian),('No tuning: repeated scalar',repeat))):
        axes[0,column].plot(z,values.T,linewidth=1)
        axes[0,column].set_title(name)
        axes[0,column].set_ylabel('Direct current response')
        axes[0,column].set_ylim(-.05,1.05)
        axes[0,column].grid(alpha=.2)
        axes[1,column].imshow(values,aspect='auto',origin='lower',extent=(-4,4,-.5,15.5),vmin=0,vmax=1,cmap='viridis')
        axes[1,column].set_ylabel('Population slot k')
        axes[1,column].set_xlabel('Normalized scalar z')
        for row in (0,1):
            for bound in (-3,3):
                axes[row,column].axvline(bound,color='gray',linestyle='--',linewidth=.8)
    axes[0,1].text(-3.7,.88,'All 16 responses overlap',fontsize=9)
    fig.suptitle('Same clipping, [0,1] range and 16 slots; only Gaussian has slot selectivity')
    fig.tight_layout(rect=(0,0,1,.95))
    root=Path(__file__).resolve().parent/'results/ett-population-ablation-20260911'
    fig.savefig(root/'input_transforms.png',dpi=180)
    fig.savefig(root/'input_transforms.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()
