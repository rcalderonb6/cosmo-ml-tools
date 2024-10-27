import numpy as np
import matplotlib.pyplot as plt

def plot_alphas(x:np.ndarray,alphas:list[np.ndarray],axs=None):
    if axs is None:
        n=len(alphas)
        shape=(n,1) if n==2 else ()
        fig,axs=plt.subplots(*shape,sharex=True,sharey='row')
    
    for ax,alpha in zip(axs.flatten(),alphas):
        ax.plot(x,alpha)