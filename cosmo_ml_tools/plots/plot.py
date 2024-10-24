import numpy as np

def plot_fill_between(x : np.ndarray, samples : np.ndarray, label:str = None,
                      ax=None,color:str='gray', lw=2., alpha=0.5, quantiles:list=[2.3, 16, 50, 84, 97.7]):
    """Plot median and +/- 2 sigma regions for a given (flatten) array of samples"""
    
    if ax is None:
        try: 
            import matplotlib.pyplot as plt
        except ModuleNotFoundError: 
            print('Cannot import matplotlib. Try installing matplotlib before!')
        fig,ax=plt.subplots()
        
    qs = np.percentile(samples, q=quantiles, axis=0)
    idx = len(qs) // 2
    median = qs[idx]
    for i in range(1, idx + 1):
        ax.fill_between(x.flatten(), qs[idx - i].flatten(), qs[idx + i].flatten(), color=color, lw=lw, alpha=alpha / i)
    ax.plot(x, median, label=label, c=color, ls='-', lw=lw)
    
    
def plot_samples_y_lkl(x:np.ndarray,samples_y:np.ndarray,samples_z:np.ndarray,
                       nsamples:int=75,seed:int|None = None,
                       quiet:bool=True,cbar_lbl:str=r'$-\ln{\mathcal{L}}$',alpha:float=0.5,lw:float=0.5):

  from matplotlib.collections import LineCollection
  if seed is not None:
    random_state = np.random.default_rng(10)
    ind=random_state.integers(low=0,high=len(samples_y), size=nsamples)
  else:
    ind=np.random.randint(len(samples_y),size=nsamples)
    
  ys=samples_y[ind]

  # Make a sequence of (x, y) pairs
  line_segments= LineCollection([np.column_stack([x, y]) for y in ys],linestyles='solid',lw=lw,alpha=alpha)
  line_segments.set_array(samples_z[ind])

  ax.add_collection(line_segments)
  fig.subplots_adjust(left=.09,right=.89,top=.9,wspace=.09,hspace=.08)
  cbar_ax = fig.add_axes([0.92,0.11,0.02,0.79])
  axcb = fig.colorbar(line_segments,cax=cbar_ax)
  axcb.set_label(cbar_lbl,fontsize='xx-large')

  ax.legend()
  # plt.sci(line_segments)# This allows interactive changing of the colormap.
  if not quiet:
      return fig,ax
    
  ''' 
  Plot n random samples from the MCMC chains. 
  Must input:
  - z is the array of values where the samples of fde are computed
  - samples_fde : a (flattened) array with samples of fde from the chains and corresponding likelihoods.
  - samples_z : an array with the corresponding likelihood values
  - nsamples:  The number of samples to plot from the chains (default: randomly selected
  - figax = (fig,ax) is a tupple to specify the figure and axes 
  - seed : specifies the seed for reproducibility (i.e. drawing the same samples)
  '''