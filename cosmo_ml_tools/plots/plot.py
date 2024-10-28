import numpy as np

def plot_fill_between(x : np.ndarray, samples_y: np.ndarray, label: str = None,
                      ax = None, color: str = 'gray', lw: float = 2., alpha:float = 0.5, quantiles:list=[2.3, 16, 50, 84, 97.7]):
    """
    Plot median and quantiles for a given (flatten) array of samples

    Args:
        x (np.ndarray): an array with x-values as a numpy array.
        samples_y (np.ndarray): a numpy array with samples for the quantity f=y(x).
        label (str, optional): labels to use in the legend. Defaults to None.
        ax (_type_, optional): a matplotlib axes instance. If None, will create a single plot with default settings. Defaults to None.
        color (str, optional): color for the contours. Defaults to 'gray'.
        lw (float, optional): length-width for the lines. Defaults to 2..
        alpha (float, optional): transparency of the contour colors. Defaults to 0.5.
        quantiles (list, optional): quantiles of the distribution to plot. Defaults to [2.3, 16, 50, 84, 97.7].
    """
    if ax is None:
        try: 
            import matplotlib.pyplot as plt
        except ModuleNotFoundError: 
            print('Cannot import matplotlib. Try installing matplotlib before!')
        fig,ax=plt.subplots()
        
    qs = np.percentile(samples_y, q=quantiles, axis=0)
    idx = len(qs) // 2
    median = qs[idx]
    for i in range(1, idx + 1):
        ax.fill_between(x.flatten(), qs[idx - i].flatten(), qs[idx + i].flatten(), color=color, lw=lw, alpha=alpha / i)
    ax.plot(x, median, label=label, c=color, ls='-', lw=lw)
    
    
def plot_colorcoded_y(x:np.ndarray,samples_y:np.ndarray,samples_z:np.ndarray,fig=None,
                       nsamples:int=75,seed:int|None = None,legend:bool=False,colorbar=False,
                       quiet:bool=True,cbar_lbl:str=r'$-\ln{\mathcal{L}}$',
                       alpha:float=0.5,lw:float=0.5):

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
  
  for ax in fig.get_axes():
    ax.add_collection(line_segments)
  
  fig.subplots_adjust(left=.09,right=.89,top=.9,wspace=.09,hspace=.08)
  if colorbar:
    cbar_ax = fig.add_axes([0.92,0.11,0.02,0.79])
    axcb = fig.colorbar(line_segments,cax=cbar_ax)
    axcb.set_label(cbar_lbl,fontsize='xx-large')
  
  if legend:
    ax.legend()
    
