
import numpy as np
import matplotlib.pyplot as plt
from .base import DoubleColumnBase,PlotBase

class PrimordialFeatures(PlotBase):
    
    def __init__(self,format,cosmo=None,output_dir=None):
        super().__init__(format=format,cosmo=cosmo,output_dir=output_dir)
        
    def plot_Pk(self,models_Pk):
        
        LCDM_Pk=self.cosmo.get_Pk(units='Mpc/h')
        
        y_labels=[r'$\mathcal{P}_\zeta(k)$',r'$P_m(k,z)$','\% diff.']
        CMB_scales=(0.0002,0.14)
        LSS_scales=(0.001,0.24)
        cmaps=[plt.cm.Blues_r(np.linspace(0.,0.8,len(y_labels))),plt.cm.Oranges_r(np.linspace(0.,0.8,len(y_labels)))]
        
        # self.axes[0,0].plot(models[0]['kh'],LogFeat(models[0]['kh']),c=colors_mod1[0])
        # self.axes[0,1].plot(test_Pk.T[0],test_Pk.T[1],c=colors_mod2[0])
        
        for i,model in enumerate(models_Pk):
            axs=self.axes[:,i]
            # axs[0].text(1.5e-4,1.9e-9,s=frequencies[i],fontsize='large')
            # axs[0].plot(model['kh'],LogFeat(model['kh'],Alog=0),ls='--',c='k')
            axs[0].set_ylim(1.75e-9,2.8e-9)
            
            # for (k,Pk),c in zip(model['Pk'].items(),cmaps[i]):
                # axs[1].plot(model['kh'],Pk,label=k,c=c)
                # axs[1].legend(frameon=True,ncol=3,fontsize='large')
                # axs[2].plot(model['kh'],(Pk/LCDM_Pk[k]-1)*1e2,label=k,c=cmaps[i][0])

        for ax,lbl in zip(self.axes[:,0],y_labels):
            ax.set_ylabel(lbl,fontsize='xx-large')
        
        for ax in self.axes[-1,:]:
            ax.set_xlabel(r'$k \; [h/\rm{Mpc]}$',fontsize='xx-large')
            ax.set_xlim(1e-4,1.1)
        
        [ax.loglog() for ax in self.axes[1,:]]
        
        for ax in self.axes.flatten():
            ax.axvspan(*CMB_scales, color='lightgray', alpha=0.3)
            ax.axvspan(*LSS_scales, color='teal', alpha=0.1)
            
        return self.fig, self.axes

if __name__=='__main__':
    
    plt.style.use(['science','high-contrast'])
    
    from format import DoubleColumn
    format=DoubleColumn(3,3)
    format.fig_kwargs={'sharex':True,'sharey':'row','gridspec_kw':{'hspace':0.03,'wspace':0.02}}

    # from .cosmology import ClassEngine
    
    # cosmo=ClassEngine()
    # print(cosmo.Cls)
    f=PrimordialFeatures(format=format)
    fig,axs=f.plot_Pk()
    for ax in axs.flatten():
        ax.set_xlim(8e-5,1.5)
    plt.show()
    
    fig.savefig('_test.pdf')
    # f.fig.show()