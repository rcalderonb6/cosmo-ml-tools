from dataclasses import dataclass, field
from abc import ABC,abstractmethod
import matplotlib.pyplot as plt
from typing import Optional
from .format import *

class PlotBase:

    def __init__(self,format,cosmo=None,output_dir:str=None):
        self.format = format
        self.cosmo = cosmo
        self.fig, self.axes= self.create_fig()
        self.out_dir= '' if output_dir is None else output_dir
        
    def create_fig(self):
        return plt.subplots(self.format.rows,self.format.cols,
                            figsize=self.size,
                            **self.format.fig_kwargs)
    
    def export(self,fname):
        self.fig.savefig(self.out_dir+fname)
    
    def save(self,fname):
        self.fig.savefig(self.out_dir+fname)
    
    def show(self):
        self.fig.show()
        
    @property
    def size(self):
        return (self.format.width,self.format.height)

class SingleColumnBase(PlotBase):
    
    def __init__(self,format=SingleColumn,cosmo=None,output_dir:str=None):
        super().__init__(format,cosmo,output_dir)

class DoubleColumnBase(PlotBase):
    
    def __init__(self,format=DoubleColumn,cosmo=None,output_dir:str=None):
        super().__init__(format,cosmo,output_dir)
        
if __name__=='__main__':
    t1=SingleColumnBase()
    t2=DoubleColumnBase()
    for t in [t1,t2]:
        print(t.create_fig())
        t.show()