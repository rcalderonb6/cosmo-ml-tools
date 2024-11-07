from dataclasses import dataclass

@dataclass
class FullPage:
    rows:int = 1
    cols:int = 1
    height:float = 6
    width:float = 8
    fontsize:int = 20
    fig_kwargs = {'tight_layout':True}


@dataclass    
class SingleColumn:
    rows:int = 1
    cols:int = 1
    height:float = 4
    width:float = 4
    fontsize:int = 12
    fig_kwargs = {'tight_layout':True}
    
@dataclass    
class DoubleColumn:
    rows:int = 1
    cols:int = 1
    height:float = 4
    width:float = 8
    fontsize:int = 16   
    fig_kwargs = {'tight_layout':True}

    
@dataclass
class Presentation:
    rows:int = 1
    cols:int = 1
    height:float = 4
    width:float = 8
    fontsize:int = 16   
    fig_kwargs = {'tight_layout':True}
    
    
@dataclass    
class JCAP:
    rows:int = 1
    cols:int = 1
    height:float = 4
    width:float = 8
    fontsize:int = 16   
    fig_kwargs = {'tight_layout':True}

    
@dataclass    
class PRD:
    rows:int = 1
    cols:int = 1
    height:float = 4
    width:float = 8
    fontsize:int = 16   
    fig_kwargs = {'tight_layout':True}
