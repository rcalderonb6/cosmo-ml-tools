def get_latex_table(samples,parameters,param_labels=None):
    """
    Get a latex table with mean and 68% credible intervals constraints for a given set of chains and cosmological parameters.

    samples: dictionary, a dictionary with getdist instances with the chains. 
    The corresponding dictionary keys are used as labels.
    
    parameters: list, a list with parameter names you want to include in the table
    
    param_labels: list (optional), a list with latex names for each of the requested parameters. 
    If none, its name on the chain is used.
    
    """
    Nparams=len(parameters)
    cols='l'+'c' * Nparams

    print(r'\begin{table*}[t]')
    print(r'\caption{68\% credible intervals for the cosmological parameters,\
        using various dataset combinations.','\n',r'\vspace{0.5em}}')
    print(r'\label{tab:tab_label}')
    print(r'\centering')
    print(r'\small')
    print(r'\resizebox{0.95\textwidth}{!}{')
    print(r'\begin{tabular}'+r'{%s}'%cols)
    print(r'\toprule')
    print(r'\toprule')
    line='Dataset'
    params=parameters if param_labels is None else param_labels
    for p in params:
        line+=f' & {p}'
    print(line+r' \\')
    print(r'\midrule[1.5pt]')
    for lbl,chain in samples.items():
        line=f'{lbl}'   
        for p in parameters:
            stats=chain.getInlineLatex(p,limit=1).split('=')[1].strip()
            line+=f' & ${stats}$ '
        print(line+r' \\')
        if lbl != list(samples.keys())[-1]: print(r'\midrule')
    print(r'\toprule')
    print(r'\toprule')
    print(r'\end{tabular}}')
    print(r'\end{table*}')