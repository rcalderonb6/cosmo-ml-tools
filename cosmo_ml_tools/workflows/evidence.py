"""
Compute the Bayesian Evidence from MCMC samples using 
the Harmonic Mean estimator and Normalizing Flows.
We use the python package Harmonic and this basic 
example workflow heavily relies on their tutorial.
"""
import harmonic as hm
    
def compute_evidence(samples, ndim :int , 
         model: str = 'NVP', N_chains=4,epochs_num=20,
         temperature=0.7,training_proportion=0.5, verbose=True,
         **hm_kwargs
         ):
    
    # Convert the samples into a harmonic-friendly shape and get posteriors
    samples, lnprob = samples.to_harmonic()
    
    # Instantiate harmonic's chains class
    chains = hm.Chains(ndim)
    chains.add_chains_3d(samples, lnprob)
    
    # Split the chains into the ones which will be used to train the machine
    # learning model and for inference
    chains_train, chains_infer = hm.utils.split_data(chains, training_proportion=training_proportion)

    if model=='NVP':
        model = hm.model.RealNVPModel(ndim,standardize=True,temperature=temperature,**hm_kwargs)
    else:
        print('Model not yet implemented')
        
    # Train model
    model.fit(chains_train.samples, epochs=epochs_num, verbose=verbose)
    
    samples = samples.reshape((-1, ndim))
    samp_num = samples.shape[0]
    flow_samples = model.sample(samp_num)
    hm.utils.plot_getdist_compare(samples, flow_samples)
    
    # Instantiate harmonic's evidence class
    ev = hm.Evidence(chains_infer.nchains, model)

    # Pass the evidence class the inference chains and compute the evidence!
    ev.add_chains(chains_infer)
    ln_inv_evidence = ev.ln_evidence_inv
    err_ln_inv_evidence = ev.compute_ln_inv_evidence_errors()
    
    print(f'ln inverse evidence (harmonic) = {-ln_inv_evidence} +/- {err_ln_inv_evidence}')
    return -ln_inv_evidence

# from sampler.pocomc import PocoMCBase

# def evidence_with_pocomc(info_yaml:str):
#     class Sampler(PocoMCBase):
        
#         def log_likelihood(self,):
#             # self.
    
    
#     return

if __name__=='__main__':
    
    compute_evidence()