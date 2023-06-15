
import pandas as pd
import json
from pathlib import Path
from constants.constants import file_tas, file_sherwood, file_zelinka


def extract_zelinka():
    """Read CMIP6 Effective Climate Sensitivity (ECS) from Zelinka.
    
    Returns
    -------
    xr.DataFrame
      Effective Climate Sensitivity indexed by model name.
    
    References
    ----------
    Zelinka, M. D., Myers, T. A., McCoy, D. T., Po-Chedley, S., Caldwell, P. M., Ceppi, P., et al. (2020). 
    Causes of higher climate sensitivity in CMIP6 models. Geophysical Research Letters, 47, e2019GL085782. 
    https://doi.org/10.1029/2019GL085782
    
    Links to data: 
    
      - https://github.com/mzelinka/cmip56_forcing_feedback_ecs
      - https://zenodo.org/record/6647291#.Y4Eb39LMJhE
    
    """

    with open("data/cmip56_forcing_feedback_ecs.json", "r") as f:
        doc = json.load(f)
    c6 = doc["CMIP6"]

    ecs = {}
    for (model, values) in c6.items():
        # Take the first realization from the list
        ecs[model] = list(values.values())[0]["ECS"]

        
    ecs = pd.Series(ecs).to_frame("ECS")
    
    return ecs


def write_zelinka():
    """Write Zelinka ECS estimates to disk."""
    ecs = extract_zelinka()
    ecs.to_json(file_zelinka)
    
    
def load_zelinka():
    """Load Zelinka ECS data."""
    out = pd.read_json(file_zelinka)
    out.index.name = "Model"
    return out


import numpy as np

def extract_sherwood():
    """Extract posterior probability density function (pdf) from Sherwood's supplementary material.
    
    Returns
    -------
    xr.DataFrame
      Posterior pdf for the effective climate sensitivity.
    
    
    References
    ----------
    Webb, M. (2020). Code and Data for WCRP Climate Sensitivity Assessment. 
    https://doi.org/10.5281/ZENODO.3945276 
    
    """
    from joblib import load
    
    def _kernel_smooth(bin_centres,y,bin_width, n_bins):

      # apply Gausian Kernel smoothing to y
      # JDA uses sd of 0.1
      kernel_sd = 0.1

      smoothed_y=np.copy(y)

      # apply kernel filter over central range
      istart=int(n_bins/2-4000)
      iend=int(n_bins/2+4000)

      smooth=True
      if smooth:
        for i in range(istart,iend):
          x=bin_centres[i]
          k = np.exp(-1*( x - bin_centres ) ** 2 / (2 * kernel_sd ** 2))
          smoothed_y[i] = np.sum(y * k)
          #print ('i=',i,'bin_center=',bin_centres[i],x,y[i],smoothed_y[i])

      smoothed_y=_normalise_pdf(smoothed_y,bin_width)

      return(smoothed_y)

    def _normalise_pdf(pdf,bin_width):
      return(pdf/np.sum(pdf, dtype=np.float64)/bin_width)

    
    # Load data
    inpath = "/home/david/projects/HQ/avis_scenarios_2140/data/WCRP_ECS_assessment_code_200714"
    calc_id = "ULI_MEDIUM_SAMPLE"
    dumpfile = inpath + '/' + calc_id + '/' + calc_id + '.lastmean.joblib'
    [transfer_unweighted_prior_pdf, transfer_weighted_prior_pdf, total_hist_erf_posterior, total_hist_erf_prior,
     ecs_pdf, posterior, n_bins, bin_boundaries, bin_centres, bin_width, n_samples, s_pdf, s_prior_pdf,
     full_l_prior_pdf, l_process_bu_likelihood, l_process_ec_likelihood, l_hist_likelihood, l_paleo_cold_likelihood,
     l_paleo_hot_likelihood, l_prior, l_posterior, s_process_bu_likelihood, s_process_ec_likelihood, s_hist_likelihood,
     s_paleo_cold_likelihood, s_paleo_hot_likelihood, full_s_prior_pdf] = load(dumpfile)

    x = np.array(bin_centres)
    i = (x < 8) * (x > 0)
    
    # Normalize and smooth posterior
    posterior = _normalise_pdf(posterior, bin_width)
    smoothed_posterior=_kernel_smooth(bin_centres,posterior,bin_width, n_bins)
    cdf = np.cumsum(smoothed_posterior) * bin_width
    
    out = pd.DataFrame({"ECS": x[i], "pdf": smoothed_posterior[i], "cdf": cdf[i]})
    return out
    
    
def write_sherwood():
    df = extract_sherwood()
    df.to_json(file_sherwood)
    
    
def load_sherwood():
    out = pd.read_json(file_sherwood)
    return out
def load_global_tas():
    out = pd.read_csv(file_tas)
    return out