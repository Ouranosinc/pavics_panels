import pandas as pd
import json

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
    ecs.to_json("zelinka_ecs.json")
    
    
def load_zelinka():
    """Load Zelinka ECS data."""
    out = pd.read_json("zelinka_ecs.json")
    out.index.name = "Model"
    return out
    
# write_zelinka()
ecs = load_zelinka()


# write_sherwood()
df = load_sherwood()
#print(df.reindex())

from matplotlib import pyplot as plt
    
def graph_baseline_ecs_pdf():
    """Plot Baseline posterior from Sherwood's paper."""

    df = pd.read_json("sherwood_ecs.json").reindex()
    pdf = df["pdf"]
    x = df["ECS"]
    cdf = df["cdf"]
    
    ac = "#36494f"
    ac2 = "orange"
    with plt.rc_context(
            {'axes.edgecolor': ac, 'axes.labelcolor': ac, 'xtick.color': ac, 'ytick.color': ac, 'figure.facecolor':
                'white'}):

        fig, ax = plt.subplots(1, 1, figsize=(6.5, 3), dpi=300)
        fig.subplots_adjust(bottom=.15)
        l1 = ax.plot(x, pdf, color="k", label="Densité de probabilité")
    
    with plt.rc_context(
            {'axes.labelcolor': ac, 'xtick.color': ac, 'ytick.color': ac2, }):
        ax2 = ax.twinx()
        l2 = ax2.plot(x, 1-cdf, color=ac2, label="Probabilité de dépassement", clip_on=False)
    
    lns = l1+l2
    labs = [l.get_label() for l in lns]
    #ax.legend(lns, labs, frameon=False)

    ax.set_xlim([0,8])
    ax.set_ylim([0, .8])
    ax2.set_xlim([0, 8])
    ax2.set_ylim([0, 1])
        
    for axi in [ax, ax2]:
        for key, spine in axi.spines.items():
            if key in ["top"]:
                spine.set_visible(False)
                
    ax.set_xlabel("Sensibilité climatique effective (K)")
    ax.set_ylabel("Densité de probabilité (K$^{-1}$)")
    ax2.set_ylabel("Probabilité de dépassement")
    
    return fig
    context = """# Context

The CMIP6 ensemble has what is called a *hot model* problem ([Hausfather et al. (2022)](https://doi.org/10.1038/d41586-022-01192-2)). That is, many models have a high climate sensitivity, and taking an unweighted average of temperature changes from the model ensemble would yield a warming higher than best estimates based on multiple historical and paleoclimate observations. This dashboard presents various strategies to weigh CMIP6 models based on climate sensivity estimates. 

## Measures of climate sensitivity

Climate sensitivity is usually defined as the global temperature increase following a doubling of CO2 concentration in the atmosphere, compared to pre-industrial levels (~260 ppm).


Transient Climate Response
: The temperature change at the moment that atmospheric CO2 has doubled in a scenario where CO2 increases at a rate of 1% each year (about 70 years).

Equilibrium Climate Sensitivity
: The temperature change once the climate has fully adjusted to a doubling of atmospheric CO2, so after thousands oy years to account for the slow response of oceans.

Effective Climate Sensitivity
: An approximation of the equilibrium climate sensitivity found by analysing the first 150 years of an abrupt 2xCO2 or 4xCO2 simulation, assuming linear climate feedbacks."""

sources = """
# Data sources

CMIP-6 model ECS
: Mark Zelinka. (2022). mzelinka/cmip56_forcing_feedback_ecs: Jun 15, 2022 Release (v2.2). Zenodo. https://doi.org/10.5281/zenodo.6647291

Expected ECS
: Sherwood, S. C., Webb, M. J., Annan, J. D., Armour, K. C., Forster, P. M., Hargreaves, J. C., et al. (2020). An assessment of Earth's climate sensitivity using multiple lines of evidence. Reviews of Geophysics, 58, e2019RG000678. https://doi.org/10.1029/2019RG000678

Hausfather Likely/Very likely range
: Hausfather, Z., Marvel, K., Schmidt, G. A., Nielsen-Gammon, J. W., & Zelinka, M. (2022). Climate simulations: Recognize the ‘hot model’problem. https://doi.org/10.1038/d41586-022-01192-2

IPCC likely/very likely range
: IPCC, 2021: Summary for Policymakers. In: Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report of the Intergovernmental Panel on Climate Change [Masson-Delmotte, V., P. Zhai, A. Pirani, S.L. Connors, C. Péan, S. Berger, N. Caud, Y. Chen, L. Goldfarb, M.I. Gomis, M. Huang, K. Leitzell, E. Lonnoy, J.B.R. Matthews, T.K. Maycock, T. Waterfield, O. Yelekçi, R. Yu, and B. Zhou (eds.)]. In Press.


## Note

The IPCC and Hausfather use Equilibrium Climate Sensitivity, which is not available for all models. We multiply these ranges by a factor of {mfact}% to obtain the Effective Climate Sensitivity, as per Sherwood et al, although we neglect the uncertainty in this adjustment factor. Alternatively, you can take account of the uncertainty by multiplying by the appropriate distribution. This is done on the 'equilibrium' tab.
"""

pdf = load_sherwood().reset_index()
models = load_zelinka().reset_index()
models.sort_values("ECS", inplace=True)
pass

import holoviews as hv
import colorcet as cc
import hvplot.xarray # gives hvplot method to pandas objects
import hvplot.pandas
from scipy import stats
from scipy import interpolate
from scipy import integrate
import numpy as np
hv.extension('bokeh')
from bokeh.models import tools
import panel as pn

# get an generator for the kernel density estimate for the sherwood models:
#
#kde_stats = stats.gaussian_kde(models.ECS, bw_method = BW_METHOD)
## get the density estimate, and add it to the models df:
#models['density'] = kde_stats.pdf(models.ECS)
## get the density estimate for all ECSs in pdf.ECS, using the generator for models. 
#kde_data = pd.DataFrame({'ECS':pdf.ECS,'PDF':kde_stats.pdf(pdf.ECS)})
## get the transform to go from the sherwood PDF to the Zelinka pdf:
#kde_data['weights'] = pdf.pdf / kde_data.PDF
## get a generator to interpolate these weights:
#weightfunc = interpolate.interp1d(kde_data.ECS,kde_data.weights)
## get the weights for each model:
#models['weight'] = weightfunc(models.ECS)
## get the new_density for each model (equivalent to running gaussian_kde with weights=models.weight):
#models['new_density'] = models.density * models.weight
def quantile_1D(data, weights, quantile):
    """
    Compute the weighted quantile of a 1D numpy array. from https://github.com/nudomarinero/wquantiles/blob/master/wquantiles.py
    Parameters
    ----------
    data : ndarray
        Input array (one dimension).
    weights : ndarray
        Array with the weights of the same size of `data`.
    quantile : float
        Quantile to compute. It must have a value between 0 and 1.
    Returns
    -------
    quantile_1D : float
        The output value.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if ((quantile > 1.) or (quantile < 0.)):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    Sn = np.cumsum(sorted_weights)
    # TODO: Check that the weights do not sum zero
    #assert Sn != 0, "The sum of the weights must not be zero"
    Pn = (Sn-0.5*sorted_weights)/Sn[-1]
    # Get the value of the weighted median
    return np.interp(quantile, Pn, sorted_data)

class ModelData():
    ''' class to hold model data
    '''
    models   = None # for model data
    sherwood = None # for sherwood data
    filtered = None # for filtered model data
    interp   = None # for interpolated filtered data to sherwood ECS
    x_sel    = [0,10.0] # filter to apply.
    is_updated = False
    weighted   = False
    bw_method = None
    adjustment_factor = 0.06
    adjustment_scale  = 0.2
    def get_eq(self,obj):
        ecs_x = obj.ECS
        pdf_x = obj.PDF
        y = ecs_x[1:]#np.linspace(0.0,10.0,100)
        interpolator_x = interpolate.interp1d(ecs_x,pdf_x,bounds_error=False,fill_value=0)
        interpolator_y = lambda x : stats.norm.pdf(x,loc=(1.+self.adjustment_factor),scale=self.adjustment_scale)
        interpolator_xy = lambda t : integrate.trapz(x=y,y=(1/(y) * interpolator_y(t / y) * interpolator_x(y))) # -inf to inf ideally, 0 to 8 is good enough.
        return [interpolator_xy(z) for z in ecs_x]
    def __init__(self,models,sherwood,lazy=True,weighted = False, bw_method='scott'):
        self.models = models
        self.sherwood  = sherwood
        self.lazy = lazy
        self.bw_method = bw_method
        self.weighted  = weighted
        self.sherwood['PDFeq'] = self.get_eq(self.sherwood)
        
    def set_filter(self,x_sel):
        if not all([self.x_sel[i] == x_sel[i] for i,x in enumerate(x_sel)]):
            self.is_updated = False
        self.x_sel = x_sel
        if (not self.lazy) and (not self.is_updated):
            self.apply_filter()
    
    def apply_filter(self):
        self.filtered = self.models[(self.models.ECS > self.x_sel[0]) & (self.models.ECS < self.x_sel[1])].copy()
        self.update_kde()
        self.is_updated = True
    
    def my_kde(self,data,bw,weights=[]):
        kde = np.zeros(self.sherwood.ECS.shape)
        mean_kernel = integrate.trapz(x=self.sherwood.ECS,y=self.sherwood.ECS * self.sherwood.PDF)
        std_kernel    = np.sqrt(integrate.trapz(x=self.sherwood.ECS,y=(self.sherwood.ECS ** 2.0) * self.sherwood.PDF) - (mean_kernel ** 2))
        centered_kernel = interpolate.interp1d((self.sherwood.ECS - mean_kernel) / std_kernel, self.sherwood.PDF * std_kernel,bounds_error=False,fill_value=0)
        n = len(data) if len(weights) != len(data) else np.sum(weights)**2 / np.sum(weights**2)
        if type(bw) == str:
            if bw == 'scott':
                bw = n ** (-1.0 / 5.)
            elif bw == 'silverman':
                bw = (n * (1. + 2.) / 4.)**(-1. / (1. + 4.))
                
        for i in data.index:
            if len(weights) == len(data):
                weight = weights[i]
            else:
                weight = n ** (-1)
            centre = data[i]
            kde += weight * (1./bw) * centered_kernel((self.sherwood.ECS - centre) / bw)
        return kde
 
    def update_kde(self,update_weights=True):
        if self.weighted and (type(self.filtered) != type(None)) and ('weight' in self.filtered.columns) and not update_weights:
            kde_stats = self.my_kde(self.filtered.ECS,bw = self.bw_method,weights=self.filtered.weight)#stats.gaussian_kde(self.filtered.ECS,, bw_method = self.bw_method)
        else:
            kde_stats = self.my_kde(self.filtered.ECS,bw = self.bw_method,weights=np.array([]))
            #kde_stats = stats.gaussian_kde(self.filtered.ECS,bw_method = self.bw_method)
        self.interp = pd.DataFrame({'ECS':self.sherwood.ECS})
        
        interpolator = interpolate.interp1d(self.sherwood.ECS,kde_stats,bounds_error=False,fill_value=0)
        self.filtered['PDF'] = interpolator(self.filtered.ECS)
        self.interp['PDF'] = interpolator(self.interp.ECS)
        
        if update_weights:
            self.interp['weight'] = self.sherwood.PDF / self.interp.PDF
            weightfunc = interpolate.interp1d(self.interp.ECS,self.interp.weight)
            self.filtered['weight'] = weightfunc(self.filtered.ECS)
            norm_weight_factor = 1.0 / np.sum(self.filtered.weight)
            self.filtered['weight'] *= norm_weight_factor
            if self.weighted:
                self.update_kde(update_weights=False)
    
    def map_quantiles(self,x_in,x_out,from_pdf,to_pdf):
        from_cdf = np.cumsum(from_pdf)
        to_cdf   = np.cumsum(to_pdf)
        
        func_to_cdf = interpolate.interp1d(x_in,to_cdf,kind='cubic',bounds_error=False)
        func_from_cdf = interpolate.interp1d(x_in,from_cdf, kind='cubic',bounds_error=False)
        
        quantiles_from = func_from_cdf(x_out)
        quantiles_to   = func_to_cdf(x_out)
        weights = quantiles_to / (quantiles_from + np.finfo(float).eps)
        return weights
    
    def set_weighted(self,weighted):
        if self.weighted != weighted:
            self.is_updated = False
        self.weighted = weighted
        if not self.lazy:
            self.apply_filter()
    
    def iter_kde(self,max_iter = 10):
        if not self.weighted:
            self.update_kde()
            return
        for i in range(max_iter):
            self.update_kde()
        
    def get_filtered(self):
        if not self.is_updated:
            self.apply_filter()
        return self.filtered
    
    def get_interp(self):
        if not self.is_updated:
            self.apply_filter()
        return self.interp
    
    def get_median(self,dataset,weights):
        return quantile_1D(dataset,weights,0.5)
    
    def get_summarystats(self):
        self.apply_filter()
        summary = pd.DataFrame({
                        ("all, sherwood"):       {"mean":0, 'median':0,'std':0},
                        ("selection, sherwood"): {"mean":0, 'median':0,'std':0},
                        ("selection, models"):   {"mean":0, 'median':0,'std':0},
                       })
        sherwood_filter = (self.sherwood.ECS > self.x_sel[0]) & (self.sherwood.ECS < self.x_sel[1])
        dataset = self.sherwood
        summary.loc['mean',  ('all, sherwood')] = integrate.trapz(x=dataset.ECS,y=dataset.ECS * dataset.PDF)
        summary.loc['median',('all, sherwood')] = self.get_median(dataset.ECS,dataset.PDF)
        summary.loc['std',   ('all, sherwood')] = np.sqrt(integrate.trapz(x=dataset.ECS,y=(dataset.ECS ** 2.0) * dataset.PDF) - (summary.loc['mean',  ('all, sherwood')] ** 2))

        dataset = self.sherwood[sherwood_filter]
        summary.loc['mean',  ('selection, sherwood')] = integrate.trapz(x=dataset.ECS,y=dataset.ECS * dataset.PDF)
        summary.loc['median',('selection, sherwood')] = self.get_median(dataset.ECS,dataset.PDF)
        summary.loc['std',   ('selection, sherwood')] = np.sqrt(integrate.trapz(x=dataset.ECS,y=(dataset.ECS ** 2.0) * dataset.PDF) - (summary.loc['mean',  ('selection, sherwood')] ** 2))
        if self.weighted:
            summary.loc['mean',  ('selection, models')] = (self.filtered.ECS * self.filtered.weight).sum()
            summary.loc['median',('selection, models')] = self.get_median(self.filtered.ECS,self.filtered.weight)
            summary.loc['std',   ('selection, models')] = ((self.filtered.ECS - summary.loc['mean',  ('selection, models')])**2 * self.filtered.weight).sum()
        else:
            summary.loc['mean',  ('selection, models')] = self.filtered.ECS.mean()
            summary.loc['median',('selection, models')] = self.filtered.ECS.median()
            summary.loc['std',   ('selection, models')] = self.filtered.ECS.std()
        return summary
    
    def get_prob(self, distr='sherwood'):
        ''' returns the probability for the ECS to be within the x_sel bounds.
        '''
        dataset = None
        if distr.lower() == 'sherwood':
            sherwood_filter = (self.sherwood.ECS >= self.x_sel[0]) & (self.sherwood.ECS <= self.x_sel[1])
            dataset = self.sherwood[sherwood_filter]
        elif (distr.lower() == 'models') or (distr.lower() == 'interp'):
            if not self.is_updated:
                self.apply_filter()
            dataset = self.interp
        else:
            raise ValueError(f'Cannot interpret distr={distr}. use sherwood or models/interp.')
        probability = integrate.trapz(x=dataset.ECS,y=dataset.PDF)
        return probability

BW_METHOD = 'scott'
model_obj = ModelData(models=models.rename({'density':'PDF'},axis='columns',inplace=False),
                      sherwood=pdf.rename({'pdf':'PDF'},axis='columns',inplace=False),
                      weighted = False,
                      lazy = True,
                      bw_method = BW_METHOD)
#print(model_obj.get_filtered().head())
model_obj.set_filter([0,10])
model_obj.set_weighted(True)
model_obj.apply_filter()
#print(model_obj.get_filtered().head())
#model_obj.iter_kde()
#print(model_obj.get_filtered().head())

def remove_tools(plot, element):
    '''remove_tools: removes unneeded (and unwanted) tools from the given bokeh plot.
    '''
    curr_tools = plot.state.tools
    new_tools = []
    exclude_list=['LassoSelectTool','PanTool']
    for tool in curr_tools:
        if not any([(exclude in str(type(tool))) for exclude in exclude_list]):
            new_tools.append(tool)
    plot.state.tools = new_tools

def fix_dims(plot, element):
    '''fix_dims: sets given bokeh plot box_select tool to only allow width (x) selections.
    '''
    dims_list = ['BoxSelectTool']
    for tool in plot.state.tools:
        if any([(boxtool in str(type(tool))) for boxtool in dims_list]):
            tool.dimensions = 'width'
            

def display_event(data={}):
    ''' Event triggered whenever selection changes. 
    Effects: 
        1- display new selection bounds
        2- display probability of selection
        3- display number of models selected
        4- updates json of models selected, and displays this
        5- updates df of models selected, and displays this.
    '''
    if data['type'] == 'Effective':
        x1str = 'x1_eff'
        x2str = 'x2_eff'
    else:
        x1str = 'x1_eq'
        x2str = 'x2_eq'
    x_sel = [data[x1str],data[x2str]]
    x1 = data[x1str]
    x2 = data[x2str]
    weighted = data['weighted']
    model_obj.set_filter(x_sel=x_sel)
    model_obj.set_weighted(weighted=weighted)
    df = model_obj.get_filtered()
    summary = model_obj.get_summarystats()
    
    text_widget.object = f"""ECS bounds: {np.round(x1,2)}°C, {np.round(x2,2)}°C
    prob (Sherwood) = {np.round(100 * model_obj.get_prob('sherwood'),2)} %
    number of models: {df.shape[0]} / {model_obj.models.shape[0]}
    """
    summary_stats.object = summary
    json_widget.object = df.to_json()
    df_widget.object = df
    
def save_notebook(event):
    ''' event triggered whenever button "save to jupyterlab" is pressed.
    Effects:
    1- Saves the json to the jupyterlab environment, with filename given in filename_widget.
    2- points the button save_widget_computer to the new filename.
    '''
    # update the filename for save_widget_computer:
    save_widget_computer.filename = get_filename()
    # dump json to file:
    json_str = json_widget.object
    with open(get_filename(), 'w') as outfile:
        outfile.write(json_str)
        
def get_filename():
    ''' wrapper to return filename_widget.value
    '''
    return filename_widget.value

def make_rectangle_eff(data=[]):
    return make_rectangle(data={'x1':data['x1_eff'], 'x2':data['x2_eff']})

def make_rectangle_eq(data=[]):
    return make_rectangle(data={'x1':data['x1_eq'], 'x2':data['x2_eq']})

def make_rectangle(data=[]):
    x1str = 'x1'
    x2str = 'x2'
    rect = hv.Rectangles([(data[x1str],size_rect[1],data[x2str],size_rect[3])])
    rect.opts(alpha = 0.1)
    if data[x2str] < data[x1str]:
        rect.opts(color='red')
    else:
        rect.opts(color='cyan')
    return rect
def make_kde_eq(data=[]):
    model_obj.set_filter(x_sel=[data['x1_eff'],data['x2_eff']])
    model_obj.set_weighted(weighted=data['weighted'])
    
    ecs = model_obj.get_interp().ECS
    pdf = model_obj.get_eq(obj=model_obj.get_interp()) # interpolated PDF of N(mu,sigma)*interp
    weighted_str = "" if not data['weighted'] else "Weighted "
    label = f'{weighted_str}Experimental Density (CMIP-6)'
    return pd.DataFrame({'ECS':ecs,'PDF':pdf}).hvplot.line(x='ECS',y='PDF',label=label)

def make_kde_eq_scatter(data=[]):
    model_obj.set_filter(x_sel=[data['x1_eff'],data['x2_eff']])
    model_obj.set_weighted(weighted=data['weighted'])
    
    interp = model_obj.get_interp()
    
    all_data = model_obj.models
    eq = model_obj.get_eq(obj=interp) # interpolated PDF of N(mu,sigma)*interp

    mu_eq = (1. + model_obj.adjustment_factor) * all_data.ECS
    sigma_eq = all_data.ECS * model_obj.adjustment_scale
    
    pdf_mu_eq = np.interp(xp=interp.ECS,fp=eq,x=mu_eq)
    
    weighted_str = "" if not data['weighted'] else "Weighted "
    label = f'{weighted_str}Selected models'
    selector = (mu_eq < data['x2_eq']) & (mu_eq > data['x1_eq'])
    sel_interp = pd.DataFrame({'ECS':mu_eq[selector], 'PDF':pdf_mu_eq[selector]})
    all_interp = pd.DataFrame({'ECS':mu_eq,'PDF':pdf_mu_eq})
    label_all = f'{weighted_str}CMIP-6 models'
    select_plot = sel_interp.hvplot.scatter(x='ECS',y='PDF',label=label, color = 'red', alpha = 1)
    all_plot    = all_interp.hvplot.scatter(x='ECS',y='PDF',label=label_all, color = 'maroon', alpha = 1)
    
    all_interp['xerr'] = sigma_eq
    errors = hv.ErrorBars(data=all_interp,horizontal=True,vdims=['PDF','xerr'],label='Error bars on EffCS to EqCS conversion')
    return errors * all_plot * select_plot
        
def make_kde(data = []):
    model_obj.set_filter(x_sel=[data['x1_eff'],data['x2_eff']])
    model_obj.set_weighted(weighted=data['weighted'])
    
    weighted_str = "" if not data['weighted'] else "Weighted "
    label = f'{weighted_str}Experimental Density (CMIP-6)'
    return model_obj.get_interp().hvplot.line(x='ECS',y='PDF', label=label)

def make_kde_scatter(data = []):
    model_obj.set_filter(x_sel=[data['x1_eff'],data['x2_eff']])
    model_obj.set_weighted(weighted=data['weighted'])
    weighted_str = "" if not data['weighted'] else "Weighted "
    label_selected = f'{weighted_str}selected models'
    label_unselected = f'{weighted_str}CMIP-6 models'
    scatter_selected = model_obj.get_filtered().hvplot.scatter(x='ECS',y='PDF', hover_cols=['Model','weight'],label = label_selected,color='red', alpha = 1)
    df = model_obj.models.copy()
    df['PDF'] = np.interp(xp=model_obj.interp.ECS,fp=model_obj.interp.PDF,x=df.ECS)
    scatter_unselected = df.hvplot.scatter(x='ECS',y='PDF',label = label_unselected).opts(color = 'maroon', alpha = 1)
    
    return  scatter_unselected * scatter_selected

def convert_eq_to_eff(eq):
    return eq / (1. + model_obj.adjustment_factor)

def convert_eff_to_eq(eff):
    return (1. + model_obj.adjustment_factor) * eff

def pipe_wrapper(*args,**kwargs):
    curr = pipe.data
    if 'type' in kwargs:
        if kwargs['type'] == 'Effective':
            curr['type'] = 'Effective'
        elif kwargs['type'] == 'Equilibrium':
            curr['type'] = 'Equilibrium'
    if 'x_selection' in kwargs:
        if curr['type'] == 'Effective':
            curr['x1_eff'] = kwargs['x_selection'][0]
            curr['x2_eff'] = kwargs['x_selection'][1]
            curr['x1_eq']  = convert_eff_to_eq(curr['x1_eff'])
            curr['x2_eq']  = convert_eff_to_eq(curr['x2_eff'])
        elif curr['type'] == 'Equilibrium':
            curr['x1_eq'] = kwargs['x_selection'][0]
            curr['x2_eq'] = kwargs['x_selection'][1]
            curr['x1_eff']  = convert_eq_to_eff(curr['x1_eq'])
            curr['x2_eff']  = convert_eq_to_eff(curr['x2_eq'])
    if 'x_eff' in kwargs:
        curr['x1_eff'] = kwargs['x_eff'][0]
        curr['x2_eff'] = kwargs['x_eff'][1]
        curr['x1_eq']  = convert_eff_to_eq(curr['x1_eff'])
        curr['x2_eq']  = convert_eff_to_eq(curr['x2_eff'])
    elif 'x_eq' in kwargs:
        curr['x1_eq'] = kwargs['x_eq'][0]
        curr['x2_eq'] = kwargs['x_eq'][1]
        curr['x1_eff']  = convert_eq_to_eff(curr['x1_eq'])
        curr['x2_eff']  = convert_eq_to_eff(curr['x2_eq'])
    if 'weight' in kwargs:
        if kwargs['weight'] == 'Weighted':
            curr['weighted'] = True
        elif kwargs['weight'] == 'Unweighted':
            curr['weighted'] = False
    pipe.send(data=curr)
    
def update_weight(*args):
    if args and type(args[0] == hv.param.Event):
        if args[0].new == 'Weighted':
            pipe_wrapper(weight='Weighted')
        if args[0].new == 'Unweighted':
            pipe_wrapper(weight='Unweighted')

def update_plottype(*args):
    if args and type(args[0] == hv.param.Event):
        if args[0].new == 0:
            pipe_wrapper(type='Effective')
        if args[0].new == 1:
            pipe_wrapper(type='Equilibrium')
            
citation_widget = pn.pane.Markdown(context + sources.format(mfact=np.round(100.0/(1.0+model_obj.adjustment_factor),2)), width=600, extensions=["extra",])

# Widget to display selection

# widget to pre-select IPCC likely, very likely, hausfather:
# note that these are EQUILIBRIUM climate sensitivities. We need to adjust via 1/(1+0.06) [Sherwood et al] to arrive at Effective Climate Sensitivity
def get_range_sherwood():
    quantiles = [66,90]
    ranges = {}
    cdf = integrate.cumtrapz(x=model_obj.sherwood.ECS,y=model_obj.sherwood.PDF,initial=0.)
    cdf_u, ind_u = np.unique(cdf,return_index=True)
    ppf = interpolate.interp1d(x=cdf[ind_u],y=model_obj.sherwood.ECS[ind_u])
    for q in quantiles:
        left = (1. - (q * 1.0 / 100)) / 2
        right = 1 - left
        ranges[q] = (ppf(left),ppf(right))
    return ranges

select_buttons = {
    "IPCC": {
             "Likely":      {'range':(2.5,4), 'button':pn.widgets.Button(name = 'Select IPCC AR6 "Likely" range') },
             "Very likely": {'range':(2, 5), 'button':pn.widgets.Button(name = 'Select IPCC AR6 "Very likely" range') }
            },
    "Hausfather": {
                   "Likely":      {'range':(2.6, 3.9), 'button':pn.widgets.Button(name = 'Select Hausfather "Likely" range') },
                   "Very likely": {'range':(2.3, 4.7), 'button':pn.widgets.Button(name = 'Select Hausfather "Very likely" range') }
                  },
    "Likelyhood": {
                   "Likely":      {'range':get_range_sherwood()[66], 
                                   'button':pn.widgets.Button(name = 'Select Likely (66%) quantiles') },
                   "Very likely" :{'range':get_range_sherwood()[90], 
                                   'button':pn.widgets.Button(name = 'Select Very Likely (90%) quantiles') },
    },
    "all" : {
              "definitely": {'range':( 0, 8), 'button':pn.widgets.Button(name = 'Select all models') } ,
            }
}
preselect_list = pn.Column( name="Select models")
_ = [preselect_list.append(select_buttons[name][prob]['button']) for name in select_buttons for prob in select_buttons[name]]

text_widget = pn.pane.Str("", width=300, height=50)
summary_stats = pn.pane.DataFrame(model_obj.get_summarystats(),width=300)

weight_button = pn.widgets.RadioButtonGroup(name="Weighted PDF", options=["Unweighted", "Weighted"], button_type="primary")

plot_options = pn.Accordion(preselect_list,weight_button, active=[0], sizing_mode='stretch_width')
# Widget to display json of models selected:
json_widget = pn.pane.JSON(models.to_json(), name='JSON',width = 600)
# widget to display dataframe of models selected
df_widget   = pn.pane.DataFrame(models,name='DF',width = 600, index = False)
# widgets to save json:
filename_widget = pn.widgets.TextInput(value='ecs_models.json')
save_widget_notebook = pn.widgets.Button(name='Save JSON to JupyterLab',button_type='primary')
save_widget_computer = pn.widgets.FileDownload(button_type='success',auto=True, callback = get_filename, filename = get_filename())

save_widget_notebook.on_click(save_notebook)

# plotting:

pipe = hv.streams.Pipe(data={'x1_eff':0,'x2_eff':8,'weighted':False,'type':'Effective'})

# plot objects:
rect_eff      = hv.DynamicMap(make_rectangle_eff,streams=[pipe]).opts(xlim=(0,8),ylim=(0,1))
selection_eff = hv.DynamicMap(make_kde,streams=[pipe], label = 'Density of selection').opts(color='purple',tools=[])
scatter_eff   = hv.DynamicMap(make_kde_scatter,streams=[pipe])
pdf_plot_eff  = model_obj.sherwood.hvplot.line(x='ECS',y='PDF', color = 'blue', label = 'Expected density (Sherwood)')

rect_eq       = hv.DynamicMap(make_rectangle_eq,streams=[pipe]).opts(xlim=(0,8),ylim=(0,1))
selection_eq  = hv.DynamicMap(make_kde_eq,streams=[pipe], label = 'Density of selection').opts(color='purple',tools=[])
scatter_eq    = hv.DynamicMap(make_kde_eq_scatter,streams=[pipe])
#errors_eq     = hv.DynamicMap(make_kde_errors,streams=[pipe]).opts(color='red',muted_alpha = 1)
pdf_plot_eq   = model_obj.sherwood.hvplot.line(x='ECS',y='PDFeq', color = 'blue', label = 'Expected density (Sherwood)')

pdf_render = hv.render(pdf_plot_eff)
size_rect = (pdf_render.x_range.start,pdf_render.y_range.start,pdf_render.x_range.end,pdf_render.y_range.end)
pipe.send({'x1_eff':size_rect[0],
           'x2_eff':size_rect[2],
           'x1_eq':convert_eff_to_eq(size_rect[0]),
           'x2_eq':convert_eff_to_eq(size_rect[2]),
           'weighted':False,
           'type':'Effective'})

pdf_plot_eff.opts(tools = ['box_select'])
pdf_plot_eq.opts(tools = ['box_select'])

pipe.add_subscriber(display_event)
weight_button.param.watch(update_weight,'value')

sstream_eff = hv.streams.SelectionXY(source=pdf_plot_eff)
sstream_eff.add_subscriber(pipe_wrapper)

sstream_eq  = hv.streams.SelectionXY(source=pdf_plot_eq)
sstream_eq.add_subscriber(pipe_wrapper)
def register_func(myrange_eq):
    return lambda event : pipe_wrapper(x_eq=myrange_eq)

for k,v in select_buttons.items():
    for vk,vv in v.items():
        vv['func'] = register_func(vv['range'],)
        vv['button'].on_click(vv['func'])
        
# because of a holoviz bug, we can't add rect to all_plot directly (issue #5056)? If all_plot is a dynamicmap plot seems to work...
all_plot_eff = rect_eff * pdf_plot_eff * selection_eff * scatter_eff 
all_plot_eff.opts(xlim = (size_rect[0],size_rect[2]), ylim = (size_rect[1],size_rect[3]))
all_plot_eff.opts(legend_position='top_right')
all_plot_eff.opts(height=400, show_grid=True)
all_plot_eff.opts(default_tools=['save','reset'], tools=[])
all_plot_eff.opts(toolbar='above')
all_plot_eff.opts(hooks=[remove_tools,fix_dims], active_tools = ['box_select'])
all_plot_eff.opts(title='Estimate of Effective Climate Sensitivity for CMIP-6')
all_plot_eff.opts(xlabel='Effective Climate Sensitivity (°C)', ylabel='Density (1/°C)')


all_plot_eq = rect_eq * pdf_plot_eq * selection_eq *  scatter_eq
all_plot_eq.opts(xlim = (size_rect[0],size_rect[2]), ylim = (size_rect[1],size_rect[3]))
all_plot_eq.opts(legend_position='top_right')
all_plot_eq.opts(height=400, show_grid=True)
all_plot_eq.opts(default_tools=['save','reset'], tools=[])
all_plot_eq.opts(toolbar='above')
all_plot_eq.opts(hooks=[remove_tools,fix_dims], active_tools = ['box_select'])
all_plot_eq.opts(title='Estimate of Equilibrium Climate Sensitivity for CMIP-6')
all_plot_eq.opts(xlabel='Equilibrium Climate Sensitivity (°C)', ylabel='Density (1/°C)')

plot_tabs = pn.Tabs(('Effective',all_plot_eff),('Equilibrium',all_plot_eq))


plot_tabs.param.watch(update_plottype,'active')
# arrange panel layout:
text_widget.height = 100
app = pn.Column(pn.Row(plot_tabs,plot_options), \
          pn.Row(text_widget,summary_stats), \
          pn.Row(
            pn.Tabs(json_widget,pn.Card(df_widget,title="DataFrame",name="DF",collapsed=True,hide_header=False,sizing_mode='stretch_width')),
            pn.Column(filename_widget,
                      save_widget_notebook,
                      save_widget_computer)
                ),
         citation_widget)

app.servable()