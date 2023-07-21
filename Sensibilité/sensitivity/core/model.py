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
from scipy import stats
from scipy import interpolate
from scipy import integrate
import numpy as np
import pandas as pd

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
