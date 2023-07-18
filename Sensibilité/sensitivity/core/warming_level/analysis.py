from .interfaces import ModelInterface, ControlInterface
import param
import xarray as xr

# Approximately: Control, from MVC abstract

class Control(ControlInterface):
    data = param.ClassSelector(class_=ModelInterface)
    
    def __init__(self, data, **params):
        super().__init__(**params)
        self.data = data

    @param.depends('data.input.param', watch=True)
    def update_selection_from_input(self):
        realization = self.data.input.realization
        generation = self.data.input.generation
        center = self.data.input.center
        scenario = self.data.input.scenario
        threshold = self.data.input.threshold
        refperiod = self.data.input.refperiod
        toggle_match = self.data.input.toggle
        
        tas = self.data.input.tas
        if realization in tas.coords['realization']:
            tas = tas.sel(realization=realization,drop=True)
        elif realization == 'all':
            tas = tas
        elif realization == 'first':
            tas = (tas
                .groupby(tas.model.realization.rename('member'))
                .first(skipna=True)
                )
        if generation in tas.coords['generation']:
            tas = tas.sel(generation=generation,drop=True)
        if center in tas.coords['center']:
            tas = tas.sel(center=center,drop=True)
        if scenario in tas.coords['scenario']:
            tas = tas.sel(scenario=scenario,drop=True)
        
        if toggle_match:
            sel = xr.DataArray(
                [el in self.data.input.zelinka.indexes['model'] for el in tas.indexes['model'].droplevel('scenario')],
                dims=tas.model.dims, coords = tas.model.coords
            )
            tas = tas.where(sel)

        self.data.output.tas = tas
        
    @param.depends('data.output.tas', watch=True)
    def update_df(self):
        ''' When data.output.tas is updated, update the corresponding dataframe
            to match.
        '''
        tas = self.data.output.tas
        if 'model' in tas.dims:
            tas = tas.dropna(dim="model",how='all')
        self.data.output.tas_df = tas.to_dataframe()
    
    @param.depends('data.output.tas','data.input.threshold', 'data.input.refperiod', watch=True)
    def update_delta_table(self):
        ''' When data.output.tas is updated, update the xr.DataArray holding the
            delta table.'''
        tas = self.data.output.tas
        refperiod = self.data.input.refperiod
        refperiod = slice(refperiod[0],refperiod[1])
        ref =  tas.sel(year=refperiod).mean(dim="year")
        delta = tas - ref
        # first compute the delta table:
        self.data.output.delta = delta
        # then find threshold years:
        threshold = self.data.input.threshold
        delta = delta.where(delta > threshold,drop=True)
        
        if False:
            tas = tas.rolling(year=30,center=True).mean()
            #tas_sel = tas_sel.rolling(year=30,center=True).mean()
            print('update_df_years')
            delta = tas_sel - ref
            delta = delta.where(delta > threshold,drop=True)
            df = delta.to_dataframe()
            deltayr = []
            #deltayr['year'] = np.nan
            #print(*delta.model.coords.keys())
            for model in delta.model.values:
                deltayr.append((*[x for x in model],delta.sel(model=model).dropna(dim='year',how='all').year.min().item()))
            #print([*[x for x in delta.model.coords.keys() if x != 'model'],'year'])
            deltayr = pd.DataFrame.from_records(deltayr,columns=[*[x for x in delta.model.coords.keys() if x != 'model'],'year'])
            #print(deltayr)
            deltayr['weight'] = 1 #stats.norm.pdf(deltayr.year,deltayr.year.mean(),deltayr.year.std()) 
            kde = stats.gaussian_kde(deltayr.year,weights=deltayr.weight)
            
            x_all = np.linspace(min(2000,deltayr.year.min()),2100,1000)
            x_models = deltayr.year.values
            y_all = kde.pdf(x_all)
            y_models = kde.pdf(x_models)
            deltayr['density'] = y_models
            
            cdf = scipy.integrate.cumtrapz(y_all,x_all,initial=0)
            
            max_prob = 0
            most_likely = (0,0,0)
            for x in np.arange(min(2000,deltayr.year.min()),2100,1):
                
                prob_x =  np.interp(x+29,x_all,cdf,left=0,right=1) - np.interp(x,x_all,cdf,left=0,right=1)
                #print(x,x+29,prob_x)
                if prob_x > max_prob:
                    most_likely = (x,x+29,prob_x)
                    max_prob = prob_x
            with param.parameterized.batch_call_watchers(data):
                data.df_all = pd.DataFrame({'year':x_all,'density':y_all})
                data.df_models = deltayr
                data.most_likely = most_likely
        pass
        
    @param.depends('data.input.param', watch=True)
    def update_string(self):
        print('updating value in Analysis')
        self.data.output.value = f'''Current value: 
            input.toggle     = {self.data.input.toggle}, 
            input.generation = {self.data.input.generation}, 
            input.center     = {self.data.input.center}, 
            input.scenario   = {self.data.input.scenario}, 
            input.realizatio = {self.data.input.realization}, 
            input.threshold  = {self.data.input.threshold}, 
            input.refperiod  = {self.data.input.refperiod}'''