
import pandas as pd
import xarray as xr
from load import load_global_tas, load_sherwood, load_zelinka
tas = load_global_tas().set_index("year")
tas.columns = tas.columns.str.split("_", expand=True)
tas = xr.DataArray(tas,dims=['year','model'],name='tas').rename({"model_level_0":"generation","model_level_1":"center","model_level_2":"scenario","model_level_3":"realization"})
import json
import urllib
zelinka_url = "https://raw.githubusercontent.com/mzelinka/cmip56_forcing_feedback_ecs/master/cmip56_forcing_feedback_ecs.json"
data = urllib.request.urlopen(zelinka_url)
zelinka = json.load(data)
zelinka = {k: v for k, v in zelinka.items() if k in ["CMIP6","CMIP5"]}
zelinka = pd.json_normalize(zelinka, sep="_")
zelinka.columns = zelinka.columns.str.split("_", expand=True)
zelinka = zelinka.transpose().unstack()
zelinka.columns = zelinka.columns.droplevel(0)
zelinka = xr.Dataset(zelinka).rename({"dim_0":"model","dim_0_level_0":"generation","dim_0_level_1":"center","dim_0_level_2":"realization"})
#ecs = load_zelinka()
sherwood = load_sherwood()


import panel as pn
import holoviews as hv
import hvplot.xarray
import hvplot.pandas
pn.extension()
import numpy as np
refyears = 29
import xscen as xs

tas_filt = tas.where(tas.generation=='CMIP6',drop=True)

refs_start = range(tas.year.values[0]+1,tas.year.values[-1]-refyears,10)
w_reference_period = pn.widgets.IntRangeSlider(name="reference period", 
                                               start=1850, end=2100,value=(1850,1900))
w_generation  = pn.widgets.Select(name="generation",  disabled = False, 
                                  options=['all',*list(np.unique(tas_filt['generation'].values))],
                                  value = 'CMIP6')
w_center      = pn.widgets.Select(name="center",      disabled = False , 
                                  options=['all',*list(np.unique(tas_filt['center'].values))] )
w_scenario    = pn.widgets.Select(name="scenario",    disabled = False , 
                                  options=['all',*list(np.unique(tas_filt['scenario'].values))] )
w_realization = pn.widgets.Select(name="realization", disabled = False , 
                                  options=['all','first',*list(np.unique(tas_filt['realization'].values))] )

w_models = pn.widgets.Select(name="Models",options=["CMIP6","Match Zelinka CMIP6",],value="Match Zelinka CMIP6")
     
fut = slice(2071,2100)

def tas_select_model(model='CMIP6',scenario='all', refperiod=(1991,2020),*args):
    tas_sel = tas_filt
    if model != 'CMIP6':
        
        sel = xr.DataArray(
            [el in zelinka.indexes['model'] for el in tas_sel.indexes['model'].droplevel('scenario')],
            dims=tas_sel.model.dims, coords = tas_sel.model.coords
        )

        tas_sel = tas_sel.where(sel)
  
    if scenario != 'all':
        tas_sel = tas_sel.where(tas_sel.scenario==scenario,drop=True)
        
    ref = tas_sel.sel(year=slice(refperiod[0],refperiod[1])).mean(dim="year")
    return tas_sel,ref
plot_width = 800
plot_height= 400

@pn.depends(model=w_models,
            scenario=w_scenario,
            refperiod=w_reference_period)
def plot_kde(**kwargs):
    
    tas_sel,ref = tas_select_model(*list(kwargs.values()))
    
    delta = tas_sel.sel(year=fut).mean(dim="year") - ref
    df = delta.to_dataframe()
    x = np.linspace(min(0,df.tas.min()),max(6,df.tas.max()),1000)
    
    kde = stats.gaussian_kde(df.tas.dropna().values)
    
    df_pane = pn.pane.DataFrame(df,max_height=plot_height,sizing_mode='stretch_both',max_rows=10)
    #df_plot = df.hvplot.kde(title=f'KDE of delta TAS for future period: {fut.start}-{fut.stop}',xlabel='Delta TAS (K)',ylabel='Density').options(width=plot_width,height=plot_height)
    y = df.tas.values
    N = len(y)
    df_plot = pd.DataFrame({'tas':x,'density':kde.pdf(x)}).hvplot(x='tas',y='density',title=f'KDE of warming level, N={N}',xlabel='Year of threshold',ylabel='Density')
    df_scatter = pd.DataFrame({'tas':y,'density':kde.pdf(y)}).hvplot.scatter(x='tas',y='density',title=f'KDE of warming level, N={N}',xlabel='Year of threshold',ylabel='Density')
    df_plot = (df_plot * df_scatter).options(width=plot_width,height=plot_height)
    return pn.Column(pn.pane.HoloViews(df_plot,linked_axes=False),df_pane,height=plot_height*2,width=plot_width)
    
    
    #return pn.Column(df_plot,df_pane,height=plot_height*2,width=plot_width)
from scipy import stats, interpolate
import param
w_threshold = pn.widgets.FloatSlider(name="threshold", value=1.5, start=0.0, end=6.0, step=0.25)
print(type(w_threshold.value))
class C(param.Parameterized):
    value = param.Number(default=1.5)
    
loading = pn.indicators.BooleanStatus(color='success',value=True)
@pn.depends(w_reference_period,
            w_threshold,watch=True)
def start_spinner(*args):
    loading.color = 'warning'

class Widgets(param.Parameterized):
    # input parameters:
    # (from widget.value)
    model = param.String()
    scenario = param.String()
    realization = param.String()
    threshold = param.Number(default=1.5)
    refperiod = param.Tuple(default=(1991,2020))
    
    # output data:
    df_all    = param.DataFrame(instantiate=True)
    df_models = param.DataFrame(instantiate=True)
    
    # output widgets:
    
class Data(param.Parameterized):
    # input parameters:
    # trigger these to update the data.
    model = param.String()
    scenario = param.String()
    realization = param.String()
    threshold = param.Number(default=1.5)
    refperiod = param.Tuple(default=(1991,2020))
    
    # output parameters
    df_all    = param.DataFrame(instantiate=True)
    df_models = param.DataFrame(instantiate=True)

data = Data()

@pn.depends(model=w_models,
            scenario=w_scenario,
            refperiod=w_reference_period.param.value_throttled,
            threshold=w_threshold.param.value_throttled,
            watch=True,
            on_init=True
            )
def update_df_years(**kwargs):
    loading.color = 'warning'
    tas_sel,ref = tas_select_model(*list(kwargs.values()))
    tas_sel = tas_sel.rolling(year=30,center=True).mean()
    print('update_df_years')
    delta = tas_sel - ref
    delta = delta.where(delta > kwargs['threshold'],drop=True)
    df = delta.to_dataframe()
    deltayr = {}
    #deltayr['year'] = np.nan
    for model in delta.model.values:
        deltayr[model] = delta.sel(model=model).dropna(dim='year',how='all').year.min()
    deltayr = pd.DataFrame.from_dict(deltayr,orient='index',columns=['year'])
    deltayr['weight'] = 1 #stats.norm.pdf(deltayr.year,deltayr.year.mean(),deltayr.year.std())  
    kde = stats.gaussian_kde(deltayr.year,weights=deltayr.weight)
    
    x_all = np.linspace(min(2000,deltayr.year.min()),2100,1000)
    x_models = deltayr.year.values
    y_all = kde.pdf(x_all)
    y_models = kde.pdf(x_models)
    deltayr['density'] = y_models
    
    with param.parameterized.batch_call_watchers(data):
        data.df_all = pd.DataFrame({'year':x_all,'density':y_all})
        data.df_models = deltayr
        
@pn.depends(data.param.df_all,
            data.param.df_models)
def year_kde(df_all,df_models):
    if (not(hasattr(df_all,'empty')) or df_all.empty) or (not(hasattr(df_models,'empty')) or df_models.empty):
        return pn.pane.HoloViews()
    N = len(df_models)
    df_plot    = df_all.hvplot(x='year',y='density',title=f'KDE of warming level, N={N}',xlabel='Year of threshold',ylabel='Density')
    df_plot = df_plot.opts(tools=[])
    df_scatter = df_models.hvplot.scatter(x='year',y='density', 
                                          hover_cols=['year','model','density','weight'],
                                          title=f'KDE of warming level, N={N}',
                                          xlabel='Year of threshold',
                                          ylabel='Density')
    df_scatter = df_scatter.opts(tools=['hover'])
    df_plot = (df_plot * df_scatter).options(ylim=(0,0.06),width=plot_width,height=plot_height)
    df_pane = pn.pane.DataFrame(df_models,max_height=plot_height,sizing_mode='stretch_both',max_rows=10)
    loading.color = 'success'
    
    return pn.Column(pn.pane.HoloViews(df_plot,linked_axes=False),df_pane,height=plot_height*2,width=plot_width)
    

# %%
dash = pn.template.BootstrapTemplate(title="warming level",)
dash.header.append(loading)
layout = pn.Column(
    pn.Row(w_models,w_scenario),
    w_reference_period,
    pn.Tabs(
       pn.Column(
           w_threshold,
           year_kde,
           name='Year KDE'
           ),
       pn.Column(
           pn.panel(f'future period: **{fut.start}-{fut.stop}**',height=w_threshold.height),
           plot_kde,
           name='Surface Temperature KDE'
           ),
    )
)
        
dash.main.append(layout)
dash.servable()




