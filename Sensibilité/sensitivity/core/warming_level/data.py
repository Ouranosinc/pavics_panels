import param
import xarray as xr
import pandas as pd
import json

from .interfaces import ModelInterface
from sensitivity.core.load import load_global_tas, load_sherwood, load_zelinka
from sensitivity.constants.constants import file_zelinka
# Approximately: Model, from MVC abstract

class ModelControl(ModelInterface):
    toggle = param.Boolean()
    generation =  param.String()
    center =      param.String()
    scenario =    param.String()
    realization = param.String()
    threshold = param.Number()
    refperiod = param.Tuple()
    
    tas = param.ClassSelector(class_=xr.DataArray)
    zelinka = param.ClassSelector(class_=xr.Dataset)
    def __init__(self,**params):
        super().__init__(**params)
        # load static data: 
        tas = load_global_tas().set_index("year")
        tas.columns = tas.columns.str.split("_", expand=True)
        tas = xr.DataArray(tas,dims=['year','model'],name='tas').rename({"model_level_0":"generation","model_level_1":"center","model_level_2":"scenario","model_level_3":"realization"})
        zelinka_url = "https://raw.githubusercontent.com/mzelinka/cmip56_forcing_feedback_ecs/master/cmip56_forcing_feedback_ecs.json"

        if file_zelinka.is_file():
            zelinka = json.load(open(file_zelinka))
        else:
            data = urllib.request.urlopen(zelinka_url)
            zelinka = json.load(data)
            json.dump(zelinka, open(file_zelinka, "w"))

        zelinka = {k: v for k, v in zelinka.items() if k in ["CMIP6","CMIP5"]}
        zelinka = pd.json_normalize(zelinka, sep="_")
        zelinka.columns = zelinka.columns.str.split("_", expand=True)
        zelinka = zelinka.transpose().unstack()
        zelinka.columns = zelinka.columns.droplevel(0)
        with param.parameterized.batch_call_watchers(self):
            self.tas = (tas)
            
            self.zelinka = xr.Dataset(zelinka).rename({"dim_0":"model","dim_0_level_0":"generation","dim_0_level_1":"center","dim_0_level_2":"realization"})

class ModelView(ModelInterface):
    value  = param.String()
    tas = param.ClassSelector(class_=xr.DataArray)
    tas_df = param.DataFrame()
    delta = param.ClassSelector(class_=xr.DataArray)
    
    def __init__(self,**params):
        super().__init__(**params)
        self.tas = xr.DataArray([],name='na')
        self.delta = xr.DataArray([],name='na')

class Model(ModelInterface):
    # simple data class to hold the input/output parameters
    # input:
    input = param.ClassSelector(class_=ModelControl)
    output = param.ClassSelector(class_=ModelView)

    # output:
    
    def __init__(self, **params):
        super().__init__(**params)
        self.input = ModelControl()
        self.output = ModelView()
        