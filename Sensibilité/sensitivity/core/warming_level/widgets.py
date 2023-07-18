from .interfaces import DataInterface,WidgetInterface
import param
import panel as pn
from panel.viewable import Viewer
import numpy as np

class Widget(Viewer):
    ''' Simple wrapper for panel widgets that 
        include the name of variables to watch for changes (e.g. 'value_throttled'),
        and the data attribute on which to apply the changes.
    '''
    item = param.ClassSelector(class_=pn.widgets.base.Widget)
    watch = param.String()
    data_attr = param.String()
    def __init__(self, item, watch='value',data_attr='', **params):
        super().__init__(**params)
        self.watch = watch
        self.item  = item
        self.data_attr = data_attr
        
    def __panel__(self):
        return self.item
    
class Widgets(WidgetInterface):
    data = param.ClassSelector(class_=DataInterface)
    
    # input widgets:
    inputs = param.List([],item_type=Widget)
    
    display = param.ClassSelector(class_=pn.pane.Str)
    
    def __init__(self, data, **params):
        super().__init__(**params)
        with param.parameterized.batch_call_watchers(self):
            self.data = data
            self.create_widgets()
            self.display = pn.pane.Str(name='Display')
            self.table = pn.pane.DataFrame(name='Table',max_rows=25,show_dimensions=True)
        self.set_watchers()
        self._input_layout = pn.FlexBox(*self.inputs,flex_direction='column')
        self._output_layout = pn.Row(self.display,self.table)
        self._layout = pn.Row(
            self._input_layout,
            self._output_layout
        )
    def set_watchers(self):
        for widget in self.inputs:
            widget.item.param.watch(self.update_data(widget.data_attr),widget.watch)
            widget.item.param.trigger(widget.watch)
    def create_widgets(self):
        # input widgets: list of  Widget(pn.widget,watch_value, data_attribute)
        zelinka_toggle = pn.widgets.Checkbox(name='Only show Zelinka models')
        self.inputs.append(Widget(zelinka_toggle,data_attr='toggle'))
        ref_slider = pn.widgets.IntRangeSlider(name="reference period",start=1850, end=2100,value=(1850,1900),step=10)
        self.inputs.append(Widget(ref_slider,'value_throttled','refperiod'))
        thresh_slider = pn.widgets.FloatSlider(name="threshold",start=0.0, end=6.0, step=0.5,value=1.5)
        self.inputs.append(Widget(thresh_slider,'value_throttled','threshold'))
        self.inputs.append(Widget(pn.widgets.Select(name="generation" , options=['']),data_attr='generation'))
        self.inputs.append(Widget(pn.widgets.Select(name="center"     , options=['']),data_attr='center'))
        self.inputs.append(Widget(pn.widgets.Select(name="scenario"   , options=['']),data_attr='scenario'))
        self.inputs.append(Widget(pn.widgets.Select(name="realization", options=['']) ,data_attr='realization'))
    
    def update_data(self, name):
        print('Creating helper in Widgets:',name)
        def event_fn(event):
            print('updating data in Widgets:',name)
            if hasattr(self.data.input,name):
                self.data.input[name] = event.new
            else:
                warnings.warn(f'Attribute not found: {name}, not updating data.')
        return event_fn
    
    @param.depends('data.input.tas', watch=True)
    def intialize_widgets(self):
        print('initializing widget options in Widgets')
        self.inputs[3].item.options = ['all',*np.unique(self.data.output.tas.generation.values)]
        self.inputs[4].item.options = ['all',*np.unique(self.data.output.tas.center.values)]
        self.inputs[5].item.options = ['all',*np.unique(self.data.output.tas.scenario.values)]
        self.inputs[6].item.options = ['all','first']
        
    
    @param.depends('data.output.tas_df', watch=True)
    def update_table(self):
        #print('updating display in Widgets')
        #self.display.object = self.data.output.value
        self.table.object = self.data.output.tas_df
        
    @param.depends('data.output.value', watch=True)
    def update_string(self):
        self.display.object = self.data.output.value
    
    
    def __panel__(self):
        print('rerendering')
        return self._layout