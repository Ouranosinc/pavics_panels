from .interfaces import DataInterface,WidgetInterface
import param
import panel as pn
from panel.viewable import Viewer
class Widget(Viewer):
    
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
        
        self.set_watchers()
        self._input_layout = pn.FlexBox(*self.inputs,flex_direction='column')
        self._output_layout = pn.Column(self.display)
        self._layout = pn.Row(
            self._input_layout,
            self._output_layout
        )
    def set_watchers(self):
        for widget in self.inputs:
            
            widget.item.param.watch(self.update_data(widget.data_attr),widget.watch)
            
    def create_widgets(self):
        # input widgets: list of  Widget(pn.widget,watch_value, data_attribute)
        self.inputs.append(
            Widget(
                pn.widgets.Checkbox(
                    name='Click to toggle'),
                    data_attr='toggle')
            )
        self.inputs.append(
            Widget(
                pn.widgets.IntRangeSlider(
                    name="reference period",
                    start=1850, end=2100,value=(1850,1900)
                    ),
                'value_throttled',
                'refperiod'
                )
            )
        self.inputs.append(
            Widget(
                pn.widgets.FloatSlider(
                    name="threshold",
                    start=0.0, end=6.0, step=0.5,value=1.5
                    ),
                'value_throttled',
                'threshold'
                )
            )
        self.inputs.append(Widget(pn.widgets.Select(name="generation"),data_attr='generation'))
        self.inputs.append(Widget(pn.widgets.Select(name="center"),data_attr='center'))
        self.inputs.append(Widget(pn.widgets.Select(name="scenario"),data_attr='scenario'))
        self.inputs.append(Widget(pn.widgets.Select(name="realization") ,data_attr='realization'))
    
    def update_data(self, name):
        print('Creating helper in Widgets:',name)
        def event_fn(event):
            print('updating data in Widgets:',name)
            if hasattr(self.data.input,name):
                self.data.input[name] = event.new
        return event_fn
        
    @param.depends('data.output.param', watch=True)
    def update_display(self):
        print('updating display in Widgets')
        self.display.object = self.data.output.value
    
    def __panel__(self):
        print('rerendering')
        return self._layout