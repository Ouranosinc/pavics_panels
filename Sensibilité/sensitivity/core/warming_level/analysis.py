from .interfaces import DataInterface, AnalysisInterface
import param

class Analysis(AnalysisInterface):
    data = param.ClassSelector(class_=DataInterface)
    
    def __init__(self, data, **params):
        super().__init__(**params)
        self.data = data
        
    @param.depends('data.input.param', watch=True)
    def update_data(self):
        print('updating value in Analysis')
        self.data.output.value = f'''Current value: 
            {self.data.input.toggle}, 
            {self.data.input.generation}, 
            {self.data.input.center}, 
            {self.data.input.scenario}, 
            {self.data.input.realization}, 
            {self.data.input.threshold}, 
            {self.data.input.refperiod}'''