from .interfaces import DataInterface
import param
class Input(DataInterface):
    toggle = param.Boolean()
    generation =  param.String()
    center =      param.String()
    scenario =    param.String()
    realization = param.String()
    threshold = param.Number()
    refperiod = param.Tuple()

class Output(DataInterface):
    value  = param.String()

class Data(DataInterface):
    # simple data class to hold the input/output parameters
    # input:
    input = param.ClassSelector(class_=Input)
    output = param.ClassSelector(class_=Output)

    # output:
    
    def __init__(self, **params):
        super().__init__(**params)
        self.input = Input()
        self.output = Output()
        