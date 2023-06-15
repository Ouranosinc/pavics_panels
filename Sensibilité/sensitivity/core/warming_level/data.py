from .interfaces import DataInterface
import param
class Input(DataInterface):
    toggle = param.Boolean(default=True)
    generation = param.String()
    center = param.String()
    scenario = param.String()
    realization = param.String()
    threshold = param.Number(default=1.5)
    refperiod = param.Tuple(default=(1991,2020))

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
        