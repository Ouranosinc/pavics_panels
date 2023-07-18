import param
from panel.viewable import Viewer
class ViewInterface(Viewer):
    pass

class ControlInterface(param.Parameterized):
    def __init__(self, **params):
        super().__init__(**params)

class ModelInterface(param.Parameterized):
    def __init__(self, **params):
        super().__init__(**params)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def __getitem__(self, key):
        return getattr(self, key) 
    
