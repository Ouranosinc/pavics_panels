from sensitivity.constants.constants import *
from sensitivity.core import warming_level
import panel as pn
import param
data = warming_level.Model()
# nb: initialize analysis first, to give it priority in param.depends
# alternative: always use param.watch
analysis = warming_level.Control(data=data)
widgets = warming_level.View(data=data)


dash = pn.template.BootstrapTemplate(title="warming level",)
#dash.header.append(loading)


dash.main.append(widgets._output_layout)
dash.sidebar.append(widgets._input_layout)
#pn.state.onload(update_df_years)
dash.servable()