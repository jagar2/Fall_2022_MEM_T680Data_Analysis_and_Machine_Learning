#!/usr/bin/env python
# coding: utf-8

# # Interactive Graphing with Dash
# 

# - Written on top of Plotly.js and React.js, Dash is ideal for building and deploying data apps with customized user interfaces. It's particularly suited for anyone who works with data.
# 
# - Through a couple of simple patterns, Dash abstracts away all of the technologies and protocols that are required to build a full-stack web app with interactive data visualization.
# 
# - Dash is simple enough that you can bind a user interface to your code in less than 10 minutes
# 
# - Dash apps are rendered in the web browser. You can deploy your apps to VMs or Kubernetes clusters and then share them through URLs. Since Dash apps are viewed in the web browser, Dash is inherently cross-platform and mobile ready
# 

# ## Dash Basics
# 
# Dash apps are composed of two parts.
# 
# - The first part is the `layout` of the app and it describes what the application looks like.
# - The second part describes the interactivity
# 

# In[1]:


from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
from jupyter_dash import JupyterDash
from IPython.display import IFrame


# ## A First Example with Dash
# 

# In[ ]:


app = Dash(__name__)

# This is the dataframe where the data is stored.
# Plot.ly and dash like to use Pandas dataframe. This is a data structure like an excel spreadsheet
df = pd.DataFrame(
    {
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"],
    }
)

# Plots a bar graph
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

app.layout = html.Div(
    children=[
        # heading
        html.H1(children="Hello Dash"),
        # subheading
        html.Div(
            children="""
        Dasher: A web application framework for your data.
    """
        ),
        dcc.Graph(id="example-graph", figure=fig),
    ]
)


# In[ ]:


app.run_server()


# 1. The layout is composed of a tree of `components` such as `html.Div` and `dcc.Graph`. The Dash HTML Components module (dash.html)
# 
# 1. has a component for every HTML tag. `The html.H1(children='Hello Dash')` component generates a `<h1>Hello Dash</h1>` HTML element in your application.
# 
# 1. The children property is special. By convention, it's always the first attribute which means that you can omit it: `html.H1(children='Hello Dash')` is the same as `html.H1('Hello Dash')`. It can contain a string, a number, a single component, or a list of components.
# 
# 1. You can apply custom css stylesheets if you know HTML and web development
# 

# ## Making Changes to Dash Plot
# 
# Dash enables hot reloading, this means you can modify the code of a plot and see the changes in real time. This is achieved by using `app.run_server(debug=True)`
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_1.py', '\nfrom dash import Dash, html, dcc\nimport plotly.express as px\nimport pandas as pd\n\napp = Dash(__name__)\n\n# This is the dataframe where the data is stored. \n# Plot.ly and dash like to use Pandas dataframe. This is a data structure like an excel spreadsheet\ndf = pd.DataFrame({\n    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],\n    "Amount": [4, 1, 2, 2, 4, 5],\n    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]\n})\n\n# Plots a bar graph\nfig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")\n\napp.layout = html.Div(children=[\n    # heading\n    html.H1(children=\'Hello Dasher\'),\n\n    # subheading\n    html.Div(children=\'\'\'\n        Dash: A web application framework for your data.\n    \'\'\'),\n\n    dcc.Graph(\n        id=\'example-graph\',\n        figure=fig\n    )\n])\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# ### HTML Componets
# 

# Dash HTML Components (`dash.html`) contains a component class for every HTML tag as well as keyword arguments for all of the HTML arguments.
# 

# Let's customize the text in our app by modifying the inline styles of the components. Create a file named dash_2.py with the following code:
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_1.py', '\n# Run this app with `dash_scripts/dash_1.py` and\n# visit http://127.0.0.1:8050/ in your web browser.\n\nfrom dash import Dash, dcc, html\nimport plotly.express as px\nimport pandas as pd\n\napp = Dash(__name__)\n\n# sets the colors in a dictionary\ncolors = {\n    \'background\': \'#111111\',\n    \'text\': \'#7FDBFF\'\n}\n\n# assume you have a "long-form" data frame\n# see https://plotly.com/python/px-arguments/ for more options\ndf = pd.DataFrame({\n    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],\n    "Amount": [4, 1, 2, 2, 4, 5],\n    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]\n})\n\nfig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")\n\nfig.update_layout(\n    plot_bgcolor=colors[\'background\'],\n    paper_bgcolor=colors[\'background\'],\n    font_color=colors[\'text\']\n)\n\n# These are all HTML tags\napp.layout = html.Div(style={\'backgroundColor\': colors[\'background\']}, children=[\n    html.H1(\n        children=\'Hello Dash\',\n        style={\n            \'textAlign\': \'center\',\n            \'color\': colors[\'text\']\n        }\n    ),\n\n    html.Div(children=\'Dash: A web application framework for your data.\', style={\n        \'textAlign\': \'center\',\n        \'color\': colors[\'text\']\n    }),\n\n    dcc.Graph(\n        id=\'example-graph-2\',\n        figure=fig\n    )\n])\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# ## Visualization Options
# 
# Bar charts are not that interesting. Plot.ly provides a lot of graph options.
# 
# - The Dash Core Components module (`dash.dcc`) includes a component called Graph.
# 
# - Graph renders interactive data visualizations using the open source plotly.js JavaScript graphing library. Plotly.js supports over 35 chart types and renders charts in both vector-quality SVG and high-performance WebGL.
# 

# ### Creating a scatterplot from a pandas dataframe
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_2.py', '\n# Run this app with `python app.py` and\n# visit http://127.0.0.1:8050/ in your web browser.\n\nfrom dash import Dash, dcc, html\nimport plotly.express as px\nimport pandas as pd\n\n\napp = Dash(__name__)\n\n# reads a dataframe from the web\ndf = pd.read_csv(\'https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv\')\n\n# plots the scatterplot\nfig = px.scatter(df, # dataframe\n                 x="gdp per capita", # x value\n                 y="life expectancy", # y value\n                 size="population", # size of scatter\n                 color="continent", # color of scatter\n                 hover_name="country", # What appears on the scatter\n                 log_x=True, size_max=60)\n\napp.layout = html.Div([\n    dcc.Graph(\n        id=\'life-exp-vs-gdp\',\n        figure=fig\n    )\n])\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# ### Dash Markdown
# 
# Since HTML can be somewhat less familar it is also possible to use markdown in dash
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_3.py', "\n# Run this app with `python app.py` and\n# visit http://127.0.0.1:8050/ in your web browser.\n\nfrom dash import Dash, html, dcc\n\napp = Dash(__name__)\n\nmarkdown_text = '''\n### Dash and Markdown\n\nDash apps can be written in Markdown.\nDash uses the [CommonMark](http://commonmark.org/)\nspecification of Markdown.\nCheck out their [60 Second Markdown Tutorial](http://commonmark.org/help/)\nif this is your first introduction to Markdown!\n'''\n\napp.layout = html.Div([\n    dcc.Markdown(children=markdown_text)\n])\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# ## Core Components
# 
# - Dash Core Components (`dash.dcc`) includes a set of higher-level components like dropdowns, graphs, markdown blocks, and more.
# 
# - Every option that is configurable is available as a keyword argument of the component.
# 
# - You can view all of the available components in the [Dash Core Components overview](https://dash.plotly.com/dash-core-components).
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_4.py', "\n# Run this app with `python app.py` and\n# visit http://127.0.0.1:8050/ in your web browser.\n\nfrom dash import Dash, html, dcc\n\napp = Dash(__name__)\n\napp.layout = html.Div([\n    html.Div(children=[\n        html.Label('Dropdown'),\n        # Dropdown\n        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),\n\n        html.Br(),\n        # Dropdown multiselect\n        html.Label('Multi-Select Dropdown'),\n        dcc.Dropdown(['New York City', 'Montréal', 'San Francisco'],\n                     ['Montréal', 'San Francisco'],\n                     multi=True),\n\n        html.Br(),\n        html.Label('Radio Items'),\n        dcc.RadioItems(['New York City', 'Montréal', 'San Francisco'], 'Montréal'),\n    ], style={'padding': 10, 'flex': 1}),\n\n    html.Div(children=[\n        html.Label('Checkboxes'),\n        # Checklist\n        dcc.Checklist(['New York City', 'Montréal', 'San Francisco'],\n                      ['Montréal', 'San Francisco']\n        ),\n\n        html.Br(),\n        # Label\n        html.Label('Text Input'),\n        dcc.Input(value='MTL', type='text'),\n\n        html.Br(),\n        html.Label('Slider'),\n        # Slider\n        dcc.Slider(\n            min=0,\n            max=9,\n            marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},\n            value=5,\n        ),\n    ], style={'padding': 10, 'flex': 1})\n], style={'display': 'flex', 'flex-direction': 'row'})\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# ## Callbacks
# 
# So far we created adaptive visualization, these did not have interactivity
# 
# - Callbacks are used to perform an operation when an action is taken
# - This is a common concept in all interactive programming
# 

# ### Simple Interactive Dash App
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_5.py', '\nfrom dash import Dash, dcc, html, Input, Output\n\napp = Dash(__name__)\n\napp.layout = html.Div([\n    # text\n    html.H6("Change the value in the text box to see callbacks in action!"),\n    html.Div([\n        "Input: ",\n        dcc.Input(id=\'my-input\', value=\'initial value\', type=\'text\')  # input\n    ]),\n    html.Br(),\n    html.Div(id=\'my-output\'), # Id\n\n])\n\n# Here is a decorator\n# when my-input or my-output is changed\n@app.callback(\n    Output(component_id=\'my-output\', component_property=\'children\'),\n    Input(component_id=\'my-input\', component_property=\'value\')\n)\n# returns a string that is the new input value\ndef update_output_div(input_value):\n    return f\'Output: {input_value}\'\n\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# 1. The "inputs" and "outputs" of our application are described as the arguments of the @app.callback decorator.
# 

# 2. In Dash, the inputs and outputs of our application are simply the properties of a particular component. In this example, our input is the "value" property of the component that has the ID "my-input". Our output is the "children" property of the component with the ID "my-output".
# 

# 3. Whenever an input property changes, the function that the callback decorator wraps will get called automatically. Dash provides this callback function with the new value of the input property as its argument, and Dash updates the property of the output component with whatever was returned by the function.
# 

# 4. The `component_id` and `component_property` keywords are optional (there are only two arguments for each of those objects). They are included in this example for clarity.
# 

# 5. Notice how we don't set a value for the children property of the my-output component in the layout. When the Dash app starts, it automatically calls all of the callbacks with the initial values of the input components in order to populate the initial state of the output components.
# 
# - In this example, if you specified the div component as html.Div(id='my-output', children='Hello world'), it would get overwritten when the app starts.
# 

# ### Common updates
# 
# 1. `children` this is the property of an HTML componet
# 2. `figure` to change the display data of a graph
# 3. `style` to change the style of a graph or an object
# 

# ## Updating Graphs
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_6.py', '\nfrom dash import Dash, dcc, html, Input, Output\nimport plotly.express as px\n\nimport pandas as pd\n\ndf = pd.read_csv(\'https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv\')\n\napp = Dash(__name__)\n\napp.layout = html.Div([\n    dcc.Graph(id=\'graph-with-slider\'),\n    dcc.Slider(\n        df[\'year\'].min(),\n        df[\'year\'].max(),\n        step=None,\n        value=df[\'year\'].min(),\n        marks={str(year): str(year) for year in df[\'year\'].unique()},\n        id=\'year-slider\'\n    )\n])\n\n\n@app.callback(\n    Output(\'graph-with-slider\', \'figure\'),\n    Input(\'year-slider\', \'value\'))\ndef update_figure(selected_year):\n    # filters the dataframe\n    filtered_df = df[df.year == selected_year]\n\n    # updates the graph with the filtered data\n    fig = px.scatter(filtered_df, x="gdpPercap", y="lifeExp",\n                     size="pop", color="continent", hover_name="country",\n                     log_x=True, size_max=55)\n\n    # animation for the updated layout\n    fig.update_layout(transition_duration=500)\n\n    return fig\n\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# In this example, the "value" property of the `dcc.Slider` is the input of the app, and the output of the app is the "figure" property of the `dcc.Graph`.
# 
# 1. Whenever the value of the `dcc.Slider` changes, Dash calls the callback function `update_figure` with the new value.
# 1. The function filters the dataframe with this new value, constructs a figure object, and returns it to the Dash application.
# 

# ### Key Features
# 

# - We use the Pandas library to load our dataframe at the start of the app: `df = pd.read_csv('...')`. This dataframe df is in the global state of the app and can be read inside the callback functions.
# 

# - Loading data into memory can be expensive. By loading querying data at the start of the app instead of inside the callback functions, we ensure that this operation is only done once -- when the app server starts.
#   - When a user visits the app or interacts with the app, that data (df) is already in memory.
#   - If possible, expensive initialization (like downloading or querying data) should be done in the global scope of the app instead of within the callback functions.
# 

# - The callback does not modify the original data, it only creates copies of the dataframe by filtering using pandas.
#   - **your callbacks should never modify variables outside of their scope.**
#   - If your callbacks modify global state, then one user's session might affect the next user's session and when the app is deployed on multiple processes or threads, those modifications will not be shared across sessions.
# 

# - We are turning on transitions with layout.transition to give an idea of how the dataset evolves with time: transitions allow the chart to update from one state to the next smoothly, as if it were animated.
# 

# ## Dash App with Multiple Inputs
# 
# - Any "output" can have multiple "input" components.
# 
# - Here's a simple example that binds five inputs (the `value` property of two `dcc.Dropdown` components, two `dcc.RadioItems` components, and one `dcc.Slider` component) to one output component (the figure property of the `dcc.Graph component`).
# 
# - Notice how `app.callback` lists all five Input items after the Output. This allows for the consideration of all filters
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_7.py', "\nfrom dash import Dash, dcc, html, Input, Output\nimport plotly.express as px\n\nimport pandas as pd\n\napp = Dash(__name__)\n\ndf = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')\n\napp.layout = html.Div([\n    html.Div([\n\n        # x column\n        html.Div([\n            dcc.Dropdown(\n                df['Indicator Name'].unique(), # all the unique indicator names\n                'Fertility rate, total (births per woman)', # default\n                id='xaxis-column'\n            ),\n            dcc.RadioItems(\n                ['Linear', 'Log'], # log or linear\n                'Linear',\n                id='xaxis-type',\n                inline=True\n            )\n        ], style={'width': '48%', 'display': 'inline-block'}),\n\n        # y column\n        html.Div([\n            dcc.Dropdown(\n                df['Indicator Name'].unique(), # all the unique indicator names\n                'Life expectancy at birth, total (years)', # default\n                id='yaxis-column'\n            ),\n            dcc.RadioItems(\n                ['Linear', 'Log'],\n                'Linear',\n                id='yaxis-type',\n                inline=True\n            )\n        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})\n    ]),\n\n    dcc.Graph(id='indicator-graphic'),\n\n    dcc.Slider(\n        df['Year'].min(), # min year range\n        df['Year'].max(), # max year range\n        step=None,\n        id='year--slider',\n        value=df['Year'].max(),\n        marks={str(year): str(year) for year in df['Year'].unique()}, # where the marks go\n\n    )\n])\n\n\n@app.callback(\n    Output('indicator-graphic', 'figure'),\n    # note that we take all of  the input paramters each time the callback is called\n    Input('xaxis-column', 'value'),\n    Input('yaxis-column', 'value'),\n    Input('xaxis-type', 'value'),\n    Input('yaxis-type', 'value'),\n    Input('year--slider', 'value'))\ndef update_graph(xaxis_column_name, yaxis_column_name,\n                 xaxis_type, yaxis_type,\n                 year_value):\n    # filters the data based on year\n    dff = df[df['Year'] == year_value]\n    \n    # filters the data in the scatterplot\n    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],\n                     y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],\n                     hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])\n\n    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')\n\n    # sets the scales of the axis\n    fig.update_xaxes(title=xaxis_column_name,\n                     type='linear' if xaxis_type == 'Linear' else 'log')\n\n    fig.update_yaxes(title=yaxis_column_name,\n                     type='linear' if yaxis_type == 'Linear' else 'log')\n\n    return fig\n\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# ## Dash App with Multiple Outputs
# 
# You can also have multiple outputs
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_8.py', "\nfrom dash import Dash, dcc, html\nfrom dash.dependencies import Input, Output\n\nexternal_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\napp.layout = html.Div([\n    dcc.Input(\n        id='num-multi',\n        type='number',\n        value=5\n    ),\n    html.Table([\n        html.Tr([html.Td(['x', html.Sup(2)]), html.Td(id='square')]),\n        html.Tr([html.Td(['x', html.Sup(3)]), html.Td(id='cube')]),\n        html.Tr([html.Td([2, html.Sup('x')]), html.Td(id='twos')]),\n        html.Tr([html.Td([3, html.Sup('x')]), html.Td(id='threes')]),\n        html.Tr([html.Td(['x', html.Sup('x')]), html.Td(id='x^x')]),\n    ]),\n])\n\n\n@app.callback(\n    # here is where the multiple outputs are defined\n    Output('square', 'children'),\n    Output('cube', 'children'),\n    Output('twos', 'children'),\n    Output('threes', 'children'),\n    Output('x^x', 'children'),\n    Input('num-multi', 'value'))\ndef callback_a(x):\n    return x**2, x**3, 2**x, 3**x, x**x\n\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# ## Dash app with chained callbacks
# 
# You can also chain outputs and inputs together: the output of one callback function could be the input of another callback function.
# 

# This pattern can be used to create dynamic UIs where, for example, one input component updates the available options of another input component. Here's a simple example.
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_9.py', "\nfrom dash import Dash, dcc, html, Input, Output\n\nexternal_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\nall_options = {\n    'America': ['New York City', 'San Francisco', 'Cincinnati'],\n    'Canada': [u'Montréal', 'Toronto', 'Ottawa']\n}\napp.layout = html.Div([\n    # country radio item\n    dcc.RadioItems(\n        list(all_options.keys()),\n        'America',\n        id='countries-radio',\n    ),\n\n    html.Hr(),\n\n    # City radio item\n    # The cities available will depend on the country\n    dcc.RadioItems(id='cities-radio'),\n\n    html.Hr(),\n\n    html.Div(id='display-selected-values')\n])\n\n# if countries change it will change the cities\n@app.callback(\n    Output('cities-radio', 'options'),\n    Input('countries-radio', 'value'))\ndef set_cities_options(selected_country):\n    return [{'label': i, 'value': i} for i in all_options[selected_country]]\n\n# updates the available city options when a country is selected\n@app.callback(\n    Output('cities-radio', 'value'),\n    Input('cities-radio', 'options'))\ndef set_cities_value(available_options):\n    return available_options[0]['value']\n\n# when the country or city is updated this will print the current values\n@app.callback(\n    Output('display-selected-values', 'children'),\n    Input('countries-radio', 'value'),\n    Input('cities-radio', 'value'))\ndef set_display_children(selected_country, selected_city):\n    return u'{} is a city in {}'.format(\n        selected_city, selected_country,\n    )\n\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# - The first callback updates the available options in the second `dcc.RadioItems` component based off of the selected value in the first `dcc.RadioItems` component.
# 
# - The second callback sets an initial value when the options property changes: it sets it to the first value in that options array.
# 
# - The final callback displays the selected value of each component.
#   - If you change the value of the countries `dcc.RadioItems` component, Dash will wait until the value of the cities component is updated before calling the final callback. This prevents your callbacks from being called with inconsistent state like with "America" and "Montréal"
# 

# ## Controls with States
# 
# Sometimes you would like to wait until the user finishes making their selections before computing an update. This is called a **State**
# 

# `State` allows you to pass along extra values without firing the callbacks.
# 
# Here's the same example as above but with the two `dcc.Input` components as `State` and a new button component as an `Input`.
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_10.py', '\n# -*- coding: utf-8 -*-\nfrom dash import Dash, dcc, html\nfrom dash.dependencies import Input, Output, State\n\nexternal_stylesheets = [\'https://codepen.io/chriddyp/pen/bWLwgP.css\']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\napp.layout = html.Div([\n    dcc.Input(id=\'input-1-state\', type=\'text\', value=\'Montréal\'),\n    dcc.Input(id=\'input-2-state\', type=\'text\', value=\'Canada\'),\n    html.Button(id=\'submit-button-state\', n_clicks=0, children=\'Submit\'),\n    html.Div(id=\'output-state\')\n])\n\n\n@app.callback(Output(\'output-state\', \'children\'),\n              Input(\'submit-button-state\', \'n_clicks\'),\n              State(\'input-1-state\', \'value\'),\n              State(\'input-2-state\', \'value\'))\ndef update_output(n_clicks, input1, input2):\n    return u\'\'\'\n        The Button has been pressed {} times,\n        Input 1 is "{}",\n        and Input 2 is "{}"\n    \'\'\'.format(n_clicks, input1, input2)\n\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# ## Interactive Visualizations
# 

# The `dcc.Graph` component has four attributes that can change through user-interaction: `hoverData`, `clickData`, `selectedData`, `relayoutData`. These properties update when you hover over points, click on points, or select regions of points in a graph.
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_11.py', '\nimport json\n\nfrom dash import Dash, dcc, html\nfrom dash.dependencies import Input, Output\nimport plotly.express as px\nimport pandas as pd\n\nexternal_stylesheets = [\'https://codepen.io/chriddyp/pen/bWLwgP.css\']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\nstyles = {\n    \'pre\': {\n        \'border\': \'thin lightgrey solid\',\n        \'overflowX\': \'scroll\'\n    }\n}\n\ndf = pd.DataFrame({\n    "x": [1,2,1,2],\n    "y": [1,2,3,4],\n    "customdata": [1,2,3,4],\n    "fruit": ["apple", "apple", "orange", "orange"]\n})\n\nfig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])\n\nfig.update_layout(clickmode=\'event+select\')\n\nfig.update_traces(marker_size=20)\n\napp.layout = html.Div([\n    dcc.Graph(\n        id=\'basic-interactions\',\n        figure=fig\n    ),\n\n    html.Div(className=\'row\', children=[\n        html.Div([\n            dcc.Markdown("""\n                **Hover Data**\n\n                Mouse over values in the graph.\n            """),\n            html.Pre(id=\'hover-data\', style=styles[\'pre\'])\n        ], className=\'three columns\'),\n\n        html.Div([\n            dcc.Markdown("""\n                **Click Data**\n\n                Click on points in the graph.\n            """),\n            html.Pre(id=\'click-data\', style=styles[\'pre\']),\n        ], className=\'three columns\'),\n\n        html.Div([\n            dcc.Markdown("""\n                **Selection Data**\n\n                Choose the lasso or rectangle tool in the graph\'s menu\n                bar and then select points in the graph.\n\n                Note that if `layout.clickmode = \'event+select\'`, selection data also\n                accumulates (or un-accumulates) selected data if you hold down the shift\n                button while clicking.\n            """),\n            html.Pre(id=\'selected-data\', style=styles[\'pre\']),\n        ], className=\'three columns\'),\n\n        html.Div([\n            dcc.Markdown("""\n                **Zoom and Relayout Data**\n\n                Click and drag on the graph to zoom or click on the zoom\n                buttons in the graph\'s menu bar.\n                Clicking on legend items will also fire\n                this event.\n            """),\n            html.Pre(id=\'relayout-data\', style=styles[\'pre\']),\n        ], className=\'three columns\')\n    ])\n])\n\n\n@app.callback(\n    Output(\'hover-data\', \'children\'),\n    Input(\'basic-interactions\', \'hoverData\'))\ndef display_hover_data(hoverData):\n    return json.dumps(hoverData, indent=2)\n\n\n@app.callback(\n    Output(\'click-data\', \'children\'),\n    Input(\'basic-interactions\', \'clickData\'))\ndef display_click_data(clickData):\n    return json.dumps(clickData, indent=2)\n\n\n@app.callback(\n    Output(\'selected-data\', \'children\'),\n    Input(\'basic-interactions\', \'selectedData\'))\ndef display_selected_data(selectedData):\n    return json.dumps(selectedData, indent=2)\n\n\n@app.callback(\n    Output(\'relayout-data\', \'children\'),\n    Input(\'basic-interactions\', \'relayoutData\'))\ndef display_relayout_data(relayoutData):\n    return json.dumps(relayoutData, indent=2)\n\n\nif __name__ == \'__main__\':\n    app.run_server(debug=True)')


# ### Update Graph on Hover
# 

# In[ ]:


get_ipython().run_cell_magic('writefile', 'dash_scripts/dash_12.py', "\nfrom dash import Dash, html, dcc, Input, Output\nimport pandas as pd\nimport plotly.express as px\n\nexternal_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']\n\napp = Dash(__name__, external_stylesheets=external_stylesheets)\n\ndf = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')\n\n\napp.layout = html.Div([\n    html.Div([\n\n        html.Div([\n            dcc.Dropdown(\n                df['Indicator Name'].unique(),\n                'Fertility rate, total (births per woman)',\n                id='crossfilter-xaxis-column',\n            ),\n            dcc.RadioItems(\n                ['Linear', 'Log'],\n                'Linear',\n                id='crossfilter-xaxis-type',\n                labelStyle={'display': 'inline-block', 'marginTop': '5px'}\n            )\n        ],\n        style={'width': '49%', 'display': 'inline-block'}),\n\n        html.Div([\n            dcc.Dropdown(\n                df['Indicator Name'].unique(),\n                'Life expectancy at birth, total (years)',\n                id='crossfilter-yaxis-column'\n            ),\n            dcc.RadioItems(\n                ['Linear', 'Log'],\n                'Linear',\n                id='crossfilter-yaxis-type',\n                labelStyle={'display': 'inline-block', 'marginTop': '5px'}\n            )\n        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})\n    ], style={\n        'padding': '10px 5px'\n    }),\n\n    html.Div([\n        dcc.Graph(\n            id='crossfilter-indicator-scatter',\n            hoverData={'points': [{'customdata': 'Japan'}]}\n        )\n    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),\n    html.Div([\n        dcc.Graph(id='x-time-series'),\n        dcc.Graph(id='y-time-series'),\n    ], style={'display': 'inline-block', 'width': '49%'}),\n\n    html.Div(dcc.Slider(\n        df['Year'].min(),\n        df['Year'].max(),\n        step=None,\n        id='crossfilter-year--slider',\n        value=df['Year'].max(),\n        marks={str(year): str(year) for year in df['Year'].unique()}\n    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})\n])\n\n\n@app.callback(\n    Output('crossfilter-indicator-scatter', 'figure'),\n    Input('crossfilter-xaxis-column', 'value'),\n    Input('crossfilter-yaxis-column', 'value'),\n    Input('crossfilter-xaxis-type', 'value'),\n    Input('crossfilter-yaxis-type', 'value'),\n    Input('crossfilter-year--slider', 'value'))\ndef update_graph(xaxis_column_name, yaxis_column_name,\n                 xaxis_type, yaxis_type,\n                 year_value):\n    dff = df[df['Year'] == year_value]\n\n    fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],\n            y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],\n            hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']\n            )\n\n    fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])\n\n    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')\n\n    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')\n\n    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')\n\n    return fig\n\n\ndef create_time_series(dff, axis_type, title):\n\n    fig = px.scatter(dff, x='Year', y='Value')\n\n    fig.update_traces(mode='lines+markers')\n\n    fig.update_xaxes(showgrid=False)\n\n    fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')\n\n    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',\n                       xref='paper', yref='paper', showarrow=False, align='left',\n                       text=title)\n\n    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})\n\n    return fig\n\n# These are the hoverdata updates\n@app.callback(\n    Output('x-time-series', 'figure'),\n    Input('crossfilter-indicator-scatter', 'hoverData'),\n    Input('crossfilter-xaxis-column', 'value'),\n    Input('crossfilter-xaxis-type', 'value'))\ndef update_y_timeseries(hoverData, xaxis_column_name, axis_type):\n    country_name = hoverData['points'][0]['customdata']\n    dff = df[df['Country Name'] == country_name]\n    dff = dff[dff['Indicator Name'] == xaxis_column_name]\n    title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)\n    return create_time_series(dff, axis_type, title)\n\n\n@app.callback(\n    Output('y-time-series', 'figure'),\n    Input('crossfilter-indicator-scatter', 'hoverData'),\n    Input('crossfilter-yaxis-column', 'value'),\n    Input('crossfilter-yaxis-type', 'value'))\ndef update_x_timeseries(hoverData, yaxis_column_name, axis_type):\n    dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]\n    dff = dff[dff['Indicator Name'] == yaxis_column_name]\n    return create_time_series(dff, axis_type, yaxis_column_name)\n\n\nif __name__ == '__main__':\n    app.run_server(debug=True)")


# ### Crossfiltering
# 
# There are many times in machine learning where you want to filter high dimensional data.
# 
# One common way to do this is called parallel coordinate plots.
# 

# In[2]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go

import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/bcdunbar/datasets/master/parcoords_data.csv"
)

fig = go.Figure(
    data=go.Parcoords(
        line=dict(
            color=df["colorVal"],
            colorscale="Electric",
            showscale=True,
            cmin=-4000,
            cmax=-100,
        ),
        dimensions=list(
            [
                dict(
                    range=[32000, 227900],
                    constraintrange=[100000, 150000],
                    label="Block Height",
                    values=df["blockHeight"],
                ),
                dict(range=[0, 700000], label="Block Width", values=df["blockWidth"]),
                dict(
                    tickvals=[0, 0.5, 1, 2, 3],
                    ticktext=["A", "AB", "B", "Y", "Z"],
                    label="Cyclinder Material",
                    values=df["cycMaterial"],
                ),
                dict(
                    range=[-1, 4],
                    tickvals=[0, 1, 2, 3],
                    label="Block Material",
                    values=df["blockMaterial"],
                ),
                dict(
                    range=[134, 3154],
                    visible=True,
                    label="Total Weight",
                    values=df["totalWeight"],
                ),
                dict(
                    range=[9, 19984],
                    label="Assembly Penalty Wt",
                    values=df["assemblyPW"],
                ),
                dict(range=[49000, 568000], label="Height st Width", values=df["HstW"]),
            ]
        ),
    )
)
fig.show()


# ## Other Cool Examples
# 

# In[3]:


IFrame(src="https://dash.gallery/Portal/", width=800, height=800)


# ## Other Interactive Visualization Tools
# 

# ### Bokeh
# 
# Very similar to plotly. Somewhat simplier to deploy but less powerful
# 

# In[4]:


IFrame(
    src="https://docs.bokeh.org/en/latest/docs/gallery.html#gallery",
    width=800,
    height=800,
)


# ### Napari
# 
# - napari is a fast, interactive, multi-dimensional image viewer for Python.
# - It’s designed for browsing, annotating, and analyzing large multi-dimensional images.
# - It’s built on top of Qt (for the GUI), vispy (for performant GPU-based rendering), and the scientific Python stack (numpy, scipy).
# - It includes critical viewer features out-of-the-box, such as support for large multi-dimensional data, and layering and annotation. By integrating closely with the Python ecosystem, napari can be easily coupled to leading machine learning and image analysis tools (e.g. scikit-image, scikit-learn, TensorFlow, PyTorch), enabling more user-friendly automated analysis.
# 

# In[5]:


IFrame(src="https://napari.org/stable/", width=800, height=800)


# ![](figs/rxrx19_stardist2-1.gif)
# 

# In[ ]:




