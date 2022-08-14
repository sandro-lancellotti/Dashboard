import numpy as np
import pandas as pd
import dash

import plotly.express as px
from dash.dependencies import Input, Output
from dash import no_update
import plotly.graph_objects as go

from dash import html
from dash import dcc
import statsmodels.api as sm



DEM_basilicata = np.loadtxt('DEM_basilicata.txt')
mask = DEM_basilicata.copy()
mask[mask > 0] = 1

def parametri_reg(x, y):
    """ Questa funzione calcola i parametri di regressione: coefficiente angolare, intercetta e relativi errori """
    x_reg = np.empty(shape=(len(x), 2))
    x_reg[:, 0] = 1
    x_reg[:, 1] = x
    ols = sm.OLS(y, x_reg)
    ols_result = ols.fit()
    return (ols_result.params, ols_result.bse)



def parametri(year, month, kind):
    """Riceve in input year e month e crea un dataframe con i dati e calcola m richiamando parametri_reg"""
    filtered_df = data[(data['year'] == year) & (data['month'] == month)]
    (_, m), _ = parametri_reg(filtered_df['altitude'], filtered_df[kind])
    return m

app = dash.Dash(__name__)

data =  pd.read_csv('complete.csv')
app.layout = html.Div(children=[html.H1('TEMPERATURE LAPS RATE',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 24}),
    #outer division starts
     html.Div([
                   # First inner divsion for  adding dropdown helper text for Selected Drive wheels
                    html.Div(
                            html.H2('Temperature:', style={'margin-right': '2em'})
                     ),


                    dcc.Dropdown(
                        id= 'dropdown-temperature',
                        options= [
                            {'label': 'Minimum', 'value': 'Temperatura Aria (°C)_min'},
                            {'label': 'Average', 'value': 'Temperatura Aria (°C)_med'},
                            {'label': 'Maximum', 'value': 'Temperatura Aria (°C)_max'},
                        ],
                        value='Temperatura Aria (°C)_med'


                    ),

                    html.Div(
                            html.H2('Month:', style={'margin-right': '2em'})
                     ),

                    dcc.Dropdown(
                        id= 'dropdown-month',
                        options= [
                            {'label': 'January', 'value': 1},
                            {'label': 'February', 'value': 2},
                            {'label': 'March', 'value': 3},
                            {'label': 'April', 'value': 4},
                            {'label': 'May', 'value': 5},
                            {'label': 'June', 'value': 6},
                            {'label': 'July', 'value': 7},
                            {'label': 'August', 'value': 8},
                            {'label': 'September', 'value': 9},
                            {'label': 'October', 'value': 10},
                            {'label': 'November', 'value': 11},
                            {'label': 'December', 'value': 12},

                        ],
                        value=1


                    ),
                    #Second Inner division for adding 2 inner divisions for 2 output graphs
                    html.Div([

                        html.Div([ ], id='plot1'),
                        html.Div([ ], id='plot2')

                    ], style={'display': 'flex'}),

                    html.Div([

                        html.Div([ ], id='plot3'),
                        html.Div([ ], id='plot4')

                    ], style={'display': 'flex'}),


    ])
    #outer division ends

])
#layout ends

# @app.callback Decorator
@app.callback([Output(component_id='plot1', component_property='children'),
               Output(component_id='plot2', component_property='children'),
               Output(component_id='plot3', component_property='children'),
               Output(component_id='plot4', component_property='children')
               ],
               [Input(component_id='dropdown-month', component_property='value'),
                Input(component_id='dropdown-temperature', component_property='value')
                ])


#Place to define the callback function .
def display_selected_drive_charts(month, kind):

    filtered_df = data[data['month']==int(month)].groupby(['source_code', 'source_name','altitude'],
                                                          as_index=False)[kind].mean()
    filtered_df = filtered_df

    (q, m), _ = parametri_reg(filtered_df['altitude'], filtered_df[kind])
    mappa = (DEM_basilicata * m + q) * mask
    mappa[mappa == 0] = -15

    anno_mese_coef = data[['year', 'month']].drop_duplicates()
    anno_mese_coef['coef'] = [parametri(x, y, kind) for x, y in anno_mese_coef.values]
    statistiche = anno_mese_coef.groupby(['month'], as_index=False)['coef'].agg([np.std, np.mean])


    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x = filtered_df['altitude'], y = filtered_df[kind], name="Station",
                         mode='markers', text = filtered_df['source_name'],
                         hovertemplate = 'Temperature: %{y:.2} °C' +'<br>Altitude: %{x:.4} m'+
                          '<br>Location: %{text}' +'<extra></extra>'))

    x = np.array(range(0, max(filtered_df['altitude'])+100, 20))
    fig1.add_trace(go.Scatter(x = x, y = m*x +q, name="Regression Line", mode = 'lines',
                          hovertemplate = 'Temperature: %{y:.2} °C' +'<br>Altitude: %{x:.4} m'+'<extra></extra>' ))


    fig1.update_xaxes(range=[0, 900])
    fig1.update_yaxes(range=[-20, 45])
    fig1.update_layout(title = 'Altitude vs Temperature', xaxis_title= 'Altitude (m)', yaxis_title='Temperature (°C)')

    fig2 = px.imshow(mappa, zmin = -1, zmax=40, color_continuous_scale = ['black', 'blue','yellow',  'red'],
                     title = 'Heat map of Basilicata',
                    labels={'color': 'Temperature'})

    fig2.update_layout(yaxis={'visible': False, 'showticklabels': False},
                       xaxis={'visible': False, 'showticklabels': False})
    fig2.update_traces(hovertemplate=None, hoverinfo='skip')



    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x = anno_mese_coef['month'], y = anno_mese_coef['coef'], name="Shape",
                         mode='markers', text = anno_mese_coef['year'],
                        hovertemplate = 'Shape: %{y:.2e}' +'<br>Year: %{text}'+'<extra></extra>'))

    fig3.add_trace(go.Scatter(x = statistiche.index, y = statistiche['mean'], name="Average Shape",
                         mode='markers',hovertemplate = 'Average shape: %{y:.2e}'+'<extra></extra>'))

    fig3.add_trace(go.Scatter(x = statistiche.index, y = statistiche['mean']+statistiche['std'],
                        showlegend = False,mode = 'lines', fill = 'none', line_color='orange', hoverinfo='skip'))
    fig3.add_trace(go.Scatter(x = statistiche.index, y = statistiche['mean']-statistiche['std'],
                          name='Std Interval',mode = 'lines',  fill = 'tonexty', line_color='orange', hoverinfo='skip'))

    fig3.update_layout(title = 'Monthly shape distribution over the twenty-year period 2000-2020',
                    legend={'traceorder':'normal'},
                   xaxis= dict(tickmode = 'array',tickvals = list(range(1, 13)),
                   ticktext =['January', 'February', 'March','April','May', 'June', 'July',
                              'August', 'September', 'October','November',  'December']),
                    yaxis = dict( tickformat='.0e'))


    fig4 = go.Figure()

    fig4.add_trace(go.Indicator(mode="number", value=m, domain={'row': 0, 'column': 1}))
    fig4.update_layout(title='Shape', )


    return [dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
            dcc.Graph(figure=fig4)]




if __name__ == '__main__':
    app.run_server()

