from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from flask import Flask
import pandas as pd
import dash
from datetime import datetime, timedelta
import datetime
from dash.exceptions import PreventUpdate
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
import numpy as np
import json
import requests
from pytz import timezone
import pytz
import datetime
from datetime import date
from dotenv import load_dotenv
import os

server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=[dbc.themes.FLATLY], prevent_initial_callbacks=True)
app.title = 'Dashboard'

unique_flight_records = pd.read_csv('data/prepared/unique_flight_number_data.csv')
df_by_date = pd.read_csv('data/prepared/delays-by-date.csv')
df_by_airport = pd.read_csv('data/prepared/delays-by-airport.csv')
df_by_airline = pd.read_csv('data/prepared/delays-by-airline.csv')
holidays = pd.read_csv('data/prepared/holidays.csv')
holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])

airports = list(df_by_airport['ORIGIN'].unique())
overall_delays = px.line(df_by_date,
                                x=df_by_date.date,
                                y=df_by_date['percent-delayed'],
                                labels={
                                    "x": "Date",
                                    "y": "Severe Delays"}, title="Daily Severe Airport Delays")
overall_delays.update_layout(xaxis_rangeslider_visible=True)
overall_delays.update_traces(line_color='#f64c72')


def remove_timezone(dt):
    # HERE `dt` is a python datetime
    # object that used .replace() method

    return dt.replace(tzinfo=None)

def takeoff_hour_rounder(time):
    '''
    Function takes in a time and returns time rounded to the
    nearest hour by adding a timedelta hour if minute >= 30
    '''
    return (time.replace(second=0, microsecond=0, minute=0, hour=time.hour)
               +timedelta(hours=time.minute//30))

app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H2("Air Travel Delays"), width={'size': 12, 'offset': 0}), style={'textAlign': 'center', 'paddingBottom': '2%', 'paddingTop': '2%'}),
        dbc.Row(dbc.Col(html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent sit amet neque dolor. Proin tincidunt ut ligula sed blandit. Nullam rutrum eros non pretium vulputate."), width={'size': 12, 'offset': 0}), style={'textAlign': 'center', 'paddingBottom': '2%', 'paddingTop': '2%'}),
        dbc.Row(dbc.Col(html.P("First, enter your flight number (e.g. AA6300)"), width={'size': 12, 'offset': 0}), style={'textAlign': 'left', 'paddingTop': '1%', 'font-weight': 'bold'}, id='instructions1'),
        dbc.Row(dbc.Col(html.Div([html.Div(dcc.Input(id='input-on-submit', type='text', debounce=True)),
                                  html.P("Next, enter your flight date (up to 14 days from today):", style= {'paddingBottom': '5px', 'paddingTop': '10px', 'font-weight': 'bold'}, id='instructions2'),
                                  html.Div([
                                        dcc.DatePickerSingle(
                                            id='my-date-picker-single',
                                            min_date_allowed=date(1995, 8, 5),
                                            max_date_allowed=date(2022, 12, 31),
                                            initial_visible_month=date(2022, 9, 15),
                                            date=date(2022, 9, 15)
                                        )
                                    ]),
                                  html.P("Now, tell us your flight time:", style={'paddingBottom': '5px', 'paddingTop': '10px', 'font-weight': 'bold'}, id='instructions3'),
                                  html.Div(dcc.Dropdown(id='time-selection')),
                                  html.P("Finally, hit \"Predict\" to have our model check your flight! ", style= {'paddingBottom': '5px', 'paddingTop': '10px', 'font-weight': 'bold'}, id='instructions4'),
                                  html.Button('Predict', id='submit-val', n_clicks=0),
                                  html.Div(id='container-button-basic',
                                          children='', style= {'paddingBottom': '5px', 'paddingTop': '10px'}),
                                  html.Div(id='prediction')]))),
        dbc.Row(dbc.Col(dcc.Graph(
            id='example-graph',
            figure=overall_delays
        ))),
        dbc.Row(dbc.Col(html.H3("Delays by Airport"), width={'size': 12, 'offset': 0}), style={'textAlign': 'center'}),
        dbc.Row(dbc.Col(html.P("Select an airport to see severe delays there over time."), width={'size': 12, 'offset': 0}), style={'textAlign': 'center', 'paddingBottom': '1%', 'paddingTop': '1%'}),
        dbc.Row(dbc.Col(html.Div([
            dcc.Dropdown(
                id='fig_dropdown-2',
                options=[{'label': x, 'value': x} for x in airports],
                value='JFK')]), width={"size": 6, "offset": 3},
                            )
                ),
        dbc.Row(dbc.Col(html.Div(id='fig_plot-2'), width={"size": 12},
                        ))
                                ])
@app.callback(dash.dependencies.Output('time-selection', 'options'),
              Input('input-on-submit', 'value'),
              Input('submit-val', 'n_clicks')
              )

def get_flight_times(value, n_clicks):
    flight_num = value
    all_flights = list(unique_flight_records['flight-number'].unique())
    if flight_num in all_flights:
        flight_time_options = list(
            unique_flight_records.loc[unique_flight_records['flight-number'] == str(flight_num)][
                'CRS_DEP_TIME'].unique())
        return flight_time_options
    else:
        flight_time_options = ['It seems we don\'t have information on this flight. Try entering another one.']
        return flight_time_options

@app.callback(
     dash.dependencies.Output('input-on-submit', 'style'),
     dash.dependencies.Output('instructions1', 'style'),
     dash.dependencies.Output('instructions2', 'style'),
     dash.dependencies.Output('instructions3', 'style'),
     dash.dependencies.Output('instructions4', 'style'),
     dash.dependencies.Output('my-date-picker-single', 'style'),
     dash.dependencies.Output('demo-dropdown', 'style'),
     dash.dependencies.Output('submit-val', 'style'),
     dash.dependencies.Input('submit-val', 'n_clicks'),
     dash.dependencies.State('input-on-submit', 'value')
)
def display_results(n_clicks):
    if n_clicks > 0:
        hide_element = {'display': 'none'}
        return hide_element, hide_element, hide_element, hide_element, hide_element, hide_element, hide_element, hide_element
@app.callback([dash.dependencies.Output('prediction', 'children')],
              [dash.dependencies.Input('my-date-picker-single', 'date'),
               dash.dependencies.Input('time-selection', 'value'),
               dash.dependencies.Input('submit-val', 'n_clicks'),
               dash.dependencies.State('input-on-submit', 'value')])

def predict(date_value, time, n_clicks, value):
    # if n_clicks == 0:
    #     raise PreventUpdate
    # else:
        all_flights = list(unique_flight_records['flight-number'].unique())
        if date_value is not None:
            date_object = date.fromisoformat(date_value)
            date_string = date_object.strftime('%Y-%m-%d')
        flight_num = value
        flight_time = time
        if date_value is not None and time is not None and value in all_flights and n_clicks > 0:
            flight_number = value
            flight_date = date_string
            flight_details = unique_flight_records.loc[
                unique_flight_records['exact-flight'] == flight_time + flight_number]

            flight_details['FL_DATE_LOCAL'] = pd.to_datetime(flight_date + ' ' + flight_time)
            flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].astype('datetime64[ns]')
            flight_details['FL_DATE_LOCAL'] = flight_details.apply(
                lambda x: x['FL_DATE_LOCAL'].replace(tzinfo=timezone(x['origin-tz'])), axis=1)
            flight_details['flight_duration'] = pd.to_timedelta(flight_details['CRS_ELAPSED_TIME'], 'm')
            flight_details['FL_ARR_DATE_REL_ORIGIN'] = flight_details['FL_DATE_LOCAL'] + flight_details[
                'flight_duration']
            # And now we convert arrival time and date to a local time
            flight_details['FL_ARR_DATE_LOCAL'] = flight_details.apply(
                lambda x: x['FL_ARR_DATE_REL_ORIGIN'].tz_convert(x['dest-tz']), axis=1)

            flight_details['DAY_OF_WEEK'] = flight_details['FL_DATE_LOCAL'].dt.dayofweek
            flight_details['DAY_OF_WEEK'] = flight_details['DAY_OF_WEEK'] + 1
            flight_details['ARR_DAY_OF_WEEK'] = flight_details['FL_ARR_DATE_LOCAL'].dt.dayofweek
            flight_details['ARR_DAY_OF_WEEK'] = flight_details['ARR_DAY_OF_WEEK'] + 1

            day_of_week_translation = {1: 'Monday',
                                       2: 'Tuesday',
                                       3: 'Wednesday',
                                       4: 'Thursday',
                                       5: 'Friday',
                                       6: 'Saturday',
                                       7: 'Sunday'}

            flight_details['DAY_OF_WEEK'].replace(day_of_week_translation, inplace=True)
            flight_details['ARR_DAY_OF_WEEK'].replace(day_of_week_translation, inplace=True)

            flight_details['FL_DATE_LOCAL_ROUNDED'] = flight_details['FL_DATE_LOCAL'].apply(takeoff_hour_rounder)
            flight_details['FL_ARR_DATE_LOCAL_ROUNDED'] = flight_details['FL_ARR_DATE_LOCAL'].apply(
                takeoff_hour_rounder)

            # Takeoff Congestion Key
            flight_details['takeoff-congestion-key'] = flight_details['ORIGIN'] \
                                                       + flight_details['DAY_OF_WEEK'].astype(str) \
                                                       + flight_details['FL_DATE_LOCAL_ROUNDED'].dt.hour.astype(
                str).str.zfill(2)

            # Arrival Congestion Key
            flight_details['arrival-congestion-key'] = flight_details['DEST'] \
                                                       + flight_details['ARR_DAY_OF_WEEK'].astype(str) \
                                                       + flight_details['FL_ARR_DATE_LOCAL_ROUNDED'].dt.hour.astype(
                str).str.zfill(2)

            # Now we add congestion data to our main dataframe
            congestion = pd.read_csv('data/prepared/airport_congestion_by_hour.csv')
            flight_details = pd.merge(flight_details, congestion, left_on='takeoff-congestion-key',
                                      right_on='congestion-key')
            # updating key
            congestion = congestion.add_prefix('dest-')
            # Now data on the congestion conditions of the airport where the flight is arriving
            flight_details = pd.merge(flight_details, congestion, left_on='arrival-congestion-key',
                                      right_on='dest-congestion-key')

            # We won't be needitn timezone info anymore, so let's remove it
            flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].apply(remove_timezone)

            flight_details = pd.merge_asof(flight_details, holidays, left_on='FL_DATE_LOCAL', right_on='holiday_date',
                                           direction='nearest', tolerance=pd.Timedelta(days=7))
            flight_details['days-from-holiday'] = (
                        flight_details['FL_DATE_LOCAL'] - flight_details['holiday_date']).dt.days
            flight_details['days-from-holiday'] = flight_details['days-from-holiday'].astype(str)
            flight_details['days-from-specific-holiday'] = flight_details['holiday_name'] + '_' + flight_details[
                'days-from-holiday'].astype(str)
            flight_details['days-from-specific-holiday'].fillna('no-close-holiday', inplace=True)
            flight_details['days-to-forecast'] = (flight_details['FL_DATE_LOCAL'] - pd.Timestamp('now')).dt.days

            forecast_data_url = 'http://api.weatherapi.com/v1/forecast.json'
            load_dotenv()
            api_key = os.getenv("API_KEY")
            start_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
            end_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
            location = flight_details['origin-lat-long'][0]
            days = flight_details['days-to-forecast'][0].astype(str)

            r_origin = requests.get(
                forecast_data_url + '?key=' + api_key + '&q=' + location + '&' + days + '&dt=' + start_date + '&end_dt=' + end_date)
            d_origin = json.loads(r_origin.text)

            dest_location = flight_details['dest-lat-long_x'][0]

            r_dest = requests.get(
                forecast_data_url + '?key=' + api_key + '&q=' + dest_location + '&' + days + '&dt=' + start_date + '&end_dt=' + end_date)
            d_dest = json.loads(r_dest.text)

            flight_details['maxtemp'] = d_origin['forecast']['forecastday'][0]['day']['maxtemp_c']
            flight_details['mintemp'] = d_origin['forecast']['forecastday'][0]['day']['mintemp_c']
            flight_details['avgtemp'] = d_origin['forecast']['forecastday'][0]['day']['avgtemp_c']
            flight_details['totalprecip'] = d_origin['forecast']['forecastday'][0]['day']['totalprecip_mm']
            flight_details['avgvis'] = d_origin['forecast']['forecastday'][0]['day']['avgvis_km']
            flight_details['maxwind'] = d_origin['forecast']['forecastday'][0]['day']['maxwind_kph']
            flight_details['avghumidity'] = d_origin['forecast']['forecastday'][0]['day']['avghumidity']
            flight_details['dest-maxtemp'] = d_dest['forecast']['forecastday'][0]['day']['maxtemp_c']
            flight_details['dest-mintemp'] = d_dest['forecast']['forecastday'][0]['day']['mintemp_c']
            flight_details['dest-avgtemp'] = d_dest['forecast']['forecastday'][0]['day']['avgtemp_c']
            flight_details['dest-totalprecip'] = d_dest['forecast']['forecastday'][0]['day']['totalprecip_mm']
            flight_details['dest-avgvis'] = d_dest['forecast']['forecastday'][0]['day']['avgvis_km']
            flight_details['dest-maxwind'] = d_dest['forecast']['forecastday'][0]['day']['maxwind_kph']
            flight_details['dest-avghumidity'] = d_dest['forecast']['forecastday'][0]['day']['avghumidity']

            flight_details['MONTH'] = flight_details['FL_DATE_LOCAL'].dt.month
            flight_details['DAY_OF_MONTH'] = flight_details['FL_DATE_LOCAL'].dt.day

            # Splitting features & target
            model_cols = ['DISTANCE',
                          'origin-elevation',
                          'dest-elevation',
                          'CRS_ELAPSED_TIME',
                          'avg-takeoff-congestion',
                          'avg-arrival-congestion',
                          'dest-avg-takeoff-congestion',
                          'dest-avg-arrival-congestion',
                          'maxtemp',
                          'mintemp',
                          'avgtemp',
                          'totalprecip',
                          'avgvis',
                          'maxwind',
                          'avghumidity',
                          'dest-maxtemp',
                          'dest-mintemp',
                          'dest-avgtemp',
                          'dest-totalprecip',
                          'dest-avgvis',
                          'dest-maxwind',
                          'dest-avghumidity',
                          'takeoff-mins-from-midnight',
                          'landing-mins-from-midnight',
                          'MONTH',
                          'DAY_OF_MONTH',
                          'DAY_OF_WEEK',
                          'MKT_CARRIER',
                          'OP_CARRIER',
                          'ORIGIN',
                          'DEST',
                          'ARR_DAY_OF_WEEK',
                          'days-from-specific-holiday']

            X = flight_details[model_cols].copy()

            filename = 'model.sav'
            loaded_model = pickle.load(open(filename, 'rb'))
            prediction = loaded_model.predict(X)
            if prediction[0] == 'No':
                return ['No major delay expected']
            else:
                return ['Our model thinks your flight is likely to experience a major delay.']

@app.callback(
dash.dependencies.Output('fig_plot-2', 'children'),
[dash.dependencies.Input('fig_dropdown-2', 'value')])
def update_output(fig_name):
    return name_to_figure(fig_name)
def name_to_figure(fig_name):
    fig = px.line(df_by_date,
                  x=df_by_airport.loc[df_by_airport['ORIGIN'] == '{}'.format(fig_name)]['date'],
                  y=df_by_airport.loc[df_by_airport['ORIGIN'] == '{}'.format(fig_name)]['percent-delayed'],
                  labels={"x": "Date",
                          "y": "Severe Delays"},
                  title="Daily Severe Airport Delays at {} Airport".format(fig_name))
    fig.update_layout(xaxis_rangeslider_visible=True)
    return dcc.Graph(figure=fig)

if __name__ == '__main__':
    app.run(debug=True)