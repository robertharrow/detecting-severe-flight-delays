from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import html
import plotly.express as px
from flask import Flask
import pandas as pd
import dash
from datetime import timedelta
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)
import json
import requests
from pytz import timezone
from datetime import date
from dotenv import load_dotenv
import os
from dash.exceptions import PreventUpdate
import numpy as np

server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=[dbc.themes.FLATLY], prevent_initial_callbacks=True)
app.title = '  air-travel-delays'

airport_lookup =  pd.read_csv('data/prepared/airport_lookup.csv')
origin_options = sorted(list(airport_lookup['ORIGIN'].unique()))
relevant_airlines = sorted(['Southwest', 'Delta', 'SkyWest', 'American Airlines', 'United Airlines', 'JetBlue', 'Alaska Airlines', 'Spirit Airlines'])
df_by_airport = pd.read_csv('data/prepared/delays-by-airport.csv')
df_by_holiday = pd.read_csv('data/prepared/delays-by-holiday.csv')
df_weekdays_times = pd.read_csv('data/prepared/df_by_timeofday_weekday.csv')
holidays = pd.read_csv('data/prepared/holidays.csv')
holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date'])
airports = sorted(list(df_by_airport['ORIGIN'].unique()))

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

disclaimer_text = 'Disclaimer: This model will try to predict whether a future flight will experience a severe delay (defined as 1 hour or more). The model isn\'t guaranteed to be correct.' \
                  ' Currently, it also only supports flights originating from 62 major US airports and the top 8 major airlines'

app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H2("Will your flight be severely delayed?"), width={'size': 12, 'offset': 0}), style={'textAlign': 'center', 'paddingBottom': '2%', 'paddingTop': '2%'}),
        dbc.Row(dbc.Col(html.P(disclaimer_text), width={'size': 6, 'offset': 3}), style={'textAlign': 'center', 'paddingBottom': '10px', 'paddingTop': '10px', 'font-style': 'italic'}),
        dbc.Row(dbc.Col(html.Div([html.P("Step 1: Select your Airline"),
                                  html.Div(dcc.Dropdown(id='select-airline', options=[{'label': x, 'value': x} for x in relevant_airlines])),
                                  html.P("Step 2: Select the airport you\'re flying FROM:", style={'paddingTop': '10px'}),
                                  html.Div(dcc.Dropdown(id='select-origin', options=[{'label': x, 'value': x} for x in origin_options])),
                                  html.P("Step 3: Select the airport you\'re flying TO:", style={'paddingTop': '10px'}),
                                  html.Div(dcc.Dropdown(id='select-dest')),
                                  html.P("Step 4: Enter your flight date (up to 14 days from today):", style={'paddingTop': '10px'}),
                                  html.Div([
                                        dcc.DatePickerSingle(
                                            id='my-date-picker-single',
                                            min_date_allowed=date.today(),
                                            max_date_allowed=date.today() + timedelta(days=14),
                                            initial_visible_month=date.today(),
                                            date=date.today()
                                        )
                                    ]),
                                  html.P("Step 5: Enter your flight time (24 Hour Format):", style={'paddingTop': '10px'}),
                                  html.Div(
                                      [dcc.Input(id="hour-time", min=0, max=24, step=1, type="number", debounce=True, placeholder="HH", style={'width': '15%'}),
                                       dcc.Input(id="minutes-time", min=0, max=60, step=1, type="number", debounce=True, placeholder="MM", style={'width': '15%'})]),
                                  html.P("Step 6: Enter your flight duration:", style={'paddingTop': '10px'}),
                                  html.Div(
                                      [dcc.Input(id="hour-duration", min=0, max=24, step=1, type="number", debounce=True, placeholder="Hours", style={'width': '15%'}),
                                          dcc.Input(id="minutes-duration", min=0, max=60, step=1, type="number", debounce=True, placeholder="Minutes", style={'width': '15%'})]),
                                  html.P("Press the \"Predict\" button! ", style={'paddingTop': '10px'}),
                                  html.Button('Predict', id='submit-val', style={'background-color': '#4681f4', 'color': 'white', 'border':'2px solid #4681f4', 'width': '100%'})],
                                  id='model-inputs', style={'font-weight': 'bold'}), width={'size': 6, 'offset': 3})),
        dbc.Row(dbc.Col(html.H4(id='prediction', style={'font-weight': 'bold', 'paddingBottom': '2%', 'paddingTop': '1%', 'border-style': 'solid', 'border-color': '#5dbea3'}), width={'size': 6, 'offset': 3})),
        dbc.Row(dbc.Col(html.H3("See Stats by Airport"), width={'size': 6, 'offset': 3}), style={'textAlign': 'center', 'paddingBottom': '1%', 'paddingTop': '3%'}),
        dbc.Row(dbc.Col(html.P("Select an airport and see more details about severe delays there."), width={'size': 6, 'offset': 3})),
        dbc.Row(dbc.Col(html.Div([
            dcc.Dropdown(
                id='airport-dropdown',
                options=[{'label': x, 'value': x} for x in airports])]), width={"size": 6, "offset": 3}, style={'paddingBottom': '2%', 'paddingTop': '1%'})),
        dbc.Row(dbc.Col(html.Div(id='airport-specific-charts-1'), width={"size": 12, "offset": 0})),
        dbc.Row(dbc.Col(html.Div(id='airport-specific-charts-2'), width={"size": 12, "offset": 0})),
        dbc.Row(dbc.Col(html.Div(id='airport-specific-charts-3'), width={"size": 12, "offset": 0}))
])

@app.callback(dash.dependencies.Output('select-dest', 'options'),
               dash.dependencies.Input('select-origin', 'value'))
def get_destination_options(origin):
    filtered_dest_options = list(airport_lookup.loc[airport_lookup['ORIGIN'] == origin]['DEST'].unique())
    destination_options = sorted(filtered_dest_options)
    return destination_options
@app.callback([dash.dependencies.Output('prediction', 'children')],
              [dash.dependencies.Input('select-airline', 'value'),
               dash.dependencies.Input('select-origin', 'value'),
               dash.dependencies.Input('select-dest', 'value'),
               dash.dependencies.Input('my-date-picker-single', 'date'),
               dash.dependencies.Input('hour-time', 'value'),
               dash.dependencies.Input('minutes-time', 'value'),
               dash.dependencies.Input('hour-duration', 'value'),
               dash.dependencies.Input('minutes-duration', 'value'),
               dash.dependencies.Input('submit-val', 'n_clicks')])

def predict(airline, origin, destination, date_value, hour_takeoff, minutes_takeoff, hour_duration, minutes_duration, n_clicks):
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%Y-%m-%d')
    relevant_airlines = {'Southwest': 'WN',
                         'Delta': 'DL',
                         'SkyWest': 'OO',
                         'American Airlines': 'AA',
                         'United Airlines': 'UA',
                         'JetBlue': 'B6',
                         'Alaska Airlines': 'AS',
                         'Spirit Airlines': 'NK'}

    if n_clicks is not None:
        flight_details = pd.DataFrame()
        flight_details['ORIGIN'] = [str(origin)]
        flight_details['DEST'] = str(destination)
        flight_details['MKT_CARRIER'] = str(airline)
        flight_details['MKT_CARRIER'].replace(relevant_airlines, inplace=True)
        flight_details['FL_DATE'] = date_string
        flight_details['time'] = str(hour_takeoff) + ':' + str(minutes_takeoff)
        flight_details['CRS_ELAPSED_TIME'] = (hour_duration * 60) + minutes_duration
        flight_details['FL_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_DATE'] + ' ' + flight_details['time'])
        flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].astype('datetime64[ns]')
        flight_details['airport-lookup-key'] = flight_details['ORIGIN'] + '-' + flight_details['DEST']
        flight_details = pd.merge(flight_details, airport_lookup, left_on='airport-lookup-key', right_on='airport-lookup-key')
        flight_details['FL_DATE_LOCAL'] = flight_details.apply(lambda x: x['FL_DATE_LOCAL'].replace(tzinfo=timezone(x['origin-tz'])), axis=1)
        flight_details['flight_duration'] = pd.to_timedelta(flight_details['CRS_ELAPSED_TIME'], 'm')
        flight_details['FL_ARR_DATE_REL_ORIGIN'] = flight_details['FL_DATE_LOCAL'] + flight_details['flight_duration']
        # And now we convert arrival time and date to a local time
        flight_details['FL_ARR_DATE_LOCAL'] = flight_details.apply(lambda x: x['FL_ARR_DATE_REL_ORIGIN'].tz_convert(x['dest-tz']), axis=1)
        # We won't be needitn timezone info anymore, so let's remove it
        flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].apply(remove_timezone)
        flight_details['FL_ARR_DATE_LOCAL'] = flight_details['FL_ARR_DATE_LOCAL'].apply(remove_timezone)
        flight_details['FL_ARR_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_ARR_DATE_LOCAL'])
        flight_details['FL_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_DATE_LOCAL'])
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

        flight_details['takeoff-hour'] = flight_details['FL_ARR_DATE_LOCAL'].dt.hour.astype(int)
        flight_details['arriving-hour'] = flight_details['FL_ARR_DATE_LOCAL'].dt.hour.astype(int)

        conditions = [
            (flight_details['takeoff-hour'] >= 5) & (flight_details['takeoff-hour'] <= 8),
            (flight_details['takeoff-hour'] >= 8) & (flight_details['takeoff-hour'] <= 12),
            (flight_details['takeoff-hour'] > 12) & (flight_details['takeoff-hour'] <= 15),
            (flight_details['takeoff-hour'] > 15) & (flight_details['takeoff-hour'] <= 17),
            (flight_details['takeoff-hour'] > 17) & (flight_details['takeoff-hour'] <= 19),
            (flight_details['takeoff-hour'] > 19) & (flight_details['takeoff-hour'] <= 21),
            (flight_details['takeoff-hour'] >= 0) & (flight_details['takeoff-hour'] < 5),
            (flight_details['takeoff-hour'] > 21)
        ]
        values = ['Early Morning', 'Late Morning', 'Early Afternoon', 'Late Afternoon', 'Early Evening', 'Late Evening',
                  'Night', 'Night']
        flight_details['takeoff-time-of-day'] = np.select(conditions, values)

        arr_conditions = [
            (flight_details['arriving-hour'] >= 5) & (flight_details['arriving-hour'] <= 8),
            (flight_details['arriving-hour'] >= 8) & (flight_details['arriving-hour'] <= 12),
            (flight_details['arriving-hour'] > 12) & (flight_details['arriving-hour'] <= 15),
            (flight_details['arriving-hour'] > 15) & (flight_details['arriving-hour'] <= 17),
            (flight_details['arriving-hour'] > 17) & (flight_details['arriving-hour'] <= 19),
            (flight_details['arriving-hour'] > 19) & (flight_details['arriving-hour'] <= 21),
            (flight_details['arriving-hour'] >= 0) & (flight_details['arriving-hour'] < 5),
            (flight_details['arriving-hour'] > 21)
        ]
        arr_values = ['Early Morning', 'Late Morning', 'Early Afternoon', 'Late Afternoon', 'Early Evening',
                      'Late Evening', 'Night', 'Night']
        flight_details['arrival-time-of-day'] = np.select(arr_conditions, arr_values)

        # Takeoff Congestion Key
        flight_details['takeoff-congestion-key'] = flight_details['ORIGIN_x'] \
                                                 + flight_details['DAY_OF_WEEK'].astype(str) \
                                                 + flight_details['takeoff-time-of-day']
        # Arrival Congestion Key
        flight_details['arrival-congestion-key'] = flight_details['DEST_x'] \
                                                 + flight_details['ARR_DAY_OF_WEEK'].astype(str) \
                                                 + flight_details['arrival-time-of-day']

        # Now we add congestion data to our main dataframe
        congestion = pd.read_csv('data/prepared/airport_congestion.csv')
        flight_details = pd.merge(flight_details, congestion, left_on='takeoff-congestion-key', right_on='congestion-key')
        # updating key
        congestion = congestion.add_prefix('dest-')
        # Now data on the congestion conditions of the airport where the flight is arriving
        flight_details = pd.merge(flight_details, congestion, left_on='arrival-congestion-key', right_on='dest-congestion-key')
        flight_details = pd.merge_asof(flight_details, holidays, left_on='FL_DATE_LOCAL', right_on='holiday_date', direction='nearest', tolerance=pd.Timedelta(days=7))
        flight_details['days-from-holiday'] = (flight_details['FL_DATE_LOCAL'] - flight_details['holiday_date']).dt.days
        flight_details['holiday'] = flight_details.loc[flight_details['days-from-holiday'] == 0, 'holiday_name']
        flight_details['holiday'].fillna('Not a Holiday', inplace=True)
        flight_details['days-from-holiday'] = flight_details['days-from-holiday'].astype(str)
        flight_details['days-from-specific-holiday'] = flight_details['holiday_name'] + '_' + flight_details['days-from-holiday'].astype(str)
        flight_details['days-from-specific-holiday'].fillna('no-close-holiday', inplace=True)
        flight_details['days-to-forecast'] = (flight_details['FL_DATE_LOCAL'] - pd.Timestamp('now')).dt.days
        forecast_data_url = 'http://api.weatherapi.com/v1/forecast.json'
        load_dotenv()
        api_key = os.getenv("API_KEY")
        start_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
        end_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
        location = flight_details['origin-lat-long'][0]
        days = flight_details['days-to-forecast'][0].astype(str)
        r_origin = requests.get(forecast_data_url + '?key=' + api_key + '&q=' + location + '&' + days + '&dt=' + start_date + '&end_dt=' + end_date)
        d_origin = json.loads(r_origin.text)
        dest_location = flight_details['dest-lat-long'][0]
        r_dest = requests.get(forecast_data_url + '?key=' + api_key + '&q=' + dest_location + '&' + days + '&dt=' + start_date + '&end_dt=' + end_date)
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
        flight_details.rename(columns={"ORIGIN_x": "ORIGIN", "DEST_x": "DEST"}, inplace=True)
        flight_details['CRS_DEP_TIME'] = flight_details['FL_DATE_LOCAL'].dt.strftime('%H:%M')
        flight_details['CRS_ARR_TIME'] = flight_details['FL_ARR_DATE_LOCAL'].dt.strftime('%H:%M')
        flight_details['takeoff-mins-from-midnight'] = ((pd.to_datetime(flight_details['CRS_DEP_TIME'])
                                                       - pd.to_datetime(flight_details['CRS_DEP_TIME']).dt.normalize()) \
                                                      / pd.Timedelta('1 minute')).astype(int)
        flight_details['CRS_ARR_TIME'] = flight_details['CRS_ARR_TIME'].replace({'24:00': '00:00'})
        flight_details['landing-mins-from-midnight'] = ((pd.to_datetime(flight_details['CRS_ARR_TIME'])
                                                       - pd.to_datetime(flight_details['CRS_ARR_TIME']).dt.normalize()) \
                                                      / pd.Timedelta('1 minute')).astype(int)

        model_cols = ['CRS_ELAPSED_TIME', 'DISTANCE', 'origin-elevation',
                      'dest-elevation', 'avg-takeoff-congestion', 'avg-arrival-congestion',
                      'dest-avg-takeoff-congestion', 'dest-avg-arrival-congestion', 'takeoff-mins-from-midnight',
                      'landing-mins-from-midnight', 'maxtemp', 'mintemp', 'avgtemp', 'totalprecip', 'avgvis',
                      'maxwind', 'avghumidity', 'dest-maxtemp', 'dest-mintemp', 'dest-avgtemp', 'dest-totalprecip',
                      'dest-avgvis', 'dest-maxwind', 'dest-avghumidity', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
                      'MKT_CARRIER', 'ORIGIN', 'DEST', 'takeoff-time-of-day', 'arrival-time-of-day', 'ARR_DAY_OF_WEEK',
                      'holiday', 'days-from-specific-holiday']
        X = flight_details[model_cols].copy()
        filename = 'model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        prediction = loaded_model.predict(X)
        if prediction[0] == 'No':
            return ['Our model doesn\'t expect a major delays for the flight you selected. Refresh to try another flight.']
        else:
            return ['Our model predicts your flight will experience a major delay. Refresh to try another flight.']
    else:
        raise PreventUpdate
@app.callback(
    dash.dependencies.Output('model-inputs', 'style'),
    Input(component_id='submit-val', component_property='n_clicks'))
def update_output(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        hide_element = {'display': 'none'}
        return hide_element

@app.callback(
dash.dependencies.Output('airport-specific-charts-1', 'children'),
dash.dependencies.Output('airport-specific-charts-2', 'children'),
dash.dependencies.Output('airport-specific-charts-3', 'children'),
[dash.dependencies.Input('airport-dropdown', 'value')])
def update_output(fig_name):
    return name_to_figure(fig_name)
def name_to_figure(fig_name):
    order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    order_times = ['Night', 'Late Evening', 'Early Evening', 'Late Afternoon', 'Early Afternoon', 'Late Morning',
                   'Early Morning']
    days_and_times = px.density_heatmap(df_weekdays_times.loc[df_weekdays_times['ORIGIN'] == '{}'.format(fig_name)],
                             x="DAY_OF_WEEK", y="takeoff-time-of-day", z='percent-delayed', histfunc="avg",
                             color_continuous_scale='OrRd',
                             category_orders={"DAY_OF_WEEK": order_days, "takeoff-time-of-day": order_times},
                             title='Percent of Flights with Severe Delays Throughout the Week')

    days_and_times.update_layout(coloraxis_colorbar=dict(
        title="% Delayed",
    ))

    delays_by_holiday = px.bar(df_by_holiday,
                             x=df_by_holiday.loc[df_by_holiday['ORIGIN'] == '{}'.format(fig_name)]['holiday'],
                             y=df_by_holiday.loc[df_by_holiday['ORIGIN'] == '{}'.format(fig_name)]['percent-delayed'],
                             color=df_by_holiday.loc[df_by_holiday['ORIGIN'] == '{}'.format(fig_name)]['percent-delayed'],
                             color_continuous_scale='OrRd',
                             labels={"x": "Holiday",
                                     "y": "Severe Delays"},
                             title="Percent of Flights Severely Delayed by Holiday at {} Airport".format(fig_name))
    overall_delays = px.line(df_by_airport,
                  x=df_by_airport.loc[df_by_airport['ORIGIN'] == '{}'.format(fig_name)]['FL_DATE'],
                  y=df_by_airport.loc[df_by_airport['ORIGIN'] == '{}'.format(fig_name)]['Severe Delays'],
                  labels={"x": "Date",
                          "y": "Severe Delays"},
                  title="Daily Severe Airport Delays at {} Airport".format(fig_name))
    overall_delays.update_layout(xaxis_rangeslider_visible=True)

    return dcc.Graph(figure=days_and_times), dcc.Graph(figure=delays_by_holiday), dcc.Graph(figure=overall_delays)

if __name__ == '__main__':
    app.run(debug=True)