# Bringing in libraries
from dash.dependencies import Output, Input, State
import dash_bootstrap_components as dbc
import dash_core_components as dcc
from dash import html
import plotly.express as px
from flask import Flask
import flask
import pandas as pd
import dash
from datetime import timedelta
import pickle
import json
import requests
from pytz import timezone
from datetime import date
from dotenv import load_dotenv
import os
from dash.exceptions import PreventUpdate
import numpy as np
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Initializing Flask server and Dash App
server = Flask(__name__)
app = dash.Dash(server=server, url_base_pathname='/flight-delays/', external_stylesheets=[dbc.themes.FLATLY], prevent_initial_callbacks=True)
app.title = 'air-travel-delays'


load_dotenv()
pg_user = os.getenv("PG_USR")
pg_pass = os.getenv("PG_PASS")
engine = create_engine('postgresql://{}:{}@database/carrier_data'.format(pg_user, pg_pass))

airports_query = '''
SELECT MIN("ORIGIN") as "ORIGIN", MIN("DEST") as "DEST", MIN("origin-elevation") as "origin-elevation", 
MIN("dest-elevation") as "dest-elevation", MIN("DISTANCE") as "DISTANCE", MIN("dest-lat-long") as "dest-lat-long",
MIN("origin-lat-long") as "origin-lat-long", MIN("origin-tz") as "origin-tz", MIN("dest-tz") as "dest-tz", MIN("route") as "route"
from airports
GROUP BY
    route
'''
airports_df = pd.read_sql(airports_query, engine)
origins_list = sorted(list(airports_df['ORIGIN'].unique())) # Sorted list of origin options

# Loading Data for Visualizations
holidays = pd.read_csv('data/prepared/holidays.csv') # US Holidays Dataframe
holidays['holiday_date'] = pd.to_datetime(holidays['holiday_date']) # Transforming date field in Holidays Dataframe to datetime
relevant_airlines = sorted(['Southwest', 'Delta', 'SkyWest', 'American Airlines', 'United Airlines', 'JetBlue', 'Alaska Airlines', 'Spirit Airlines']) # Supported Airlines List

#Supporting functions
def remove_timezone(dt):
    '''
    Here `dt` is a python datetime
    object that used .replace() method
    Necessary to remove timezone info when transforming data for modeling
    '''
    return dt.replace(tzinfo=None)

# Disclaimer text to display in layout
disclaimer_text = 'This web app will predict whether a future US domestic flight will have a severe delay (defined as 1 hour or more).' \
                  ' Currently, the app only supports flights from 62 major US airports and the top 8 major airlines.'

# Main app content
app.layout = dbc.Container([
        dbc.Row(dbc.Col([
                         html.H2("Will your flight be severely delayed?"),
                         html.P(disclaimer_text)
                        ]), style={'textAlign': 'center', 'paddingTop': '2%'}),
        dbc.Row(dbc.Col([html.Div([
        dbc.Row(dbc.Col([
                         html.P("1: Select your Airline", style={'paddingTop': '10px', 'font-weight': 'bold'}),
                         html.Div(dcc.Dropdown(id='select-airline', options=[{'label': x, 'value': x} for x in relevant_airlines]))
                        ])),
        dbc.Row(
            [
        dbc.Col([
                 html.P("2: Flying FROM:", style={'paddingTop': '10px', 'font-weight': 'bold'}),
                 html.Div(dcc.Dropdown(id='select-origin', options=[{'label': x, 'value': x} for x in origins_list]))
                         ]),
        dbc.Col([
                 html.P("3: Flying TO:", style={'paddingTop': '10px', 'font-weight': 'bold'}),
                 html.Div(dcc.Dropdown(id='select-dest'))
                        ])]),
        dbc.Row(
            [
        dbc.Col([
                 html.P("4: Flight Date:", style={'paddingTop': '10px', 'font-weight': 'bold'}),
                 html.Div([dcc.DatePickerSingle(id='my-date-picker-single',
                                               min_date_allowed=date.today() + timedelta(days=1),
                                               max_date_allowed=date.today() + timedelta(days=14),
                                               initial_visible_month=date.today(),
                                               date=date.today() + timedelta(days=1))])]),
        dbc.Col([
                 html.P("5: Flight Time (24 Hour Format):", style={'paddingTop': '10px', 'font-weight': 'bold'}),
                 html.Div([dcc.Input(id="hour-time", min=0, max=24, step=1, type="number", debounce=True, placeholder="HH", style={'width': '15%'}),
                           dcc.Input(id="minutes-time", min=0, max=60, step=1, type="number", debounce=True, placeholder="MM", style={'width': '15%'})])
                ])]),
       dbc.Row(dbc.Col(html.Button('Predict', id='submit-val', style={'background-color': '#4681f4', 'color': 'white', 'border':'2px solid #4681f4', 'width': '100%'}), style={'paddingTop': '10px'})),
        ], id='model-inputs'),
       dbc.Row(html.H4(id='prediction'))])),
       dbc.Row(
           dbc.Col(html.H3("Delay Severity Charts by Airport"), style={'textAlign': 'center', 'paddingTop': '2%'})),
       dbc.Row([
           dbc.Col([html.P("Select an airport", style={'textAlign': 'center'}),
                    html.Div(dcc.Dropdown(id='airport-dropdown', options=[{'label': x, 'value': x} for x in origins_list]))
                   ]),
           dbc.Col([html.P("Select a graph", style={'textAlign': 'center'}),
                    html.Div([dcc.Dropdown(id='airport-dropdown-2', options=['Holidays', 'Throughout the Week'])])])
               ]),
       dbc.Row(dbc.Col([
                        html.Div(id='airport-specific-charts-1'),]), style={'textAlign': 'center', 'paddingTop': '1%'})
                            ])
# Callbacks
@app.callback(dash.dependencies.Output('select-dest', 'options'),
               dash.dependencies.Input('select-origin', 'value'))
def get_destination_options(origin):
    filtered_dest_options = list(airports_df.loc[airports_df['ORIGIN'] == origin]['DEST'].unique())
    destination_options = sorted(filtered_dest_options)
    return destination_options
@app.callback([dash.dependencies.Output('prediction', 'children')],
              [dash.dependencies.Input('select-airline', 'value'),
               dash.dependencies.Input('select-origin', 'value'),
               dash.dependencies.Input('select-dest', 'value'),
               dash.dependencies.Input('my-date-picker-single', 'date'),
               dash.dependencies.Input('hour-time', 'value'),
               dash.dependencies.Input('minutes-time', 'value'),
               dash.dependencies.Input('submit-val', 'n_clicks')])

def predict(airline, origin, destination, date_value, hour_takeoff, minutes_takeoff, n_clicks):
    '''
    Function accepts user inputs from dash app
    Uses inputs to create a dataframe which it then passes to pickled model
    Returns a prediction about whether a flight will be delayed or not
    '''

    # Convert entered date into string
    if date_value is not None:
        date_object = date.fromisoformat(date_value)
        date_string = date_object.strftime('%Y-%m-%d')

    # Dictionary of Airlines and their Codes
    relevant_airlines = {'Southwest': 'WN',
                         'Delta': 'DL',
                         'SkyWest': 'OO',
                         'American Airlines': 'AA',
                         'United Airlines': 'UA',
                         'JetBlue': 'B6',
                         'Alaska Airlines': 'AS',
                         'Spirit Airlines': 'NK'}

    # Function should only run once "Predict" button is clicked to avoid errors
    if n_clicks is not None:
        # Initalize an empty dataframe
        flight_details = pd.DataFrame()
        # Set first values of dataframe based on above user input: Origin Airport, Destination Airport, Airline, Flight Date, Flight Time and Flight Duration
        flight_details['ORIGIN'] = [str(origin)]
        flight_details['DEST'] = str(destination)
        # Create airport lookup key
        flight_details['route'] = flight_details['ORIGIN'] + '-' + flight_details['DEST']
        routes_query = '''
        SELECT *
        FROM route_times
        '''
        route_times_df = pd.read_sql(routes_query, engine)
        flight_details = pd.merge(flight_details, route_times_df, left_on='route',
                                  right_on='route')
        flight_details['MKT_CARRIER'] = str(airline)
        flight_details['MKT_CARRIER'].replace(relevant_airlines, inplace=True)
        flight_details['FL_DATE'] = date_string
        flight_details['time'] = str(hour_takeoff) + ':' + str(minutes_takeoff)
        # Flight duration must be expressed in minutes so we transform it here

        # Add date and time into single FL_DATE field
        flight_details['FL_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_DATE'] + ' ' + flight_details['time'])
        flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].astype('datetime64[ns]')
        # Merge in additional airport details from airport datafame using the above created key
        flight_details = pd.merge(flight_details, airports_df, left_on='route',
                                  right_on='route')
        # Add timezone data to datetime column so that we can calculate arrival date
        flight_details['FL_DATE_LOCAL'] = flight_details.apply(
            lambda x: x['FL_DATE_LOCAL'].replace(tzinfo=timezone(x['origin-tz'])), axis=1)
        flight_details['flight_duration'] = pd.to_timedelta(flight_details['CRS_ELAPSED_TIME'], 'm')
        flight_details['FL_ARR_DATE_REL_ORIGIN'] = flight_details['FL_DATE_LOCAL'] + flight_details['flight_duration']
        # And now we convert arrival time and date to a local time
        flight_details['FL_ARR_DATE_LOCAL'] = flight_details.apply(
            lambda x: x['FL_ARR_DATE_REL_ORIGIN'].tz_convert(x['dest-tz']), axis=1)
        # We won't be need timezone info anymore, so let's remove it
        flight_details['FL_DATE_LOCAL'] = flight_details['FL_DATE_LOCAL'].apply(remove_timezone)
        flight_details['FL_ARR_DATE_LOCAL'] = flight_details['FL_ARR_DATE_LOCAL'].apply(remove_timezone)
        flight_details['FL_ARR_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_ARR_DATE_LOCAL'])
        flight_details['FL_DATE_LOCAL'] = pd.to_datetime(flight_details['FL_DATE_LOCAL'])
        # Grab day of the week and add 1 so that it matches key from model
        flight_details['DAY_OF_WEEK'] = flight_details['FL_DATE_LOCAL'].dt.dayofweek
        flight_details['DAY_OF_WEEK'] = flight_details['DAY_OF_WEEK'] + 1
        flight_details['ARR_DAY_OF_WEEK'] = flight_details['FL_ARR_DATE_LOCAL'].dt.dayofweek
        flight_details['ARR_DAY_OF_WEEK'] = flight_details['ARR_DAY_OF_WEEK'] + 1
        # Make the day of the week keys more reader friendly by translating integers to actual days
        day_of_week_translation = {1: 'Monday',
                                   2: 'Tuesday',
                                   3: 'Wednesday',
                                   4: 'Thursday',
                                   5: 'Friday',
                                   6: 'Saturday',
                                   7: 'Sunday'}
        flight_details['DAY_OF_WEEK'].replace(day_of_week_translation, inplace=True)
        flight_details['ARR_DAY_OF_WEEK'].replace(day_of_week_translation, inplace=True)

        # Get flight hour of the day from datetime fields
        flight_details['takeoff-hour'] = flight_details['FL_ARR_DATE_LOCAL'].dt.hour.astype(int)
        flight_details['arriving-hour'] = flight_details['FL_ARR_DATE_LOCAL'].dt.hour.astype(int)
        # Use the above fetched hour field to calculate the time of day a flight takes off and arrives
        # Definitions of 'time of day' taken from Britannica: https://www.britannica.com/dictionary/eb/qa/parts-of-the-day-early-morning-late-morning-etc
        conditions = [(flight_details['takeoff-hour'] >= 5) & (flight_details['takeoff-hour'] <= 8),
                      (flight_details['takeoff-hour'] >= 8) & (flight_details['takeoff-hour'] <= 12),
                      (flight_details['takeoff-hour'] > 12) & (flight_details['takeoff-hour'] <= 15),
                      (flight_details['takeoff-hour'] > 15) & (flight_details['takeoff-hour'] <= 17),
                      (flight_details['takeoff-hour'] > 17) & (flight_details['takeoff-hour'] <= 19),
                      (flight_details['takeoff-hour'] > 19) & (flight_details['takeoff-hour'] <= 21),
                      (flight_details['takeoff-hour'] >= 0) & (flight_details['takeoff-hour'] < 5),
                      (flight_details['takeoff-hour'] > 21)]
        values = ['Early Morning', 'Late Morning', 'Early Afternoon', 'Late Afternoon', 'Early Evening', 'Late Evening',
                  'Night', 'Night']
        flight_details['takeoff-time-of-day'] = np.select(conditions, values)

        arr_conditions = [(flight_details['arriving-hour'] >= 5) & (flight_details['arriving-hour'] <= 8),
                          (flight_details['arriving-hour'] >= 8) & (flight_details['arriving-hour'] <= 12),
                          (flight_details['arriving-hour'] > 12) & (flight_details['arriving-hour'] <= 15),
                          (flight_details['arriving-hour'] > 15) & (flight_details['arriving-hour'] <= 17),
                          (flight_details['arriving-hour'] > 17) & (flight_details['arriving-hour'] <= 19),
                          (flight_details['arriving-hour'] > 19) & (flight_details['arriving-hour'] <= 21),
                          (flight_details['arriving-hour'] >= 0) & (flight_details['arriving-hour'] < 5),
                          (flight_details['arriving-hour'] > 21)]
        arr_values = ['Early Morning', 'Late Morning', 'Early Afternoon', 'Late Afternoon', 'Early Evening',
                      'Late Evening', 'Night', 'Night']
        flight_details['arrival-time-of-day'] = np.select(arr_conditions, arr_values)

        # Create Takeoff Congestion Key based on the origin, day of the week and takeoff time
        flight_details['takeoff-congestion-key'] = flight_details['ORIGIN_x'] \
                                                   + flight_details['DAY_OF_WEEK'].astype(str) \
                                                   + flight_details['takeoff-time-of-day']
        # Create Arrival Congestion Key based on origin, day of the week and takeoff time
        flight_details['arrival-congestion-key'] = flight_details['DEST_x'] \
                                                   + flight_details['ARR_DAY_OF_WEEK'].astype(str) \
                                                   + flight_details['arrival-time-of-day']

        # Now we add congestion data to our main dataframe and merge it using the above generated key
        congestion = pd.read_csv('data/prepared/airport_congestion.csv')
        flight_details = pd.merge(flight_details, congestion, left_on='takeoff-congestion-key',
                                  right_on='congestion-key')
        # Update our key for the destination
        congestion = congestion.add_prefix('dest-')
        # Now data on the congestion conditions of the airport where the flight is arriving are merged in like above
        flight_details = pd.merge(flight_details, congestion, left_on='arrival-congestion-key',
                                  right_on='dest-congestion-key')

        # Next we calculate the proximity of the flight to a holiday
        # Start by merging the holidays dataframe into our flight_details dataframe with a tolerance of 7 days before or after
        flight_details = pd.merge_asof(flight_details, holidays, left_on='FL_DATE_LOCAL', right_on='holiday_date',
                                       direction='nearest', tolerance=pd.Timedelta(days=7))
        # Calculate the distance between the nearest holiday and the flight date in days
        flight_details['days-from-holiday'] = (flight_details['FL_DATE_LOCAL'] - flight_details['holiday_date']).dt.days
        # If the flight is exactly on the date of the holiday (0 days from it) we pull in the name of the holiday itself
        flight_details['holiday'] = flight_details.loc[flight_details['days-from-holiday'] == 0, 'holiday_name']
        # If it wasn't exactly on a holiday, we fill in the value as "Not a Holiday"
        flight_details['holiday'].fillna('Not a Holiday', inplace=True)

        # Next we create a field that tells us exactly how many days the flight is from a specific holiday if any
        flight_details['days-from-holiday'] = flight_details['days-from-holiday'].astype(str)
        flight_details['days-from-specific-holiday'] = flight_details['holiday_name'] + '_' + flight_details[
            'days-from-holiday'].astype(str)
        # If not close to a holiday, simply replace with 'no-close-holiday'
        flight_details['days-from-specific-holiday'].fillna('no-close-holiday', inplace=True)

        # Next we pull in the forecasted weather data using weatherAPI
        # First calculate how many days into the future our forecast is
        flight_details['days-to-forecast'] = (flight_details['FL_DATE_LOCAL'] - pd.Timestamp('now')).dt.days
        # Declare forecast URL for fetching
        forecast_data_url = 'http://api.weatherapi.com/v1/forecast.json'
        # Load enviromental variables where we store our own API key. If replicating this code, you need to replace this with your own API key/path to the key
        load_dotenv()
        api_key = os.getenv("API_KEY")
        # API requires us to declare the start and end dates for when we're pulling weather forecasts for so we declare those
        start_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
        end_date = flight_details['FL_DATE_LOCAL'].dt.date.astype(str)[0]
        # Set the location (in latitude and longitude) for where the API is pulling weather forecasts for
        location = flight_details['origin-lat-long'][0]
        # Declare the request plus store it in a json afterwards for both the origin and destination
        r_origin = requests.get(
            forecast_data_url + '?key=' + api_key + '&q=' + location + '&dt=' + start_date + '&end_dt=' + end_date)
        d_origin = json.loads(r_origin.text)
        dest_location = flight_details['dest-lat-long'][0]
        r_dest = requests.get(
            forecast_data_url + '?key=' + api_key + '&q=' + dest_location + '&dt=' + start_date + '&end_dt=' + end_date)
        d_dest = json.loads(r_dest.text)
        # Next we parse the JSON document created above to add weather forecast details to the new columns
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
        # Declare the month and day of the month fields
        flight_details['MONTH'] = flight_details['FL_DATE_LOCAL'].dt.month
        flight_details['DAY_OF_MONTH'] = flight_details['FL_DATE_LOCAL'].dt.day
        # Some of the above merges renamed our columns so we translate those back to what they should be
        flight_details.rename(columns={"ORIGIN_x": "ORIGIN", "DEST_x": "DEST"}, inplace=True)
        # Translate departure time and arrival time to the standard datetime format
        # Calculate the time difference (in minutes) between midnight and the arrival/departure times
        flight_details['CRS_DEP_TIME'] = flight_details['FL_DATE_LOCAL'].dt.strftime('%H:%M')
        flight_details['CRS_ARR_TIME'] = flight_details['FL_ARR_DATE_LOCAL'].dt.strftime('%H:%M')
        flight_details['takeoff-mins-from-midnight'] = ((pd.to_datetime(flight_details['CRS_DEP_TIME'])
                                                         - pd.to_datetime(
                    flight_details['CRS_DEP_TIME']).dt.normalize()) / pd.Timedelta('1 minute')).astype(int)
        flight_details['CRS_ARR_TIME'] = flight_details['CRS_ARR_TIME'].replace({'24:00': '00:00'})
        flight_details['landing-mins-from-midnight'] = ((pd.to_datetime(flight_details['CRS_ARR_TIME'])
                                                         - pd.to_datetime(
                    flight_details['CRS_ARR_TIME']).dt.normalize()) / pd.Timedelta('1 minute')).astype(int)
        # Below we declare all the columns our model needs to run
        model_cols = ['CRS_ELAPSED_TIME', 'DISTANCE', 'origin-elevation',
                      'dest-elevation', 'avg-takeoff-congestion', 'avg-arrival-congestion',
                      'dest-avg-takeoff-congestion', 'dest-avg-arrival-congestion', 'takeoff-mins-from-midnight',
                      'landing-mins-from-midnight', 'maxtemp', 'mintemp', 'avgtemp', 'totalprecip', 'avgvis',
                      'maxwind', 'avghumidity', 'dest-maxtemp', 'dest-mintemp', 'dest-avgtemp', 'dest-totalprecip',
                      'dest-avgvis', 'dest-maxwind', 'dest-avghumidity', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK',
                      'MKT_CARRIER', 'ORIGIN', 'DEST', 'takeoff-time-of-day', 'arrival-time-of-day', 'ARR_DAY_OF_WEEK',
                      'holiday', 'days-from-specific-holiday']
        # We create a copy of the above dataframe with just the relevant columns for the model
        # This gets rid of all the random columns we created to fetch the data (like the keys, etc.)
        X = flight_details[model_cols].copy()
        X['MONTH'] = X['MONTH'].astype(str)
        X['DAY_OF_MONTH'] = X['DAY_OF_MONTH'].astype(str)
        # Load our model
        filename = 'model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        # Generate a prediction
        prediction = loaded_model.predict(X)
        # Depending on the output of the prediction we return one of 2 messages for the reader
        # Either there is no severe delay detect, the model returns 'No'
        # Or if the model does think there will be a severe delay it returns 'Yes'
        if prediction[0] == 'No':
            return ['Our model doesn\'t expect a major delays for the flight you selected. Refresh to try another flight.']
        else:
            return ['Our model predicts your flight will experience a major delay. Refresh to try another flight.']
    else:
        # If the "Predict" button is not pressed, we don't want anything to run so we prevent an udpate
        raise PreventUpdate

# This callback hides the inputs after the "Predict" button is pressed
@app.callback(
    dash.dependencies.Output('model-inputs', 'style'),
    Input(component_id='submit-val', component_property='n_clicks'))
def update_output(n_clicks):
    '''
    Function takes the number of clicks (Default None). Once pressed, the button returns a hide display style element
    back to the input HTML elements in our app
    '''
    if n_clicks is None:
        raise PreventUpdate
    else:
        hide_element = {'display': 'none'}
        return hide_element

# Callback for retreiving airport-specific charts based on user selection
@app.callback(
dash.dependencies.Output('airport-specific-charts-1', 'children'),
[dash.dependencies.Input('airport-dropdown', 'value'),
 dash.dependencies.Input('airport-dropdown-2', 'value')])
def name_to_figure(fig_name, graph_type):
    '''
    function takes airport name string
    uses it to filter dataframes for airport specific data
    returns 3 plotly express graphs for selected airport
    '''
    origin = "'{}'".format(fig_name)
    if fig_name is not None and graph_type is not None:
        load_dotenv()
        pg_user = os.getenv("PG_USR")
        pg_pass = os.getenv("PG_PASS")
        engine = create_engine('postgresql://{}:{}@database/carrier_data'.format(pg_user, pg_pass))
        if graph_type == "Holidays":
            holiday_query = '''
            WITH total AS (
                SELECT holiday, SUM("Yes" + "No") AS total_flights
                FROM airports
                WHERE "ORIGIN" = {}
                GROUP BY holiday
            )
            SELECT total.holiday, total.total_flights, (
                    SELECT SUM("Yes")
                    FROM airports
                    WHERE airports.holiday = total.holiday
                        AND "ORIGIN" = {}
                ) / total.total_flights as percentage_delayed
            FROM airports, total 
            WHERE airports.holiday = total.holiday
            GROUP BY total.holiday, total.total_flights
            '''.format(origin, origin, origin)

            holiday_query_output_df = pd.read_sql(holiday_query, engine)

            delays_by_holiday = px.bar(holiday_query_output_df,
                                       x=holiday_query_output_df['holiday'],
                                       y=holiday_query_output_df['percentage_delayed'],
                                       color=holiday_query_output_df['percentage_delayed'],
                                       color_continuous_scale='OrRd',
                                       labels={"x": "Holiday",
                                               "y": "Severe Delays"},
                                       title="Percent of Flights Severely Delayed by Holiday at {} Airport".format(
                                           fig_name))
            return dcc.Graph(figure=delays_by_holiday)

        elif graph_type == "Throughout the Week":
            week_query = '''
            WITH total AS (
                SELECT "DAY_OF_WEEK", "takeoff-time-of-day", SUM("Yes" + "No") AS total_flights
                FROM airports
                WHERE "ORIGIN" = {}
                GROUP BY "DAY_OF_WEEK", "takeoff-time-of-day"
                )
                SELECT total."DAY_OF_WEEK", total."takeoff-time-of-day", total.total_flights, (
                    SELECT SUM("Yes")
                    FROM airports
                    WHERE airports."takeoff-time-of-day" = total."takeoff-time-of-day"
                        AND airports."DAY_OF_WEEK" = total."DAY_OF_WEEK"
                        AND "ORIGIN" = {}
                    ) / total.total_flights as percentage_delayed
                FROM airports, total 
                WHERE airports."takeoff-time-of-day" = total."takeoff-time-of-day"
                    AND airports."DAY_OF_WEEK" = total."DAY_OF_WEEK"
                    AND "ORIGIN" = {}
                GROUP BY total."DAY_OF_WEEK", total."takeoff-time-of-day", total.total_flights
            '''.format(origin, origin, origin)

            week_query_query_output_df = pd.read_sql(week_query, engine)

            order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            order_times = ['Night', 'Late Evening', 'Early Evening', 'Late Afternoon', 'Early Afternoon', 'Late Morning', 'Early Morning']

            days_and_times = px.density_heatmap(week_query_query_output_df,
                             x="DAY_OF_WEEK", y="takeoff-time-of-day", z='percentage_delayed', histfunc="avg",
                             color_continuous_scale='OrRd',
                             category_orders={"DAY_OF_WEEK": order_days, "takeoff-time-of-day": order_times},
                             title='Percent of Flights with Severe Delays Throughout the Week')

            days_and_times.update_layout(coloraxis_colorbar=dict(
            title="% Delayed"))
            return dcc.Graph(figure=days_and_times)
@server.route('/flight-delays', methods=['GET'])
def index():
    return flask.redirect('/flight-delays')

if __name__ == '__main__':
    server.run(debug=True)