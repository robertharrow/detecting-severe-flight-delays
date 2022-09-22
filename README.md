LEAD IMAGE

# Predicting Major Flight Delays

Our client, FlightChicken, is developing a service to help users predict whether their flight may experience a severe delay. They would like us to develop a model for this. Giving users a heads up about a potential travel disruption can be tremendously helpful. Severe delays can cause travelers to miss connecting flights, miss important events and more.

There are millions of flights each year across thousands of airports and airlines (major and minor). For this MVP FlightChicken is tasking us to develop a minimum viable product (MVP) that supports 8 of the biggest airlines and 62 major domestic airports.

## Business Understanding

Our task is to solve the following business problems:
* Define what constitutes a major delay
* Collect and clean data on US flight delays
* Use supervised learning to develop a model for detecting major delays

Our client asked to build an MVP that support major US airlines and airports. These will be defined as follows:

* [Top 8 US Airlines by market share](https://www.statista.com/statistics/250577/domestic-market-share-of-leading-us-airlines/)
 * American Airlines
 * Delta Air Lines
 * United Airlines
 * Southwest Airlines
 * Alaska Airlines
 * JetBlue Airways
 * Spirit
 * SkyWest
* [Large airport hubs](https://www.faa.gov/airports/planning_capacity/passenger_allcargo_stats/passenger/media/cy20-commercial-service-enplanements.pdf)
 * "The term hub is used by the FAA to identify very busy commercial service airports. Large hubs are the airports that each account for at least one percent of total U.S. passenger enplanements."
 * In 2020 these accounted for 84% of all enplanements
 
 ### Our Success Metric

We are looking for a model that achieves the best **F1 score**, without letting accuracy fall below 70%.

**Why F1?** F1 scores are a balance between Precision and Recall. It's important for a startup like Flight Chicken to build consumer trust. That's why it's important for it to detect not miss predicting a delay. On its own, that means we would want to use *Recall*. However, if we optimize a model for Recall, it may have a lower Precision. In other words, it may too agressively guess that something is a delay. But in doing so, it would unnecessairly worry Flight Chicken's users about a flight that will be on time. F1 is the harmonic average between Precision and Recall which is why it's our primary success metric for this project.

#### F1 Formula
$$ F1 = {2 * Precision x Recall \over Precision + Recall} $$

#### Precision Formula
$$ Precision = {True Positives \over Predicted Positives} $$

#### Recall Formula
$$ Recall = {True Positivies \over Actual Total Positives} $$

### What is a 'major delay'?

* For the purposes of this project a major delay is an **arrival delay of 1 hour or more**

**Justification:**
No one likes any delay, but there is a world of difference between a 5 minute and a 3 hour delay. So the first challenge is to understand what our target is.

The biggest and most common consequence of a delay is that it may cause you to miss a connecting flight. Therefore, we will use average layover times to define a "major" delay. In other words, if a delay is severe enough to cause a traveler to potentially miss their connecting flight, we will call that a "major delay".

Multiple sources ([1](https://travel-made-simple.com/layover-long-enough/), [2](https://www.alternativeairlines.com/dealing-with-a-short-layover), [3](https://www.mic.com/articles/192954/how-much-time-do-you-really-need-for-a-layover), [4](https://traveltips.usatoday.com/minimum-time-should-allow-layovers-109029.html)) advise travelers to allow at least 1 hour for a connecting flight. Therefore, we will use that as the cutoff. For this project, we define a "major delay" as an arrival delay of 1+ hours.

### What causes delays?

Before we dive into data, it would be good to know common delay causes.

The Bureau of Tranpostation Statistics (BTS) reports the following:

1. **Air Carrier Delay 41%** "Maintenance or crew problems, aircraft cleaning, baggage loading, fueling, etc."
2. **Aircraft Arriving Late 30%**  "A previous flight with same aircraft arrived late, causing the present flight to depart late."
3. **National Aviation System Delay 22%** Things such as "non-extreme weather conditions, airport operations, heavy traffic volume, and air traffic control."
4. **Security Delay 0.2%** "Eacuation of a terminal or concourse, re-boarding of aircraft because of security breach..."
5. **Extreme Weather 7%** "Significant meteorological conditions...such as tornado, blizzard or hurricane."

The BTS notes that the above categorization system makes it seem like weather delays aren't a significant cause. That's because only extreme weather is coded seperately. For example, weather can be what causes an airplane to arrive late or be what is causing delays tagged 'National Aviation System Delay'. **The BTS attributes weather causes to 27% of delay minutes in 2020.**

## Data

To complete this project, we will be using data from several sources.

1. **Bureau of Transportation Statistics: Carrier On-Time Performence Database.** This database contains scheduled and actual departure and arrival times reported by certified U.S. air carriers that account for at least one percent of domestic scheduled passenger revenues. The data is collected by the Office of Airline Information, Bureau of Transportation Statistics (BTS).
2. **weatherAPI** Because weather is such a contributing factor to delays, we'll connect to a weather API service to pull in weather conditions for each flight. This includes information on: average visibility, precipitation, temperatures and wind speeds.
3. **Additional information on airports from the 'airportsdata' library :**  We installed and imported airports data which contains data inside json documents one each airport. This data includes things like: elevation, latitude, longitude, and timezone.

Looks like we have over 7 million records. Below is a dictionary that explains what each column means:

| **Feature**         | **Description**                                                                                                                                                 |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| YEAR                | Year flight took place                                                                                                                                          |
| MONTH               | Month flight took place                                                                                                                                         |
| DAY_OF_MONTH        | Day of the month flight took place                                                                                                                              |
| DAY_OF_WEEK         | Day of the week flight took place (1=Monday, 2 = Tuesday, etc.)                                                                                                 |
| FL_DATE             | Flight Date (yyyymmdd)                                                                                                                                          |
| MKT_CARRIER         | Code assigned by IATA and commonly used to identify a carrier.                                                                                                  |
| MKT_CARRIER_FL_NUM  | Flight Number                                                                                                                                                   |
| OP_CARRIER          | Code assigned by IATA and commonly used to identify a carrier. This one applies to the airline operating a flight (not always the flight that sold the tickets) |
| TAIL_NUM            | Tail number of plane assigned for the flight (unique to each aircraft)                                                                                          |
| OP_CARRIER_FL_NUM   | Flight number for the operating airline                                                                                                                         |
| ORIGIN              | Airport where the flight originated                                                                                                                             |
| DEST                | Destination airport                                                                                                                                             |
| CRS_DEP_TIME        | Scheduled departure time                                                                                                                                        |
| DEP_DELAY           | Departure delay in minutes. Early departures are represented by negative values.                                                                                |
| DEP_DELAY_NEW       | Departure delay in minutes. Early departures are represented by 0.                                                                                              |
| CRS_ARR_TIME        | Arrival delay in minutes. Early arrivals are represented by negative values.                                                                                    |
| ARR_DELAY_NEW       | Arrival delay in minutes. Early arrivals are represented by 0.                                                                                                  |
| CANCELLED           | Cancelled Flight Indicator (1=Yes)                                                                                                                              |
| CRS_ELAPSED_TIME    | Scheduled time of flight in minutes (flight duration in minutes)                                                                                                |
| DISTANCE            | Distance between airports (miles)                                                                                                                               |
| CARRIER_DELAY       | Carrier Delay, in Minutes                                                                                                                                       |
| WEATHER_DELAY       | Weather Delay, in Minutes                                                                                                                                       |
| NAS_DELAY           | National Air System Delay, in Minutes                                                                                                                           |
| SECURITY_DELAY      | Security Delay, in Minutes                                                                                                                                      |
| LATE_AIRCRAFT_DELAY | Late Aircraft Delay, in Minutes                                                                                                                                 |
### Target Distribution
GRAPH SHOWING TARGET DISTRIBUTION

#### Delays by Time of Day

IMAGE

#### Delays by Day of Week
IMAGE

#### Delays by Holiday

## Modeling

#### Methodology

### Results

TABLE SHOWING MODEL RESULTS

CONFUSION MATRIX

IMAGE SHOWING FEATURE IMPORTANCES

## Conclusions

We've reached a stopping point for the modeling portion of this project.

Given more time, future iterations of this product might include things like generating more samples with SMOTE, increasing the training sample size or trying different types of models (including Deep learning).

* We managed to build a model that achieved a 0.229 F1 score and 71% accuracy
* Our model is a custom XGBoost Classifier which succesfully detects 64% of severe delays
* Proximity to holidays, precipitation and certain days of the year are among the strongest predictors
* The biggest gains in performence came from dealing with the severe class imbalance and increasing the sample size of the training data
* We'll next work to productionize this model using Flask and Dash and deploy it to a remote server