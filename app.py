import numpy
from sqlalchemy import extract
from json import JSONEncoder
from datetime import datetime, timedelta
import json
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, and_
pd.set_option('max_rows', 10)
pd.plotting.register_matplotlib_converters()

# %matplotlib inline


covid_data_path = "C:\\Users\\UEA\Desktop\\IV_Coursework\\IV_Backend\\owid-covid-data.csv"
swine_data_path = "C:\\Users\\UEA\Desktop\\IV_Coursework\\IV_Backend\\swine_flu.csv"
ebola_data_path = "C:\\Users\\UEA\Desktop\\IV_Coursework\\IV_Backend\\ebola.csv"
sars_data_path = "C:\\Users\\UEA\Desktop\\IV_Coursework\\IV_Backend\\sars_2003.csv"
vaccinations_path = "C:\\Users\\UEA\Desktop\\IV_Coursework\\IV_Backend\\vaccinations.csv"
covid_data = pd.read_csv(covid_data_path)
swine_data = pd.read_csv(swine_data_path)
ebola_data = pd.read_csv(ebola_data_path)
sars_data = pd.read_csv(sars_data_path)
vaccinations_data = pd.read_csv(vaccinations_path)

# Knn Imputer
nan = np.nan

# Designate the features to become X
features = ['Deaths']
X = swine_data[features]
# Apply KNN imputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
ImputedX = imputer.fit_transform(X)

# Convert output to a data frame to show the stats
imputed_df = pd.DataFrame.from_records(ImputedX)
imputed_df.columns = features
imputed_df['Country'] = swine_data['Country']
imputed_df['Cases'] = swine_data['Cases']
imputed_df['Update Time'] = swine_data['Update Time']


interested_features = [
    'iso_code',
    'continent',
    'location',
    'date',
    'new_cases',
    'new_deaths',
    'new_tests',
    'total_deaths',
    'total_cases'
]
Cleaned_SwineFrame = imputed_df
covid_data.dropna(subset=["continent"], inplace=True)
covid_data.dropna(subset=["population"], inplace=True)
ebola_data.dropna(
    subset=["Cumulative no. of confirmed, probable and suspected cases"], inplace=True)

for i in covid_data[interested_features].columns:
    if covid_data[i].isna().sum() > 0:
        covid_data[i] = covid_data[i].fillna(value=0)

for i in covid_data[interested_features].columns:
    if covid_data[i].isna().sum() > 0:
        print(i, covid_data[i].isna().sum())


# print('Length of values after missing values handling::::::::::::::::::::::::::::::::::::::::::', len(covid_data))
app = Flask(__name__)
CORS(app)


app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/pandemic_db5'
db = SQLAlchemy(app)

new_vaccinations_data = vaccinations_data[['location', 'date',
                                           'daily_vaccinations', 'iso_code']]
new_vaccinations_data = new_vaccinations_data.fillna(value=0)
# print(new_vaccinations_data.isna().sum())


class COVIDENTRY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    iso_code = db.Column(db.String(80))
    continent = db.Column(db.String(80))
    location = db.Column(db.String(80))
    date = db.Column(db.Date)
    new_cases = db.Column(db.Integer)
    new_deaths = db.Column(db.Integer)
    dateTimeStamp = db.Column(db.Integer)
    totalDeaths = db.Column(db.Integer)
    totalCases = db.Column(db.Integer)
    population = db.Column(db.Integer)

    def __init__(
        self, iso_code, continent, location, date, new_cases, new_deaths, dateTimeStamp, totalDeaths, totalCases, population
    ):
        self.iso_code = iso_code
        self.continent = continent
        self.location = location
        self.date = date
        self.new_cases = new_cases
        self.new_deaths = new_deaths
        self.dateTimeStamp = dateTimeStamp
        self.totalDeaths = totalDeaths
        self.totalCases = totalCases
        self.population = population

    def serialize(self):
        return {
            "id": self.id,
            "deaths": self.deaths,
            "confirm": self.confirm,
            "date": str(self.date),
            "country": self.country
        }

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]

    def __repr__(self):
        return '<User %>' % self.name


class SWINEENTRY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confirm = db.Column(db.Integer)
    deaths = db.Column(db.Integer)
    country = db.Column(db.String(80))
    date = db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

    def __repr__(self):
        return '<User %>' % self.name


class EBOLAENTRY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confirm = db.Column(db.Integer)
    deaths = db.Column(db.Integer)
    country = db.Column(db.String(80))
    date = db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

    def __repr__(self):
        return '<User %>' % self.name


class SARSENTRY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    confirm = db.Column(db.Integer)
    deaths = db.Column(db.Integer)
    country = db.Column(db.String(80))
    date = db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

    def __repr__(self):
        return '<User %>' % self.name


class VACCINEENTRY(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    daily_vaccinations = db.Column(db.Integer)
    location = db.Column(db.String(80))
    date = db.Column(db.Date)
    iso_code = db.Column(db.String(80))

    def __init__(self, daily_vaccinations, location, date, iso_code):
        self.daily_vaccinations = daily_vaccinations
        self.location = location
        self.date = date
        self.iso_code = iso_code

    def __repr__(self):
        return '<User %>' % self.name


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


@app.route('/load-adult-data-to-db')
def getUser():

    covidEntries = []
    swineEntries = []
    ebolaEntries = []
    sarsEntries = []
    vaccEntries = []
    for ind in covid_data.index:
        if datetime.strptime(covid_data['date'][ind], '%Y-%m-%d') >= datetime.strptime('2020-2-24', '%Y-%m-%d'):
            covidEntries.append(
                COVIDENTRY(
                    iso_code=str(covid_data['iso_code'][ind]),
                    continent=str(covid_data['continent'][ind]),
                    location=str(covid_data['location'][ind]),
                    new_cases=int(covid_data["new_cases"][ind]),
                    new_deaths=int(covid_data["new_deaths"][ind]),
                    date=datetime.strptime(
                        covid_data['date'][ind], '%Y-%m-%d'),
                    totalCases=int(covid_data["total_cases"][ind]),
                    totalDeaths=int(covid_data["total_deaths"][ind]),
                    population=int(covid_data["population"][ind]),
                    dateTimeStamp=datetime.timestamp(
                        datetime.strptime(covid_data['date'][ind], '%Y-%m-%d'))
                )
            )

    # print(Cleaned_SwineFrame.isna().sum())
    for ind in Cleaned_SwineFrame.index:

        swineEntries.append(
            SWINEENTRY(
                country=str(Cleaned_SwineFrame['Country'][ind]),
                confirm=int(Cleaned_SwineFrame['Cases'][ind]),
                deaths=int(Cleaned_SwineFrame['Deaths'][ind]),
                date=datetime.strptime(Cleaned_SwineFrame['Update Time'][ind], '%m/%d/%Y %H:%M'))
        )

    for ind in ebola_data.index:
        ebolaEntries.append(
            EBOLAENTRY(
                country=str(ebola_data['Country'][ind]),
                confirm=int(
                    ebola_data['Cumulative no. of confirmed, probable and suspected cases'][ind]),
                deaths=int(
                    ebola_data['Cumulative no. of confirmed, probable and suspected deaths'][ind]),
                date=datetime.strptime(ebola_data['Date'][ind], '%Y-%m-%d'))
        )

    for ind in sars_data.index:
        sarsEntries.append(
            SARSENTRY(
                country=str(sars_data['Country'][ind]),
                confirm=int(
                    sars_data['Cumulative number of case(s)'][ind]),
                deaths=int(
                    sars_data['Number of deaths'][ind]),
                date=datetime.strptime(sars_data['Date'][ind], '%Y-%m-%d'))
        )

    for ind in new_vaccinations_data.index:
        vaccEntries.append(
            VACCINEENTRY(
                location=str(new_vaccinations_data['location'][ind]),
                daily_vaccinations=int(
                    new_vaccinations_data['daily_vaccinations'][ind]),
                date=datetime.strptime(
                    new_vaccinations_data['date'][ind], '%Y-%m-%d'),
                iso_code=str(new_vaccinations_data['iso_code'][ind])
            )
        )

    try:
        db.session.bulk_save_objects(covidEntries)
        db.session.bulk_save_objects(swineEntries)
        db.session.bulk_save_objects(ebolaEntries)
        db.session.bulk_save_objects(sarsEntries)
        db.session.bulk_save_objects(vaccEntries)
        db.session.commit()
        json = {
            'name': 'Entries Added',
        }
        return jsonify(json)
    except Exception as e:
        return (str(e))

# Countries for Drop Down


@ app.route('/get-covid-countries', methods=["GET"])
def getCovidCountries():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location).filter(
            COVIDENTRY.new_cases > 0).distinct(COVIDENTRY.location)
        countries = []
        for i in covidentry:
            countries.append(i.location)
        # print(len(countries))
        return jsonify({"countries": countries})
    except Exception as e:
        return (str(e))

# Countries for Drop Down


@ app.route('/get-bar-chart-values', methods=["GET"])
def getAreaChartValues():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location, func.sum(COVIDENTRY.new_cases).label(
            'totalConfirm'), func.sum(COVIDENTRY.new_deaths).label('totalDeaths')).group_by(COVIDENTRY.location).all()
        countriesStats = []
        for i in covidentry:
            countriesStats.append({
                "Country": i[0],
                "ConfirmCases": i[1],
                "ConfirmDeaths": i[2]
            })
        return jsonify({"barStats": countriesStats})
    except Exception as e:
        return (str(e))


@ app.route('/get-bubble-chart-values', methods=["GET"])
def getBubbleChartValues():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location, COVIDENTRY.continent, func.sum(COVIDENTRY.new_cases).label(
            'totalConfirm'), func.sum(COVIDENTRY.new_deaths).label('totalDeaths')).group_by(COVIDENTRY.location, COVIDENTRY.continent).all()
        countriesStats = []
        for i in covidentry:
            countriesStats.append({
                "name": i[0],
                "value": i[2],
                "Continents": i[1],
            })
        return jsonify({"bubbleStats": countriesStats})
    except Exception as e:
        return (str(e))


# API for Time Series Data
@ app.route('/get-confirm-time-chart-values', methods=["GET"])
def getTimeSeriesConfirmedChartValues():
    try:
        covidentry = COVIDENTRY.query.filter_by(**request.args.to_dict()).with_entities(func.sum(COVIDENTRY.new_cases).label(
            'totalConfirm')).order_by(COVIDENTRY.dateTimeStamp).group_by(COVIDENTRY.dateTimeStamp).all()
        response = []
        for i in covidentry:
            response.append(i[0])
        return jsonify({"TImeSeriesChart": response})
    except Exception as e:
        return (str(e))


@ app.route('/get-deaths-time-chart-values', methods=["GET"])
def getTimeSeriesDeathChartValues():
    try:

        covidentry = COVIDENTRY.query.filter_by(**request.args.to_dict()).with_entities(
            func.sum(COVIDENTRY.new_deaths
                     ).label('totalDeaths')
        ).order_by(COVIDENTRY.dateTimeStamp
                   ).group_by(COVIDENTRY.dateTimeStamp).all()

        response = []
        for i in covidentry:
            response.append(i[0])
        return jsonify({"TImeSeriesChart": response})
    except Exception as e:
        return (str(e))


@ app.route('/get-dashboard-stats-values', methods=["GET"])
def getDashboardStatsValues():
    try:

        covidentry = COVIDENTRY.query.with_entities(
            COVIDENTRY.location,
            func.max(COVIDENTRY.totalCases).label('totalCasesConfirm'),
            func.max(COVIDENTRY.totalDeaths).label('totalDeathsConfirm'),
            func.max(COVIDENTRY.population).label('totalPopulation'),
        ).order_by(COVIDENTRY.location
                   ).group_by(COVIDENTRY.location).all()
        response = []
        for i in covidentry:
            response.append({
                "Country": i[0],
                "ConfirmCases": i[1],
                "ConfirmDeaths": i[2],
                "TotalPopulation": i[3],
            })
        return jsonify({"barStats":  sorted(response, key=lambda k: k['ConfirmCases'], reverse=True)})
    except Exception as e:
        return (str(e))

# Get Paramters


@app.route('/get-combineGraph-values')
def getParameter():
    try:
        # func method
        # from sqlalchemy import func
        # Group according to the date of a certain month of a certain year.
        start_date = datetime.strptime('2020-2-24', '%Y-%m-%d')
        end_date = datetime.strptime('2021-4-26', '%Y-%m-%d')
        if 'start_date' in request.args.to_dict():
            start_date = datetime.strptime(
                request.args.to_dict()['start_date'], '%Y-%m-%d')
        if 'end_date' in request.args.to_dict():
            end_date = datetime.strptime(
                request.args.to_dict()['end_date'], '%Y-%m-%d')

        covidEntries = COVIDENTRY.query.filter(
            COVIDENTRY.location == request.args.to_dict()['country'],
            COVIDENTRY.date >= start_date,
            COVIDENTRY.date <= end_date
        ).with_entities(
            func.max(COVIDENTRY.totalCases),
            func.max(COVIDENTRY.totalDeaths),
            COVIDENTRY.location,
            extract('year', COVIDENTRY.date),
            extract('month', COVIDENTRY.date)
        ).group_by(COVIDENTRY.location, extract('year', COVIDENTRY.date), extract('month', COVIDENTRY.date)
                   ).order_by(extract('year', COVIDENTRY.date), extract('month', COVIDENTRY.date)).all()
        response = []
        for i in covidEntries:

            response.append({
                "Cases": i[0],
                "Deaths": i[1],
                "Country": i[2],
                "Year": i[3],
                "month": i[4],
            })

        return jsonify({'combineGraph': response})
    except Exception as e:
        return (str(e))


# Countries Comparison API
@app.route('/get-countries-compared-data', methods=["GET"])
def get_comparison_results():
    try:
        start_date = datetime.strptime('2020-2-24', '%Y-%m-%d')
        end_date = datetime.strptime('2021-4-26', '%Y-%m-%d')
        if 'start_date' in request.args.to_dict():
            start_date = datetime.strptime(
                request.args.to_dict()['start_date'], '%Y-%m-%d')
        if 'end_date' in request.args.to_dict():
            end_date = datetime.strptime(
                request.args.to_dict()['end_date'], '%Y-%m-%d')

        covidEntries = COVIDENTRY.query.filter(
            COVIDENTRY.location.in_(
                [request.args.to_dict()['country1'], request.args.to_dict()['country2']]),
            COVIDENTRY.date >= start_date,
            COVIDENTRY.date <= end_date
        ).with_entities(
            func.max(COVIDENTRY.totalCases),
            func.max(COVIDENTRY.totalDeaths),
            COVIDENTRY.location,
            extract('year', COVIDENTRY.date),
            extract('month', COVIDENTRY.date),
            func.max(COVIDENTRY.population),
        ).group_by(COVIDENTRY.location, extract('year', COVIDENTRY.date), extract('month', COVIDENTRY.date)
                   ).order_by(extract('year', COVIDENTRY.date), extract('month', COVIDENTRY.date)).all()
        # covidentry = COVIDENTRY.query.with_entities(
        #     COVIDENTRY.location,
        #     COVIDENTRY.population,
        #     func.max(COVIDENTRY.totalCases).label('totalCasesConfirm'),
        #     func.max(COVIDENTRY.totalDeaths).label('totalDeathsConfirm'),
        #     COVIDENTRY.continent,
        # ).order_by(COVIDENTRY.location
        #            ).group_by(COVIDENTRY.location, COVIDENTRY.population, COVIDENTRY.continent).all()
        response = []
        for i in covidEntries:
            response.append({
                "Cases": i[0],
                "Deaths": i[1],
                "Country": i[2],
                "Year": i[3],
                "month": i[4],
                "Population": i[5]
            })

        return jsonify({"comparison-result":  response})
    except Exception as e:
        return (str(e))


# Overall Summary API
@app.route('/get-overall-summary-values', methods=["GET"])
def getSummaryValues():
    try:

        covidentry = COVIDENTRY.query.with_entities(
            COVIDENTRY.location,
            COVIDENTRY.population,
            func.max(COVIDENTRY.totalCases).label('totalCasesConfirm'),
            func.max(COVIDENTRY.totalDeaths).label('totalDeathsConfirm'),
            COVIDENTRY.continent,
        ).order_by(COVIDENTRY.location
                   ).group_by(COVIDENTRY.location, COVIDENTRY.population, COVIDENTRY.continent).all()
        response = []
        for i in covidentry:
            response.append({
                "Country": i[0],
                "Population": i[1],
                "ConfirmCases": i[2],
                "ConfirmDeaths": i[3],
                "Continent": i[4],
            })
        return jsonify({"summary":  response})
    except Exception as e:
        return (str(e))


# API for Time Series Data
@app.route('/get-daily-time-series-values', methods=["GET"])
def getDailyTimeSeriesConfirmedChartValues():
    try:
        start_date = datetime.strptime('2020-2-24', '%Y-%m-%d')
        end_date = datetime.strptime('2021-4-26', '%Y-%m-%d')
        if 'start_date' in request.args.to_dict():
            start_date = datetime.strptime(
                request.args.to_dict()['start_date'], '%Y-%m-%d')
        if 'end_date' in request.args.to_dict():
            end_date = datetime.strptime(
                request.args.to_dict()['end_date'], '%Y-%m-%d')
        covidentry = COVIDENTRY.query.filter(
            COVIDENTRY.location == request.args.to_dict()['location'],
            COVIDENTRY.date >= start_date,
            COVIDENTRY.date <= end_date
        ).with_entities(func.sum(COVIDENTRY.new_cases).label(
            'totalConfirm')).order_by(COVIDENTRY.dateTimeStamp).group_by(COVIDENTRY.dateTimeStamp).all()
        coviddeathseries = COVIDENTRY.query.filter(
            COVIDENTRY.location == request.args.to_dict()['location'],
            COVIDENTRY.date >= start_date,
            COVIDENTRY.date <= end_date
        ).with_entities(
            func.sum(COVIDENTRY.new_deaths
                     ).label('totalDeaths')
        ).order_by(COVIDENTRY.dateTimeStamp
                   ).group_by(COVIDENTRY.dateTimeStamp).all()
        response = []
        response1 = []
        for i in covidentry:
            response.append(i[0])
        for i in coviddeathseries:
            response1.append(i[0])
        return jsonify({"TImeSeriesChart": response, "TimeSeriesDeaths": response1})
    except Exception as e:
        return (str(e))


# API for COMPARISON Pandemics
@app.route('/get-swine-countries', methods=["GET"])
def getSwineCountrieResult():
    try:
        swineentry = SWINEENTRY.query.with_entities(
            SWINEENTRY.country).distinct(SWINEENTRY.country)
        countries = []
        for i in swineentry:
            countries.append(i.country)
        # print(len(countries))
        return jsonify({"swine_countries": countries})
    except Exception as e:
        return (str(e))


@app.route('/get-ebola-countries', methods=["GET"])
def getEbolaResult():
    try:
        ebolaentry = EBOLAENTRY.query.with_entities(
            EBOLAENTRY.country).distinct(EBOLAENTRY.country)
        countries = []
        for i in ebolaentry:
            countries.append(i.country)
        # print(len(countries))
        return jsonify({"ebolaentry_countries": countries})
    except Exception as e:
        return (str(e))


@app.route('/get-sars-countries', methods=["GET"])
def getSARSResult():
    try:
        sarsentry = SARSENTRY.query.with_entities(
            SARSENTRY.country).distinct(SARSENTRY.country)
        countries = []
        for i in sarsentry:
            countries.append(i.country)
        # print(len(countries))
        return jsonify({"sars_countries": countries})
    except Exception as e:
        return (str(e))


@app.route('/get-pandemic-100-days-results', methods=["GET"])
def get100DayResult():
    try:
        swine_start_date = datetime.strptime("23-05-2009", "%d-%m-%Y")
        covid_start_date = datetime.strptime('2020-2-24', '%Y-%m-%d')
        ebola_start_date = datetime.strptime('2014-8-29', '%Y-%m-%d')
        sars_start_date = datetime.strptime('2003-3-17', '%Y-%m-%d')
        covidentry = COVIDENTRY.query.filter(
            COVIDENTRY.date >= covid_start_date,
            COVIDENTRY.date <= covid_start_date + timedelta(days=100)
        ).with_entities(func.sum(COVIDENTRY.totalCases).label(
            'totalCases')).order_by(COVIDENTRY.dateTimeStamp).group_by(COVIDENTRY.dateTimeStamp).all()
        covid_response = []
        for i in covidentry:
            covid_response.append(i[0])

        # SWINE Flu Querry for 100 Days
        swineentry = SWINEENTRY.query.filter(
            SWINEENTRY.date >= swine_start_date,
            SWINEENTRY.date <= swine_start_date + timedelta(days=100)
        ).with_entities(func.sum(SWINEENTRY.confirm).label(
            'totalCases')).order_by(SWINEENTRY.date).group_by(SWINEENTRY.date).all()
        swine_response = []
        for i in swineentry:
            swine_response.append(i[0])

        # Ebola Querry for 100 Days
        ebulaentry = EBOLAENTRY.query.filter(
            EBOLAENTRY.date >= ebola_start_date,
            EBOLAENTRY.date <= ebola_start_date + timedelta(days=100)
        ).with_entities(func.sum(EBOLAENTRY.confirm).label(
            'totalCases')).order_by(EBOLAENTRY.date).group_by(EBOLAENTRY.date).all()
        ebola_response = []
        for i in ebulaentry:
            ebola_response.append(i[0])

        # SARS Querry for 100 Days
        sarsentry = SARSENTRY.query.filter(
            SARSENTRY.date >= sars_start_date,
            SARSENTRY.date <= sars_start_date + timedelta(days=100)
        ).with_entities(func.sum(SARSENTRY.confirm).label(
            'totalCases')).order_by(SARSENTRY.date).group_by(SARSENTRY.date).all()
        sars_response = []
        for i in sarsentry:
            sars_response.append(i[0])
        return jsonify({"100_days_Result": {
            "covid_response": covid_response,
            "swine_response": swine_response,
            "ebola_response": ebola_response,
            "sars_response": sars_response
        }})

    except Exception as e:
        return (str(e))


@app.route('/get-pandemic-100-days-deaths-results', methods=["GET"])
def get100DeathsDayResult():
    try:
        swine_start_date = datetime.strptime("23-05-2009", "%d-%m-%Y")
        covid_start_date = datetime.strptime('2020-2-24', '%Y-%m-%d')
        ebola_start_date = datetime.strptime('2014-8-29', '%Y-%m-%d')
        sars_start_date = datetime.strptime('2003-3-17', '%Y-%m-%d')
        covidentry = COVIDENTRY.query.filter(
            COVIDENTRY.date >= covid_start_date,
            COVIDENTRY.date <= covid_start_date + timedelta(days=100)
        ).with_entities(func.sum(COVIDENTRY.totalDeaths).label(
            'totalDeaths')).order_by(COVIDENTRY.dateTimeStamp).group_by(COVIDENTRY.dateTimeStamp).all()
        covid_response = []
        for i in covidentry:
            covid_response.append(i[0])

        # SWINE Flu Querry for 100 Days
        swineentry = SWINEENTRY.query.filter(
            SWINEENTRY.date >= swine_start_date,
            SWINEENTRY.date <= swine_start_date + timedelta(days=100)
        ).with_entities(func.sum(SWINEENTRY.deaths).label(
            'totalDeaths')).order_by(SWINEENTRY.date).group_by(SWINEENTRY.date).all()
        swine_response = []
        for i in swineentry:
            swine_response.append(i[0])

        # Ebola Querry for 100 Days
        ebulaentry = EBOLAENTRY.query.filter(
            EBOLAENTRY.date >= ebola_start_date,
            EBOLAENTRY.date <= ebola_start_date + timedelta(days=100)
        ).with_entities(func.sum(EBOLAENTRY.deaths).label(
            'totalDeaths')).order_by(EBOLAENTRY.date).group_by(EBOLAENTRY.date).all()
        ebola_response = []
        for i in ebulaentry:
            ebola_response.append(i[0])

        # SARS Querry for 100 Days
        sarsentry = SARSENTRY.query.filter(
            SARSENTRY.date >= sars_start_date,
            SARSENTRY.date <= sars_start_date + timedelta(days=100)
        ).with_entities(func.sum(SARSENTRY.deaths).label(
            'totalDeaths')).order_by(SARSENTRY.date).group_by(SARSENTRY.date).all()
        sars_response = []
        for i in sarsentry:
            sars_response.append(i[0])
        return jsonify({"100_days_deaths_Result": {
            "covid_response": covid_response,
            "swine_response": swine_response,
            "ebola_response": ebola_response,
            "sars_response": sars_response
        }})

    except Exception as e:
        return (str(e))


# Vacocination COuntries
@app.route('/get-vaccination-countries', methods=["GET"])
def getVaccinationCountries():
    try:
        vaccentry = VACCINEENTRY.query.with_entities(
            VACCINEENTRY.location).distinct(VACCINEENTRY.location)
        countries = []
        for i in vaccentry:
            countries.append(i.location)
        # print(len(countries))
        return jsonify({"countries": countries})
    except Exception as e:
        return (str(e))


@app.route('/get-vacc-time-chart-values', methods=["GET"])
def getVaccineSeriesValues():
    try:
        start_date = datetime.strptime('2021-2-22', '%Y-%m-%d')
        end_date = datetime.strptime('2021-4-26', '%Y-%m-%d')
        vaccentry = VACCINEENTRY.query.filter(
            VACCINEENTRY.location == request.args.to_dict()['location'],
            VACCINEENTRY.date >= start_date,
            VACCINEENTRY.date <= end_date
        ).with_entities(
            func.sum(VACCINEENTRY.daily_vaccinations).label('totalVaccined'),
            VACCINEENTRY.date
        ).order_by(VACCINEENTRY.date).group_by(VACCINEENTRY.date).all()
        response = []
        for i in vaccentry:
            response.append({
                'vaccines': i[0],
                'date': i[1]
            })
        return jsonify({"TImeSeriesVacc": response})
    except Exception as e:
        return (str(e))


@app.route('/get-vacc-country-values', methods=["GET"])
def getVaccineCountryValues():
    try:
        start_date = datetime.strptime('2021-2-22', '%Y-%m-%d')
        end_date = datetime.strptime('2021-4-26', '%Y-%m-%d')
        vaccentry = VACCINEENTRY.query.filter(
            VACCINEENTRY.date >= start_date,
            VACCINEENTRY.date <= end_date
        ).with_entities(
            func.sum(VACCINEENTRY.daily_vaccinations).label('totalVaccined'),
            VACCINEENTRY.location,
            VACCINEENTRY.iso_code
        ).order_by(VACCINEENTRY.location).group_by(VACCINEENTRY.location, VACCINEENTRY.iso_code).all()
        response = []
        for i in vaccentry:
            response.append({
                'vaccines': i[0],
                'location': i[1],
                'iso_code': i[2]
            })
        return jsonify({"countries-vaccines": response})
    except Exception as e:
        return (str(e))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
