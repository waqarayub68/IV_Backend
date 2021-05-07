from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
pd.set_option('max_rows', 10)
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns




covid_data_path = "F:\covid-19-data\public\data\owid-covid-data.csv"
swine_data_path = "J:\swine_flu.csv"
covid_data = pd.read_csv(covid_data_path)
swine_data = pd.read_csv(swine_data_path)



# Knn Imputer
from sklearn.impute import KNNImputer
nan = np.nan

# Designate the features to become X
features=['Deaths']
X= swine_data[features]
# Apply KNN imputer
imputer = KNNImputer(n_neighbors=2, weights="uniform")
ImputedX=imputer.fit_transform(X)

# Convert output to a data frame to show the stats
imputed_df = pd.DataFrame.from_records(ImputedX)
imputed_df.columns = features
imputed_df['Country'] = swine_data['Country']
imputed_df['Cases'] = swine_data['Cases']
imputed_df['Update Time'] = swine_data['Update Time']


# Categorical Encoders
# import category_encoders as ce
# enc = ce.OrdinalEncoder(cols=["Country","Update Time"],handle_missing='return_nan',return_df= True)

#We now fit the model and transform the data and put it in X which is a dataframe
# X=enc.fit_transform(imputed_df)


# Outlier Detection
# from sklearn.neighbors import LocalOutlierFactor
# clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# y_pred = clf.fit_predict(X)
# totalOutliers=0
# for pred in y_pred:
#     if pred == -1:
#         totalOutliers=totalOutliers+1
# print ("Number of predicted outliers:",totalOutliers)

# # Removing Outliers
# X['Country'] = imputed_df['Country']
# X['Update Time'] = imputed_df['Update Time']
# mask2 = (y_pred != -1)
# # print(len(preds),len(X))
Cleaned_SwineFrame = imputed_df
# print('******************************************************')
# print(len(Cleaned_SwineFrame))


# Converting to log to show in boxplot
# plt.figure(figsize=(8,6))
# transformed_df = {'transformed_cases': imputed_df["Cases"].apply(np.log), 'transformed_death': imputed_df["Deaths"].replace(0,np.nan).apply(np.log)}
# print(transformed_df["transformed_death"])
# df = pd.DataFrame (transformed_df, columns = ['transformed_cases','transformed_death'])
# sns.boxplot(data=df)
# sns.boxplot(imputed_df["Deaths"].apply(np.log))
# plt.show()

# --------------------- COVID Data Pre Processing -------------------
# Missing Values
# missing_covid_values_count = covid_data.isnull().sum()
# missing_covid_values_count

# covid_data_filtered_frame = covid_data[['Country/Region', 'Confirmed', 'Deaths', 'Date']]
# covid_data_filtered_frame.sample(3)


# Encoder Categorical
# import category_encoders as ce
# enc1 = ce.OrdinalEncoder(cols=["Country/Region","Date"],handle_missing='return_nan',return_df= True)

# #We now fit the model and transform the data and put it in X which is a dataframe
# COVIDX=enc1.fit_transform(covid_data_filtered_frame)
# print(COVIDX.sample(3))


# Outlier Detection In COVID Data
# covid_clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
# y_covid_pred = covid_clf.fit_predict(COVIDX)
# totalOutliers=0
# for pred in y_covid_pred:
#     if pred == -1:
#         totalOutliers=totalOutliers+1
# # print("Number of predicted outliers:", len(covid_data))
# # print("Number of predicted outliers:", totalOutliers)

# COVIDX['Country/Region'] = covid_data_filtered_frame['Country/Region']
# COVIDX['Date'] = covid_data_filtered_frame['Date']
# # print(COVIDX[COVIDX['Deaths']>10])

# mask3 = (y_covid_pred != -1)
# # print(len(preds),len(X))
# Cleaned_COVIDFrame = COVIDX[mask3]
# print(len(Cleaned_COVIDFrame))

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

covid_data.dropna(subset = ["continent"], inplace=True)

for i in covid_data[interested_features].columns:
    if covid_data[i].isna().sum() > 0:
        covid_data[i] = covid_data[i].fillna(value=0)

for i in covid_data[interested_features].columns:
    if covid_data[i].isna().sum() > 0:
        print(i, covid_data[i].isna().sum())


# print('Length of values after missing values handling::::::::::::::::::::::::::::::::::::::::::', len(covid_data))
app = Flask(__name__)
CORS(app)

import json
from datetime import datetime
from json import JSONEncoder
import numpy

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:root@localhost:5432/pandemic_db6'
db = SQLAlchemy(app)


class COVIDENTRY(db.Model):
    id=db.Column(db.Integer, primary_key=True)
    iso_code=db.Column(db.String(80))
    continent=db.Column(db.String(80))
    location=db.Column(db.String(80))
    date=db.Column(db.Date)
    new_cases=db.Column(db.Integer)
    new_deaths=db.Column(db.Integer)
    dateTimeStamp=db.Column(db.Integer)
    totalDeaths=db.Column(db.Integer)
    totalCases=db.Column(db.Integer)
    def __init__(self, iso_code, continent, location, date, new_cases, new_deaths, dateTimeStamp, totalDeaths, totalCases):
        self.iso_code = iso_code
        self.continent = continent
        self.location = location
        self.date = date
        self.new_cases = new_cases
        self.new_deaths = new_deaths
        self.dateTimeStamp = dateTimeStamp
        self.totalDeaths = totalDeaths
        self.totalCases = totalCases

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
    id=db.Column(db.Integer, primary_key=True)
    confirm=db.Column(db.Integer)
    deaths=db.Column(db.Integer)
    country=db.Column(db.String(80))
    date=db.Column(db.Date)

    def __init__(self, confirm, deaths, country, date):
        self.confirm = confirm
        self.deaths = deaths
        self.country = country
        self.date = date

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
    for ind in covid_data.index:
        if datetime.strptime(covid_data['date'][ind], '%Y-%m-%d') >= datetime.strptime('2020-2-24', '%Y-%m-%d'):
            covidEntries.append(
                COVIDENTRY( 
                    iso_code= str(covid_data['iso_code'][ind]),
                    continent= str(covid_data['continent'][ind]),
                    location= str(covid_data['location'][ind]),
                    new_cases=int(covid_data["new_cases"][ind]),
                    new_deaths=int(covid_data["new_deaths"][ind]),
                    date=datetime.strptime(covid_data['date'][ind], '%Y-%m-%d'),
                    totalCases=int(covid_data["total_cases"][ind]),
                    totalDeaths=int(covid_data["total_deaths"][ind]),
                    dateTimeStamp=datetime.timestamp(datetime.strptime(covid_data['date'][ind], '%Y-%m-%d'))
                )
            )
    
    
    # print(Cleaned_SwineFrame.isna().sum())
    for ind in Cleaned_SwineFrame.index:
        # if (pd.isnull(swine_data['Deaths'][ind]) == False):
        swineEntries.append(
            SWINEENTRY( 
                country= str(Cleaned_SwineFrame['Country'][ind]),
                confirm= int(Cleaned_SwineFrame['Cases'][ind]),
                deaths=int(Cleaned_SwineFrame['Deaths'][ind]),
                date=Cleaned_SwineFrame['Update Time'][ind]
            )
        )
    
    try:
        db.session.bulk_save_objects(covidEntries)
        db.session.bulk_save_objects(swineEntries)
        db.session.commit()
        json = {
            'name':'Entries Added',
        }
        return jsonify(json)
    except Exception as e:
        return (str(e))

# Countries for Drop Down
@app.route('/get-covid-countries', methods=["GET"])
def getCovidCountries():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location).filter(COVIDENTRY.new_cases > 0).distinct(COVIDENTRY.location)
        countries = []
        for i in covidentry:
            countries.append(i.location)
        # print(len(countries))
        return jsonify({"countries": countries})
    except Exception as e:
        return (str(e))

# Countries for Drop Down
@app.route('/get-bar-chart-values', methods=["GET"])
def getAreaChartValues():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location, func.sum(COVIDENTRY.new_cases).label('totalConfirm'), func.sum(COVIDENTRY.new_deaths).label('totalDeaths')).group_by(COVIDENTRY.location).all()
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

# Countries for Drop Down
@app.route('/get-bubble-chart-values', methods=["GET"])
def getBubbleChartValues():
    try:
        covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.location, COVIDENTRY.continent, func.sum(COVIDENTRY.new_cases).label('totalConfirm'), func.sum(COVIDENTRY.new_deaths).label('totalDeaths')).group_by(COVIDENTRY.location, COVIDENTRY.continent).all()
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
@app.route('/get-confirm-time-chart-values', methods=["GET"])
def getTimeSeriesConfirmedChartValues():
    try:
        covidentry = COVIDENTRY.query.with_entities(func.sum(COVIDENTRY.new_cases).label('totalConfirm')).order_by(COVIDENTRY.dateTimeStamp).group_by(COVIDENTRY.dateTimeStamp).all()
        response = []
        for i in covidentry:
            response.append(i[0])
        return jsonify({"TImeSeriesChart": response})
    except Exception as e:
        return (str(e))

@app.route('/get-deaths-time-chart-values', methods=["GET"])
def getTimeSeriesDeathChartValues():
    try:
        # covidentry = COVIDENTRY.query.with_entities(COVIDENTRY.new_deaths, COVIDENTRY.dateTimeStamp).all()
        # covidentry = COVIDENTRY.query.with_entities(
        #     COVIDENTRY.dateTimeStamp, 
        #     func.sum(COVIDENTRY.new_deaths).label('totalDeaths')
        #     ).group_by(COVIDENTRY.dateTimeStamp).all()
        covidentry = COVIDENTRY.query.with_entities(
            func.sum(COVIDENTRY.new_deaths
            ).label('totalDeaths')
            ).order_by(COVIDENTRY.dateTimeStamp
            ).group_by(COVIDENTRY.dateTimeStamp).all()
        # print(covidentry)
        response = []
        for i in covidentry:
            response.append(i[0])
        return jsonify({"TImeSeriesChart": response})
    except Exception as e:
        return (str(e))


@app.route('/get-dashboard-stats-values', methods=["GET"])
def getDashboardStatsValues():
    try:
        
        covidentry = COVIDENTRY.query.with_entities(
            COVIDENTRY.location,
            func.max(COVIDENTRY.totalCases).label('totalCasesConfirm'),
            func.max(COVIDENTRY.totalDeaths).label('totalDeathsConfirm'),
            ).order_by(COVIDENTRY.location
            ).group_by(COVIDENTRY.location).all()
        response = []
        for i in covidentry:
            response.append({
                "Country": i[0],
                "ConfirmCases": i[1],
                "ConfirmDeaths": i[2],
            })
        return jsonify({"barStats": response})
    except Exception as e:
        return (str(e))

# Get Paramters
# @app.route('/get-parameter')
# def getParameter():
#     # return parameter + " " + optional_parameter
#     print(request.args.to_dict())
#     return jsonify(request.args.to_dict())


# @app.route('/')
# def hello():
#     encodedNumpyData = json.dumps(adult_data.columns.to_numpy(), cls=NumpyArrayEncoder)
#     return jsonify({
#         "csv_columns": json.loads(encodedNumpyData)
#     })

# @app.route('/sum', methods=["GET", "POST"])
# def sum():
#     print(request)
#     data = request.get_json()
#     if "a" not in data:
#         return jsonify({
#             "error": 'A variable not found'
#         })
#     # print(data)
#     return jsonify({
#         "SUM": data["a"] + data["b"]
#     })


# @app.route('/bye')
# def bye():
#     try:
#         users=[
#             User(name= 'Joseph'),
#             User(name= 'Simon'),
#             User(name= 'Fawad'),
#         ]
#         db.session.bulk_save_objects(users)
#         db.session.commit()
#         json = {
#             'name':'Users Added ',
#         }
#         return jsonify(json)
#     except Exception as e:
#         return (str(e))

# @app.route('/get-user')
# def getUser():
#     try:
#         user = User.query.filter_by(name='usama').count()
#         json = {
#             'name': user,
#         }
#         return jsonify(json)
#     except Exception as e:
#         return (str(e))




if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
