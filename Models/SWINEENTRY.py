from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy()
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

    pass