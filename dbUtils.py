from sqlalchemy import create_engine  
  
class DbUtils:  
    db_string = "postgresql+psycopg2://admin:admin@localhost/postgres"  
      
    def createTable(self):  
        db = create_engine(self.db_string)  
        db.execute("CREATE TABLE IF NOT EXISTS films (title text, director text, year text)")    
  
    def addNewFilm(self, title, director, year):  
        db_string = "postgresql+psycopg2://admin:admin@localhost/postgres"  
        db = create_engine(db_string)  
        db.execute("INSERT INTO films(title, director, year) VALUES (%s,%s, %s)", title, director, year)  
  
    def getFilms(self):  
        db_string = "postgresql+psycopg2://admin:admin@localhost/postgres"  
        db = create_engine(db_string)  
        films = db.execute("SELECT * FROM films")  
        return films