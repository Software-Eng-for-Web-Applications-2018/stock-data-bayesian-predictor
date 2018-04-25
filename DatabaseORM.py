#
# CREATED BY JOHN GRUN
#   APRIL 18 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#
from config import (MYSQL_DB_URI)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, DateTime

#Database Connection 

Base = declarative_base();


class StockPriceDay(Base):

    __tablename__ = 'stock_price_day'

    dateid = Column(Date, primary_key=True)
    sym = Column(String(5), primary_key=True)
    volume = Column(Float)
    close = Column(Float)
    high = Column(Float)
    _open = Column('open', Float)
    low = Column(Float)


class StockPriceMinute(Base):

    __tablename__ = 'stock_price_minute'

    dateid = Column(DateTime, primary_key=True)
    sym = Column(String(5), primary_key=True)
    volume = Column(Float)
    close = Column(Float)
    high = Column(Float)
    _open = Column('open', Float)
    low = Column(Float)


engine = create_engine(MYSQL_DB_URI,echo=True);

Session = sessionmaker(bind=engine);
session = Session();

#print(session.query(StockPriceMinute.sym).first()) 
