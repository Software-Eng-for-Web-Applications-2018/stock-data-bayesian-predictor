

#Test of the database ORM.# . WORKS 
#
# CREATED BY JOHN GRUN
#   APRIL 18 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#


from DatabaseORM import session, StockPriceMinute

# QueryList = session.query(StockPriceMinute.high, StockPriceMinute.volume).filter(StockPriceMinute.sym == 'AMD').all()
# Formatedlist = []; 
# for DataElement in  QueryList:
# 	Formatedlist.append(list(DataElement))

# print(Formatedlist)
# . WORKS 


def GetStockPriceVolumeList(session,stocksym):
	QueryList = session.query(StockPriceMinute.high, StockPriceMinute.volume).filter(StockPriceMinute.sym == stocksym).all()
	
	Formatedlist = []; 
	for DataElement in  QueryList:
		Formatedlist.append(list(DataElement))
	return Formatedlist;
	#print(Formatedlist)



print(GetStockPriceVolumeList(session,'AMD'))