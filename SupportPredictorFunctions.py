import os
import tensorflow as tf
import numpy as np
import sys
from DatabaseORM import session, StockPriceMinute

def GetStockDataList(session,DatabaseTables,stocksym):
    #Grabs the given data elements from the passed in database table object as defined in the ORM for a given stock symbol
    QueryList = session.query(DatabaseTables.high,DatabaseTables.low,DatabaseTables.volume).filter(DatabaseTables.sym == stocksym).all()
    #QueryList = session.query(DatabaseTables.high).filter(DatabaseTables.sym == stocksym).all()
    #, StockPriceMinute.volume
    Formatedlist = []; 
    for DataElement in  QueryList:
        Formatedlist.append(list(DataElement))
    # Returns a np.array of [[row],[row],[row],[row]] where row = [element1, element2,etc] rows are defined by the query above
    return np.array(Formatedlist);
    #print(Formatedlist)

# Xdata = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]);

def SaveModelAndQuit(sessionname,ModelName):
    saver = tf.train.Saver()  
    ModelName = './' + ModelName
    saver.save(sessionname, ModelName)
    print("Exiting Normally\n")
    sys.exit(0)

# def RestoreModel(ModelName):
#     new_saver = tf.train.import_meta_graph(ModelName)
#     new_saver.restore(sess, tf.train.latest_checkpoint('./')) 
#     graph = tf.get_default_graph()
#     w1 = graph.get_tensor_by_name("w1:0")
#     w2 = graph.get_tensor_by_name("w2:0")
#     feed_dict ={w1:13.0,w2:17.0}
