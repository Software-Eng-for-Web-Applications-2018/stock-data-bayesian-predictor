# BAYESIAN

#
# CREATED BY JOHN GRUN
#   APRIL 25 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#

#Based upon examples from the tensorflow cookbook
#https://github.com/data61/aboleth

import os
import argparse;
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
import sys
from sklearn.preprocessing import MinMaxScaler

from DatabaseORM import session, StockPriceMinute, StockPriceDay
from DataArrayTools import ShitftAmount,TrimArray
from SupportPredictorFunctions import GetStockDataList, SaveModelAndQuit



import logging
import bokeh.plotting as bk
from sklearn.metrics import r2_score
import aboleth as ab



ops.reset_default_graph()
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '', 'Working directory.')
tf.app.flags.DEFINE_string('sym', '', 'Stock Symbol')  
tf.app.flags.DEFINE_integer('shiftamount', 1, 'Amount of time we wish to attept to predict into the future')
tf.app.flags.DEFINE_integer('DEBUG', 0, 'Enable the debugging output') 
tf.app.flags.DEFINE_integer('RT', 0, 'Future 0 or Historical 1') 
FLAGS = tf.app.flags.FLAGS

# adapted from aboleth tutorials 
#https://github.com/data61/aboleth/blob/master/demos/regression_tutorial.py

def bayesian_linear(X, Y,NumberOfSamples):
    """Bayesian Linear Regression."""
    lambda_ = 100.
    std = (1 / lambda_) ** .5  # Weight st. dev. prior
    noise = tf.Variable(1.)  # Likelihood st. dev. initialisation, and learning

    net = (
        ab.InputLayer(name="X", n_samples=NumberOfSamples) >>
        ab.DenseVariational(output_dim=1, prior_std=std, full=True)
    )

    f, kl = net(X=X)
    lkhood = tf.distributions.Normal(loc=f, scale=ab.pos(noise)).log_prob(Y)
    loss = ab.elbo(lkhood, kl, 100)

    return f, loss



def GetLastNElements(NumberOfElements,Data):
    # Grabs the given numer of elements off the end of the loaded array
    return Data[len(Data)- NumberOfElements : len(Data)];

def GetS_Matrix(Alpha,Beta,MPolyDegree,EphiX,phiX):
    return numpy.linalg.inv(Alpha*numpy.identity(MPolyDegree)+Beta*numpy.dot(EphiX,phiX.T));

def GetVariance(Beta,phiX,S):
    return (1/Beta) + numpy.dot((phiX.T),numpy.dot(S,phiX));

def GetMean(Beta,phiX,S,ET):
    return Beta*numpy.dot(phiX.T,numpy.dot(S,ET));


def GetPhiX(MPolyDegree,Exponentbase):
    phiX = numpy.zeros((MPolyDegree,1),float)
    for i in range(MPolyDegree):
        phiX[i]=math.pow(Exponentbase,i);
    return phiX;

def GetEPhiX(ExponentsList, NumberOfElements,MPolyDegree):
    EphiX = numpy.zeros((MPolyDegree,1),float);

    for j in range(NumberOfElements):
        for i in range(MPolyDegree):
            EphiX[i]+=math.pow(ExponentsList[j],i);
    return EphiX;

def GetET(Data,EPhiX,NumberOfElements,MPolyDegree):
    ET  = numpy.zeros((MPolyDegree,1),float);
    for j in range(NumberOfElements): 
        for i in range(MPolyDegree):
            ET[i] = Data[j]*EPhiX[i];
    return ET;

def GetExponentsList(UpperLimit):
    ExponentsList =[]   # powers of x in phix(n)
    for i in range(1, (UpperLimit+1)):
        ExponentsList.append(i)
    return ExponentsList;

# function to find the prediction value
def BayesianCurveFit(Alpha,Beta,MPolyDegree,Data,NumOfElements):
    
    DataSet = []
    DataSet.append(Data);

    #print "------------ numpy ----------------- "
    ColumnVector = numpy.array(DataSet).transpose()
    #print ColumnVector

    # the referenced book has one of the worst explainations of this concept
    # See https://www.youtube.com/watch?v=5NMxiOGL39M&t=3s

    #This is the sum of the elements  
    Explist = GetExponentsList(NumOfElements);

    EPhiX = GetEPhiX(Explist, NumOfElements,MPolyDegree);

    ET = GetET(ColumnVector,EPhiX, NumOfElements, MPolyDegree);

    phiX = GetPhiX(MPolyDegree,Explist[NumOfElements-1]);
    # The magical S matrix. 
    S_Matrix = GetS_Matrix(Alpha,Beta,MPolyDegree,EPhiX,phiX);
    # Based on formula (1.70)
    # The mean should be the next prediction
    Predicted = GetMean(Beta,phiX,S_Matrix,ET);
    #Do we even use this???????????
    # Based on formula (1.71)  umm ok what do I do with this...........
    Variance = GetVariance(Beta,phiX,S_Matrix);
    
    return (Predicted[0][0], Variance, ColumnVector[0]);


def TrainBayesian(session,DatabaseTables,stocksym,RelativeTimeShift,DEBUG,typeString):
    # Create graph
    sess = tf.Session()


    #Xdata = GetStockDataList(session,StockPriceMinute,'AMD');
    Xdata = GetStockDataList(session,DatabaseTables,stocksym);

    #print(Xdata)

    # Shitf the training dat by X timeuits into the "future"
    Ydata = ShitftAmount(Xdata,RelativeTimeShift)

    #Make the data arrays the same length 
    Xdata = TrimArray(Xdata,(-1*RelativeTimeShift))

    LengthOfDataSet = len(Xdata)

    train_start = 0
    train_end = int(np.floor(0.8*LengthOfDataSet))
    test_start = train_end + 1
    test_end = LengthOfDataSet

    Xdata_train = Xdata[np.arange(train_start, train_end), :]
    Ydata_train = Ydata[np.arange(train_start, train_end), :]

    Xdata_test = Xdata[np.arange(test_start, test_end), :]
    Ydata_test = Ydata[np.arange(test_start, test_end), :]

    #Scale the data -- Really more for comparison between this and other prediction outputs
    Xscaler = MinMaxScaler(feature_range=(-1, 1))
    Xscaler.fit(Xdata_train)
    Xdata_train = Xscaler.transform(Xdata_train)
    Xdata_test = Xscaler.transform(Xdata_test)

    Yscaler = MinMaxScaler(feature_range=(-1, 1))
    Yscaler.fit(Ydata_train)
    Ydata_train = Yscaler.transform(Ydata_train)
    Ydata_test = Yscaler.transform(Ydata_test)

    # # This svm is only 1 dim at the moment
    # Xdata_train = np.array([x[0] for x in Xdata_train])
    # Xdata_test = np.array([x[0] for x in Xdata_test])

    # #XdataTrainTest = [Xdata_train, Xdata_test]
    # #Roll seems to be rolling several axes for some damn reason. 
    # Ydata_train = np.array([y[2] for y in Ydata_train])
    # Ydata_test = np.array([y[2] for y in Ydata_test])

    # YdataTrainTest = [Ydata_train, Ydata_test]

    batch_size = 50



    # # Build X and y
    X_train = Xdata_train;
    y_train = Ydata_train[:, 2]

    if(DEBUG == 1):
        print("X data\n")
        print(X_train)
        print("\n")
        print("y_train\n")
        print(y_train[0])

    # # print(y_train)
    X_test = Xdata_test
    y_test = Ydata_test[:, 2]

    #NumElementsPerRow = Xdata_train.shape[1]
    # NumElementsOut = y_train.shape[0]

    # if(DEBUG == 1):
    #     print("Length of y_train "+ str( len( y_train) ))
    #     print("NumElementsPerRow " + str(NumElementsPerRow) + " NumElementsOut " + str(NumElementsOut))

    # Initialize placeholders
    NumElementsPerRow = X_train.shape[1]
    NumElementsOut = y_train.shape[0]

    X = tf.placeholder('float', shape=[None, NumElementsPerRow])
    Y = tf.placeholder('float', shape=[None])

    # Declare model operations
    #Do the Bayesian magic here
    Out, loss = bayesian_linear(X, Y, LengthOfDataSet);
    #Loss is related to Y...

    #global_step = tf.train.create_global_step()
    #train = optimizer.minimize(loss, global_step=global_step)
    Cost = loss
    Optimizer = tf.train.AdamOptimizer().minimize(Cost, global_step=tf.train.create_global_step())

    # Cost function
    #Cost = tf.reduce_mean(tf.squared_difference(Out, Y))

    sess.run(tf.global_variables_initializer())


    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)


      # # Setup plot
    if(DEBUG == 1):
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        line1, = ax1.plot(y_test)
        line2, = ax1.plot(y_test * 0.5)
        plt.show()

    # Fit Bayesian 
    batch_size = 1
    Cost_train = []
    Cost_test = []

    # Run
    epochs = 5
    for e in range(epochs):
        print('Epoch' + str(e))

        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run Optimizerimizer with batch
            sess.run(Optimizer, feed_dict={X: batch_x, Y: batch_y})

            # Show progress
            if np.mod(i, 50) == 0:
                # Cost train and test
                # Cost_train.append(sess.run(Cost, feed_dict={X: X_train, Y: y_train}))
                # Cost_test.append(sess.run(Cost, feed_dict={X: X_test, Y: y_test}))

                # Prediction
                # pred = sess.run(Out, feed_dict={X: X_test})
                # Error = np.average(np.abs(y_test - pred))
                if(1):
                     #ModelName = 'NN' + stocksym
                    #SaveModelAndQuit(net,ModelName)
                        # Export model
                    export_path_base = FLAGS.work_dir + 'NN_' + typeString + '_'+ stocksym
                    export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
                    #export_path = ModelName + '/' + export_path 
                    print('Exporting trained model to', export_path)
                    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

                    tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
                    tensor_info_y = tf.saved_model.utils.build_tensor_info(Out) #THIS IS IMPORTANT!!! NOT THE PLACEHOLDER!!!!!!!!

                    prediction_signature = (
                        tf.saved_model.signature_def_utils.build_signature_def(
                          inputs={'input': tensor_info_x},
                          outputs={'output': tensor_info_y},
                          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

                    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
                    builder.add_meta_graph_and_variables(
                        sess, [tf.saved_model.tag_constants.SERVING],
                        signature_def_map={
                          'prediction':
                              prediction_signature,
                      },
                      legacy_init_op=legacy_init_op)

                    builder.save()

                    print('Done exporting!')
                    sys.exit(0)


                if(DEBUG == 1):
                   # line2.set_ydata(pred)
                    plt.title('Epoch ' + str(e) + ', Batch ' + str(i))


                    print('Cost Train: ', Cost_train[-1])
                    print('Cost Test: ', Cost_test[-1])
                    print("Pred shape 0: " + str(pred.shape[0]) + ", 1: " + str(pred.shape[1]))
                    print(pred)
                    print("\n")
                    
                    print("Error\n")
                    print(Error)
                    print("\n")

            if(DEBUG == 1):
                plt.pause(0.01)
    # epoch = 300

    # # Training loop
    # train_loss = []
    # test_loss = []
    # for i in range(epoch):
    #     rand_index = np.random.choice(len(Xdata_train), size=batch_size)
    #     rand_x = np.transpose([Xdata_train[rand_index]])
    #     rand_y = np.transpose([Ydata_train[rand_index]])
    #     #rand_x = Xdata_train[rand_index]
    #     #rand_y = Ydata_train[rand_index]
    #     sess.run(train_step, feed_dict={X: rand_x, Y: rand_y})
        
    #     temp_train_loss = sess.run(loss, feed_dict={X: np.transpose([Xdata_train]), Y: np.transpose([Ydata_train])})
    #     train_loss.append(temp_train_loss)
        
    #     temp_test_loss = sess.run(loss, feed_dict={X: np.transpose([Xdata_test]), Y: np.transpose([Ydata_test])})
    #     test_loss.append(temp_test_loss)

    #     if(DEBUG == 1):
    #         if (i+1)%50==0:
    #             print('-----------')
    #             print('Generation: ' + str(i+1))
    #             print('A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
    #             print('Train Loss = ' + str(temp_train_loss))
    #             print('Test Loss = ' + str(temp_test_loss))


    # if(DEBUG == 1):
    #     # Extract Coefficients
    #     [[slope]] = sess.run(A)
    #     [[y_intercept]] = sess.run(b)
    #     [width] = sess.run(epsilon)

    #     # Get best fit line
    #     best_fit = []
    #     best_fit_upper = []
    #     best_fit_lower = []
    #     for i in Xdata_train:
    #       best_fit.append(slope*i+y_intercept)
    #       best_fit_upper.append(slope*i+y_intercept+width)
    #       best_fit_lower.append(slope*i+y_intercept-width)


    #     # Plot fit with data
    #     plt.plot(Xdata_train, Ydata_train, 'o', label='Data Points')
    #     plt.plot(Xdata_train, best_fit, 'r-', label='SVM Regression Line', linewidth=3)
    #     plt.plot(Xdata_train, best_fit_upper, 'r--', linewidth=2)
    #     plt.plot(Xdata_train, best_fit_lower, 'r--', linewidth=2)
    #     plt.ylim([0, 1])
    #     plt.legend(loc='lower right')
    #     plt.title('Current Price vs Future Price')
    #     plt.xlabel('Current Price')
    #     plt.ylabel('Future Price')
    #     plt.show()

    #     # Plot loss over time
    #     plt.plot(train_loss, 'k-', label='Train Set Loss')
    #     plt.plot(test_loss, 'r--', label='Test Set Loss')
    #     plt.title('L2 Loss per Generation')
    #     plt.xlabel('Generation')
    #     plt.ylabel('L2 Loss')
    #     plt.legend(loc='upper right')
    #     plt.show()

    # ModelName = 'BAY'+ stocksym
    # #SaveModelAndQuit(sess,ModelName)

    #  # Export model
    # export_path_base = FLAGS.work_dir + 'BAY_'+ typeString + '_'+stocksym
    # export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
    # #export_path = ModelName + '/' + export_path 
    # print('Exporting trained model to', export_path)
    # builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(Out) #THIS IS IMPORTANT!!! NOT THE PLACEHOLDER!!!!!!!!

    # prediction_signature = (
    #     tf.saved_model.signature_def_utils.build_signature_def(
    #       inputs={'input': tensor_info_x},
    #       outputs={'output': tensor_info_y},
    #       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    # builder.add_meta_graph_and_variables(
    #     sess, [tf.saved_model.tag_constants.SERVING],
    #     signature_def_map={
    #       'prediction':
    #           prediction_signature,
    #   },
    #   legacy_init_op=legacy_init_op)

    # builder.save()

    # print('Done exporting!')
    # sys.exit(0)


def main():

    #user input
    # learning rate 
    parser = argparse.ArgumentParser();
    parser.add_argument('--sym', dest= 'sym', default='');
    parser.add_argument('--DEBUG',type=int, dest= 'debug', default=0);
    parser.add_argument('--shiftamount',type=int, dest= 'shiftamount', default=1);
    parser.add_argument('--RT',type=int, dest= 'history', default=0);

    args = parser.parse_args();

    print("Input Arguments: ") 
    print(args)
    if(args.history == 0):
        TrainBayesian(session,StockPriceMinute,args.sym,args.shiftamount,args.debug,'RT');
    else:
        TrainBayesian(session,StockPriceDay,args.sym,args.shiftamount,args.debug,'PAST');

main();
