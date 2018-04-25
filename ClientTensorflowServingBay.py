"""This script wraps the client into a Flask server. It receives POST request with 
prediction data, and forward the data to tensorflow server for inference. 
"""

from flask import Flask, render_template, request, url_for, jsonify
import json
import tensorflow as tf 
import numpy as np 
import os 
import argparse
import sys
from datetime import datetime 

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

tf.app.flags.DEFINE_string('server', 'localhost:9040', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS

app = Flask(__name__)

class mainSessRunning():

    def __init__(self):

        host, port = FLAGS.server.split(':')
        channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = 'BAYMODEL'
        self.request.model_spec.signature_name = 'prediction'

    def inference(self, val_x):
        #temp_data = numpy.random.randn(100, 3).astype(numpy.float32)
        #temp_data = val_x.astype(np.float32).reshape(-1, 3)
        #temp_data = val_x.astype(np.float32)
        #data, label = temp_data, np.sum(temp_data * np.array([1,2,3]).astype(np.float32), 1)
        data = val_x
        print("Data is \n")
        print(data[0])
        print("\n")

        # print("Data shape is \n")
        # print(data.shape)
        # print("\n")

        # print("Label is \n")
        # print(label)
        # print("\n")
        
        self.request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(data))
        #self.request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(data, shape=data.shape))
        result = self.stub.Predict(self.request, 15.0)
        return result

run = mainSessRunning()

print("Initialization done. ")

# Define a route for the default URL, which loads the form
@app.route('/inference', methods=['POST'])
def inference():
    request_data = request.json
    #input_data = np.expand_dims(np.array(request_data), 0)
    result = run.inference(request_data)
    floats = result.outputs['output'].float_val
    # print(floats[0])
    # print("\n")
    #print(result.outputs.float_val)
    return jsonify({'ScaledPrediction': floats[0]})
    
@app.route('/test', methods=['GET'])
def test_serv():
    return ("Hello")
        
if __name__ == "__main__":
    app.run(host= '0.0.0.0')







