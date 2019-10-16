import sys
import copy
import json
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense,Activation, Conv2D, BatchNormalization, ReLU,Flatten, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, model_from_json

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns

#Custom plotting functions
import lib.Plots as p

sns.set_context('poster')
sns.set(font_scale=3.0)

DEBUG = False
TRAIN_MODEL = False
SAVE_MODEL = False
LOAD_MODEL = True

ONEHOT_OUTPUTDATA = False
NUM_TRAIN_EVENTS = 2000
NUM_VALIDATE_EVENTS = 500

numpy_infilename = "./data/LiliaComb_05072019_pixelmap_input.npy"
numpy_outfilename = "./data/LiliaComb_05072019_pixelmap_output.npy"
weights_filename = "./model_out/ringfinder_weights_0.h5"
model_filename = "./model_out/ringfinder_model_0.json"

def define_model_struct(data_in):
    print("#### TRAINING MODEL WITH INPUT DATA ####")
    
    xp = Conv2D(32,3)(data_in)
    x = ReLU()(xp)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(32,3, strides=2)(x)
    x = Dropout(.1)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3)(data_in)
    #xp = Add()([x,Conv2D(64,3, strides=2)(xp)])
    x = Dropout(.1)(x)    
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3, strides=2)(x)
    x = Dropout(.1)(x)    
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    #x = Conv2D(128,3)(x)
    #x = Add()([x,Conv2D(128,3,strides=2)(xp)])
    x = Dropout(.1)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Dense(output_width)(x)
    x = Activation('sigmoid')(x)
    
    cnnmodel = Model(inputs=data_in,outputs=x)
    return cnnmodel

def open_numpy_allowpickle(thefile):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True,**k)
    alleventdata = np.load(thefile)
    np.load = np_load_old
    return alleventdata 

def OneHotEncodeBinaryArgs(output_data):
    '''
    Takes in an array of outputs and one-hot encodes the
    output data.  A single output is assumed to be of arbitrary length,
    but each entry in the output could be either zero or one.  

    Example where an entry in output data has pi0 and/or piminus only:

    0 0             1 0               0 1             1 1
     |               |                 |               |
     V               V                 V               V
    1 0 0 0        0 1 0 0           0 0 1 0         0 0 0 1
    '''
    numoutputs = len(output_data[0])
    num_onehots = (numoutputs)**2
    onehot_output = []
    for j,entry in enumerate(output_data): #For each entry, put a 1 in the proper onehot position
        onehot_ind = None
        this_onehotoutput = np.zeros(num_onehots)
        entry = np.array(entry)
        num_pions = np.sum(entry)
        if num_pions > numoutputs:
            print("OH SHOOT, THIS EVENT HAD TWO PIONS OF A SINGLE TYPE")
            print("TRAIN IT AS IF ONE OF EACH TYPE FOR NOW...")
            onehot_ind = num_onehots - 1
        elif num_pions == 0:
           onehot_ind = 0 
        elif num_pions == numoutputs:
            onehot_ind = num_onehots-1
        else:  #Only one pion was produced of some type, and we must identify which
            onehot_mask = np.zeros(numoutputs)
            onehot_ind = 1
            onehot_mask[0] = 1
            num_rolls = 0
            have_match = False
            while not have_match:
                if np.sum(onehot_mask*entry)== np.sum(entry):
                    have_match = True
                elif num_rolls == numoutputs - 1:
                    onehot_ind +=1
                    onehot_mask[0] = 1
                    num_rolls = 0
                else:
                    onehot_ind +=1
                    onehot_mask = np.roll(onehot_mask,1)
                    num_rolls+=1
        this_onehotoutput[onehot_ind] = 1
        onehot_output.append(this_onehotoutput)
    return np.array(onehot_output)



            

if __name__=='__main__':
    input_data = open_numpy_allowpickle(numpy_infilename)
    output_data = open_numpy_allowpickle(numpy_outfilename)

    output_data = np.sum(output_data, axis=1)
    print("OUTPUT DATA: " + str(output_data))
    #Now, just set any entries with more than one pion as a one
    possible_multiring_ind = np.where(output_data > 0)[0]
    print("INDICES FOR EVENTS WITH POSSIBLE MULTIRING: " + str(possible_multiring_ind))
    output_data[possible_multiring_ind] = 1

    if DEBUG:
        print("TESTING: printing a single event's data out")
        print(input_data[0])
        print("single event's x-pixel = 0 row info")
        print(input_data[0][0])
        print("single event's x-pixel = 3, y-pixel = 3 row info")
        print(input_data[0][3][3])
        print("single event's outputs (piminuscount, pi0count, pipluscount)")
        print(output_data[0])
        #Let's make the first event's pixel map, summed over all the channels
        p.ShowChargeDistributions(input_data,output_data)
        p.ShowSingleEvent(input_data,0)
        p.ShowMeanMaps(input_data,output_data)
        print("SOME TOTAL DATASET DIAGNOSTICS:")
        print("TOTAL LENGTH OF DATASET:")
        print(len(output_data))
        print("EXAMPLE OUTPUTS OF HAS PION OR DOESNT HAVE PION")
        print(output_data[0:10])

    if ONEHOT_OUTPUTDATA:
        print("### OUTPUT DATA BEING CONVERTED TO ONE-HOT ENCODING ###")
        output_bin = OneHotEncodeBinaryArgs(output_data)
        print("COMPARISON OF OUTPUT DEFAULT TO ONEHOT:")
        print(output_data[41:80])
        print(output_bin[41:78])
        output_data = output_bin

    x_pixel_width = len(input_data[0])
    y_pixel_width = len(input_data[0][0])
    timewindow_width = len(input_data[0][0][0])

    print("#### SPLITTING INPUT DATA INTO TRAIN, VALIDATE, AND TEST FRACTIONS ####")
    x_train = input_data[0:NUM_TRAIN_EVENTS-1]
    x_val = input_data[NUM_TRAIN_EVENTS:(NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS-1)]
    x_test = input_data[NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS:]

    y_train = output_data[0:NUM_TRAIN_EVENTS-1]
    y_val = output_data[NUM_TRAIN_EVENTS:(NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS-1)]
    y_test = output_data[NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS:]

    output_width = 1

    model = None
    if TRAIN_MODEL:
        data_in = tensorflow.keras.layers.Input(shape=(x_pixel_width,y_pixel_width,timewindow_width))
        model = define_model_struct(data_in)
    if LOAD_MODEL:
        print("#### LOADING MODEL THAT HAS PREVIOUSLY BEEN TRAINED ######")
        modelfile = open(model_filename,"r")
        loaded_model_json = modelfile.read()
        modelfile.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_filename)
    if model is None:
        print("You need to either train a model or load a model!")
        sys.exit()

    model.compile(loss='mse', metrics=['mape'],optimizer=Adam())
   
    if TRAIN_MODEL:
        model.fit(x_train,y_train,epochs=50,validation_data=[x_val,y_val])
  
    if SAVE_MODEL:
        model_json = model.to_json()
        with open(model_filename,"w") as f:
            f.write(model_json)
        model.save_weights(weights_filename)


    print("#### BEGIN PLOTTING OF SOME VALIDATION CROSS-CHECKS ####")
    #First, let's build histograms of the predictions on test data
    predictions = model.predict(x_test)
    predictions = np.sum(predictions,axis=1)
    print("PREDICTIONS GIVEN TEST DATA: " + str(predictions))
    truths = y_test

    #Show Validation plots
    p.ShowRingValidationPlots(predictions,truths)

    #Print the first 5 predictions and tests
    print(model.predict(x_test[:5]),y_test[:5])
    print(model.evaluate(x_test))

    #Things to do next:
    #  - Somehow  There's a way in the x_predict to fuse in monte carlo techniques
    #    To help give you a spread on your neural network's efficiency and could
    #    Lead towards estimating uncertainties.  We should look into this.
    #  - One thing we could think about; we could add in a gaussian spread on the
    #    Time and Charge to re-shoot values for time and charge and re-train the
    #    neural network to see how much the efficiency changes based on these
    #    Fundamental factors.  This will basically quantify the systematic errors
    #    for our neural network.
