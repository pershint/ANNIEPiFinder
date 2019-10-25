import sys
import copy
import json
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json

#Custom plotting functions
import lib.Plots.CNNPlots as clp
import lib.NNArchs.CNN as car
import lib.OneHotEncoding as ohe
import lib.Normalization as nor

DEBUG = False
ONEHOT_OUTPUTDATA =False 

TRAIN_MODEL = True
SAVE_MODEL = True 
LOAD_MODEL = False
PLOTS = True

NUM_TRAIN_EVENTS = 12000
NUM_VALIDATE_EVENTS = 3000 
TARGETTYPE = "SINGLERING"  #MUPOS, PICOUNT, or SINGLERING

def open_numpy_allowpickle(thefile):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True,**k)
    alleventdata = np.load(thefile)
    np.load = np_load_old
    return alleventdata 

########## BEGIN MAIN PROGRAM ########
if __name__=='__main__':
    numpy_infilename = None
    numpy_outfilename = None
    weights_filename = None 
    model_filename = None
    
    if TARGETTYPE == "MUPOS":
        numpy_infilename = "./data/Processed/nparray_Data/LiliaComb_05072019_pixelmap_wMuTruth_input.npy"
        numpy_outfilename = "./data/Processed/nparray_Data/LiliaComb_05072019_pixelmap_wMuTruth_output.npy"
        weights_filename = "./model_out/mufitter_weights_0.h5"
        model_filename = "./model_out/mufitter_model_0.json"
    
    ##INPUT DATA AND TARGET DATA FOR PI COUNTING NN##
    if TARGETTYPE == "PICOUNT":
        numpy_infilename = "./data/Processed/nparray_Data/LiliaComb_05072019_pixelmap_input.npy"
        numpy_outfilename = "./data/Processed/nparray_Data/LiliaComb_05072019_pixelmap_output.npy"
        weights_filename = "./model_out/pifinder_weights_0.h5"
        model_filename = "./model_out/pifinder_model_0.json"
    
    #INPUT DATA AND TARGET DATA FOR SINGLE RING PREDICTION#
    if TARGETTYPE == "SINGLERING":
        numpy_infilename = "./data/Processed/nparray_Data/PMTVolume_06262019_pixelmap_input.npy"
        numpy_outfilename = "./data/Processed/nparray_Data/PMTVolume_06262019_pixelmap_output.npy"
        weights_filename = "./model_out/ringfinder_weights_0.h5"
        model_filename = "./model_out/ringfinder_model_0.json"

    input_data = open_numpy_allowpickle(numpy_infilename)
    output_data = open_numpy_allowpickle(numpy_outfilename)
    output_width = None

    ### Select targets specific to desired analysis ###
    if TARGETTYPE == "MUPOS":
        output_data_muinfo = output_data[0:len(output_data),3:9]
        output_data_muinfo = nor.NormalizePosDir(output_data_muinfo)
        output_data = output_data_muinfo
        output_width = len(output_data[0])
    elif TARGETTYPE == "PICOUNT":
        output_data_picount = output_data[0:len(output_data),0:3]
        output_width = len(output_data[0])
    elif TARGETTYPE == "SINGLERING":
        output_data_picount = output_data[0:len(output_data),0:3]
        output_data_pitotal = np.sum(output_data_picount, axis=1)
        possible_multiring_ind = np.where(output_data_pitotal > 0)[0]
        print("INDICES FOR EVENTS WITH POSSIBLE MULTIRING: " + str(possible_multiring_ind))
        output_data_pitotal[possible_multiring_ind] = 1
        output_data = output_data_pitotal
        output_width = 1
    else:
        print("Selected target type not recognized.  Exitting")
        sys.exit(0)
   
    if ONEHOT_OUTPUTDATA:
        print("### TARGETS BEING CONVERTED TO ONE-HOT ENCODING ###")
        output_bin = ohe.OneHotEncodeBinaryArgs(output_data)
        if DEBUG:
            print("COMPARISON OF TARGETS TO ONEHOT TARGETS FOR 10 EVENTS:")
            print(output_data[41:51])
            print(output_bin[41:51])
        output_data = output_bin


    print("OUTPUT_WIDTH: " + str(output_width))
    
    if DEBUG:  #Print some examples of input data and targets
        print("TESTING: printing a single event's data out")
        print(input_data[0])
        print("single event's x-pixel = 0 row info")
        print(input_data[0][0])
        print("single event's x-pixel = 3, y-pixel = 3 row info")
        print(input_data[0][3][3])
        print("Targets for first five entries")
        print(output_data[0:5])
        #Let's make the first event's pixel map, summed over all the channels
        p.ShowChargeDistributions(input_data,output_data)
        p.ShowSingleEvent(input_data,0)
        p.ShowMeanMaps(input_data,output_data)
        print("SOME TOTAL DATASET DIAGNOSTICS:")
        print("TOTAL LENGTH OF DATASET:")
        print(len(output_data))
        print("TOTAL PI COUNTS IN DATASET [total_pi0, total_pi-]:")
        print(np.sum(output_data,axis=0))

    x_pixel_width = len(input_data[0])
    y_pixel_width = len(input_data[0][0])
    timewindow_width = len(input_data[0][0][0])
    print("TIME WINDOW WIDTH: " + str(timewindow_width))

    print("#### SPLITTING INPUT DATA INTO TRAIN, VALIDATE, AND TEST FRACTIONS ####")
    x_train = input_data[0:NUM_TRAIN_EVENTS-1]
    x_val = input_data[NUM_TRAIN_EVENTS:(NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS-1)]
    x_test = input_data[NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS:]

    y_train = copy.deepcopy(output_data[0:NUM_TRAIN_EVENTS-1])
    y_val = copy.deepcopy(output_data[NUM_TRAIN_EVENTS:(NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS-1)])
    y_test = copy.deepcopy(output_data[NUM_TRAIN_EVENTS+NUM_VALIDATE_EVENTS:len(output_data)])

    model = None
    if TRAIN_MODEL:
        data_in = tensorflow.keras.layers.Input(shape=(x_pixel_width,y_pixel_width,timewindow_width))
        model = car.define_model_struct(data_in,output_width)
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
    truths = y_test

    if TARGETTYPE == "MUPOS":
        predictions = Nor.ReverseNormalizePosDir(predictions)
        truths = nor.ReverseNormalizePosDir(y_test)
        #Show Validation plots
        clp.ShowRecoValidationPlots(predictions,truths)
    if TARGETTYPE == "PICOUNT":
        clp.ShowPionValidationPlots(predictions,truths)
    if TARGETTYPE == "SINGLERING":
        predictions = np.sum(predictions,axis=1)
        clp.ShowRingValidationPlots(predictions,truths)

    #Print the first 5 predictions and tests
    print(model.predict(x_test[:5]),y_test[:5])
    print(model.evaluate(x_test))
