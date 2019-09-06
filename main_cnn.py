import sys
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

sns.set_context('poster')
sns.set(font_scale=3.0)

DEBUG = True
TRAIN_MODEL = False
SAVE_MODEL = False
LOAD_MODEL = True

NUM_TRAIN_EVENTS = 2000
NUM_VALIDATE_EVENTS = 500

numpy_infilename = "./data/LiliaComb_05072019_pixelmap_input.npy"
numpy_outfilename = "./data/LiliaComb_05072019_pixelmap_output.npy"
weights_filename = "./model_out/pifinder_weights_0.h5"
model_filename = "./model_out/pifinder_model_0.json"

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

if __name__=='__main__':
    input_data = open_numpy_allowpickle(numpy_infilename)
    output_data = open_numpy_allowpickle(numpy_outfilename)

    print("#### IGNORING PI+ FOR NOW... THERE WAS 1 in 4000 MC EVENTS ####")
    output_data = output_data[0:len(output_data),1:3]
    if DEBUG:
        print("TESTING: printing a single event's data out")
        print(input_data[0])
        print("single event's x-pixel = 0 row info")
        print(input_data[0][0])
        print("single event's x-pixel = 3, y-pixel = 3 row info")
        print(input_data[0][3][3])
        print("single event's outputs (piminuscount, pi0count, pipluscount)")
        print(output_data[0])
        print("SOME TOTAL DATASET DIAGNOSTICS:")
        print("TOTAL LENGTH OF DATASET:")
        print(len(output_data))
        print("TOTAL PI COUNTS IN DATASET [total_pi0, total_pi-]:")
        print(np.sum(output_data,axis=0))


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

    output_width = len(output_data[0])

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
    truths = y_test
    diffs = predictions-truths
    #pidiffs = {"piplus_diff": diffs[0:len(diffs),0], "pi0_diff": diffs[0:len(diffs),1],
    #        "piminus_diff": diffs[0:len(diffs),2], "piplus_truth": truths[0:len(truths),0],
    #        "pi0_truth": truths[0:len(truths),1],"piminus_truth": truths[0:len(truths),2]}
    pidiffs = {"pi0_diff": diffs[0:len(diffs),0],"piminus_diff": diffs[0:len(diffs),1], 
            "pi0_truth": truths[0:len(truths),0],"piminus_truth": truths[0:len(truths),1],
            "pi0_predict": predictions[0:len(predictions),0],
            "piminus_predict": predictions[0:len(predictions),1]}
    pidiffs = pd.DataFrame(pidiffs)

   
    #Let's first get indices of entries with no pions
    nopion_ind = np.where(pidiffs["pi0_truth"] + pidiffs["piminus_truth"] == 0)[0]
    
    pion_ind = np.where(pidiffs["pi0_truth"] + pidiffs["piminus_truth"] > 0)[0]
    piminus_ind = np.where(pidiffs["piminus_truth"] > 0)[0]

    pi0_count = pidiffs["pi0_truth"]
    pi0_ind = np.where(pi0_count > 0)[0]

    total_pioncountdiff = pidiffs["piminus_diff"] + pidiffs["pi0_diff"]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(pidiffs["pi0_predict"][nopion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{0}$ prediction', histtype="step", color='red')
    plt.hist(pidiffs["piminus_predict"][nopion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{-}$ prediction', histtype="step", color='blue')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction of pion presence with no pions produced")
    plt.show()

    #Let's first get indices of entries with pions

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(pidiffs["pi0_predict"][pion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{0}$ prediction', histtype="step", color='red')
    plt.hist(pidiffs["piminus_predict"][pion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{-}$ prediction', histtype="step", color='blue')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction of pion presence for events with a pion")
    plt.show()

    predicts_wpions = pd.DataFrame({"pi0p":pidiffs["pi0_predict"][pion_ind],
        "pimp":pidiffs["piminus_predict"][pion_ind]})
    fig = plt.figure()
    g = sns.jointplot(x="pi0p", y="pimp", data=predicts_wpions,kind="hex",stat_func=None)
    g = g.set_axis_labels("$\pi^{0}$ count prediction", "$\pi^{-}$ count prediction")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.9,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("Distribution of $\pi^{0}$ and $\pi^{-}$ predictions for events with a pion")
    plt.show()

    plt.hist(pidiffs["pi0_diff"][pi0_ind],bins=30,color='red',alpha=0.8)
    plt.title("Prediction-Truth for count of $\pi^{0}$ in events with $\pi^{0}$")
    plt.show()

    plt.hist(pidiffs["piminus_diff"][piminus_ind],bins=30,color='green',alpha=0.8)
    plt.title("Prediction-Truth for count of $\pi^{-}$ in events with $\pi^{-}$")
    plt.show()

    plt.hist(total_pioncountdiff[nopion_ind],bins=30,color='purple',alpha=0.8)
    plt.title("Sum of $\pi^{0}$ and $\pi^{-}$ Prediction-Truth for no-$\pi$ events")
    plt.show()


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
