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

sns.set_context('poster')
sns.set(font_scale=3.0)

DEBUG = False
TRAIN_MODEL = False
SAVE_MODEL = False
LOAD_MODEL = True

ONEHOT_OUTPUTDATA = True
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

def ShowMeanMaps(in_data,out_data):
    nopion_mean = {'xpixel':[], 'ypixel':[], 'channel_avg':[], 'channel_stdev':[]}
    pion_mean = {'xpixel':[], 'ypixel':[], 'channel_avg':[], 'channel_stdev':[]}
     
    nopion_inds = np.where(np.sum(output_data,axis=1) == 0)[0]
    pion_inds = np.where(np.sum(output_data,axis=1) > 0)[0]
    for xpixel in range(len(in_data[0])):
        for ypixel in range(len(in_data[0][xpixel])):
            pixel_chans = len(in_data[0][xpixel][ypixel])
            nopion_mean['xpixel'].append(xpixel)
            nopion_mean['ypixel'].append(ypixel)
            pion_mean['xpixel'].append(xpixel)
            pion_mean['ypixel'].append(ypixel)
            nopion_channel_vals = input_data[nopion_inds,xpixel,ypixel,0:pixel_chans]
            nopion_channel_sums = np.sum(nopion_channel_vals,axis=1)
            nopion_channel_average = np.sum(nopion_channel_sums)/len(nopion_channel_sums)
            nopion_channel_stdev = np.std(nopion_channel_sums)
            pion_channel_vals = input_data[pion_inds,xpixel,ypixel,0:pixel_chans]
            pion_channel_sums = np.sum(pion_channel_vals,axis=1)
            pion_channel_average = np.sum(pion_channel_sums)/len(pion_channel_sums)
            pion_channel_stdev = np.std(pion_channel_sums)
            nopion_mean['channel_avg'].append(nopion_channel_average)
            pion_mean['channel_avg'].append(pion_channel_average)
            nopion_mean['channel_stdev'].append(nopion_channel_stdev)
            pion_mean['channel_stdev'].append(pion_channel_stdev)
    diff_mean = copy.deepcopy(pion_mean)
    diff_mean["channel_avg"] = np.array(pion_mean["channel_avg"]) - np.array(nopion_mean["channel_avg"])
    diff_mean["channel_stdev"] = np.sqrt(np.array(pion_mean["channel_stdev"])**2 + np.array(nopion_mean["channel_stdev"])**2)
    nopion_mean = pd.DataFrame(nopion_mean)
    pion_mean = pd.DataFrame(pion_mean)
    diff_mean = pd.DataFrame(diff_mean)
    
    nopm = nopion_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    pm = pion_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    diffm = diff_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    nops = nopion_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    ps = pion_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    diffs = diff_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    sns.heatmap(nopm)
    plt.title("Average charge distribution of events with no pion (channels summed)")
    plt.show()
    sns.heatmap(nops)
    plt.title("Std. dev. of charge distributions in events with no pion (channels summed)")
    plt.show()
    sns.heatmap(pm)
    plt.title("Average charge distribution of events with a pion (channels summed)")
    plt.show()
    sns.heatmap(ps)
    plt.title("Std. dev. of charge distributions in events with a pion (channels summed)")
    plt.show()
    sns.heatmap(diffm)
    plt.title("Pion - No Pion average charge distribution (channels summed)")
    plt.show()
    sns.heatmap(diffs)
    plt.title("Pion - No Pion errors Std. Devs. added in quadrature (channels summed)")
    plt.show()

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
        #Let's make the first event's pixel map, summed over all the channels
        single_event = {'xpixel':[], 'ypixel':[], 'channel_sum':[]}
        for xpixel,val in enumerate(input_data[0]):
            for ypixel,pixel_chans in enumerate(val):
                single_event['xpixel'].append(xpixel)
                single_event['ypixel'].append(ypixel)
                single_event['channel_sum'].append(np.sum(val[ypixel]))
        single_event = pd.DataFrame(single_event)
        se = single_event.pivot(index='ypixel',columns='xpixel',values='channel_sum')
        sns.heatmap(se)
        plt.title("Example event (channels summed)")
        plt.show()
        ShowMeanMaps(input_data,output_data)
        print("SOME TOTAL DATASET DIAGNOSTICS:")
        print("TOTAL LENGTH OF DATASET:")
        print(len(output_data))
        print("TOTAL PI COUNTS IN DATASET [total_pi0, total_pi-]:")
        print(np.sum(output_data,axis=0))

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
    pidiffs = {"pi0_diff": diffs[0:len(diffs),1],"piminus_diff": diffs[0:len(diffs),2], 
            "pi0_truth": truths[0:len(truths),1],"piminus_truth": truths[0:len(truths),2],
            "pi0_predict": predictions[0:len(predictions),1],
            "piminus_predict": predictions[0:len(predictions),2]}
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
