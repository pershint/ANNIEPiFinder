import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import json
import numpy as np
import keras
from keras.layers import Dense
from keras.models import Model

numpy_filename = "./data/nparray_Data/LiliaComb_05072019.npy"

def open_numpy_allowpickle(thefile):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True,**k)
    alleventdata = np.load(thefile)
    np.load = np_load_old
    return alleventdata 

if __name__=='__main__':
    alleventdata = open_numpy_allowpickle(numpy_filename)
    print("TESTING: printing a single event's data out")
    print(alleventdata[0])
    print("single event's time data")
    print(alleventdata[0][0])
    print("single event's charge data")
    print(alleventdata[0][1])
    print("single event's outputs (piminuscount, pi0count, pipluscount)")
    print(alleventdata[0][2])

    ##Now, we want to give these data as inputs/outputs to neural networks
    ##We'll follow an example on how to compine input from several models just
    ##To try something out.
    #inputTime = keras.Input(shape=(len(alleventdata[0][0]),))
    #inputCharge = keras.Input(shape=(len(alleventdata[0][1]),))

    ##First branch operates on the time values
    #t = Dense(64, activation="relu")(inputTime)
    #t = Dense(32, activation="relu")(t)
    #t = Dense(16,activation="relu")(t)
    #t = Model(inputs=inputTime, outputs=t)

    ##Second branch operates on the charge values
    #q = Dense(64, activation="relu")(inputCharge)
    #q = Dense(32, activation="relu")(q)
    #q = Dense(16,activation="relu")(q)
    #q = Model(inputs=inputTime, outputs=q)

    ## we combine the output from these two dense neural networks to form
    ## The inputs for our combined network
    #inputCombo = np.concatenate([t.output,q.output])

    #final = Dense(16, activation="relu")(inputCombo)
    #final = Dense(8, activation="relu")(final)
    #final = Dense(3, activation="linear")(final)

    #model = Model(inputs=[t.input,q.input], outputs=final)
