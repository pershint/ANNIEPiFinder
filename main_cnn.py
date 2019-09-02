import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import json
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense,Activation, Conv2D, BatchNormalization, ReLU,Flatten, Dropout, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

numpy_infilename = "./data/LiliaComb_05072019_pixelmap_input.npy"
numpy_outfilename = "./data/LiliaComb_05072019_pixelmap_output.npy"

def open_numpy_allowpickle(thefile):
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a,allow_pickle=True,**k)
    alleventdata = np.load(thefile)
    np.load = np_load_old
    return alleventdata 

if __name__=='__main__':
    input_data = open_numpy_allowpickle(numpy_infilename)
    output_data = open_numpy_allowpickle(numpy_outfilename)
    print("TESTING: printing a single event's data out")
    print(input_data[0])
    print("single event's x-pixel = 0 row info")
    print(input_data[0][0])
    print("single event's x-pixel = 3, y-pixel = 3 row info")
    print(input_data[0][3][3])
    print("single event's outputs (piminuscount, pi0count, pipluscount)")
    print(output_data[0])

    x_pixel_width = len(input_data[0])
    y_pixel_width = len(input_data[0][0])
    timewindow_width = len(input_data[0][0][0])


    x_train = input_data[0:1999]
    x_val = input_data[2000:2499]
    x_test = input_data[2500:]

    y_train = output_data[0:1999]
    y_val = output_data[2000:2499]
    y_test = output_data[2500:]

    output_width = len(output_data[0])
    
    data_in = tensorflow.keras.layers.Input(shape=(x_pixel_width,y_pixel_width,timewindow_width))
    
    xp = Conv2D(32,3)(data_in)
    x = ReLU()(xp)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(32,3, strides=2)(x)
    x = Dropout(.1)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3)(data_in)
   # xp = Add()([x,Conv2D(64,3, strides=2)(xp)])
    x = Dropout(.1)(x)    
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3, strides=2)(x)
    x = Dropout(.1)(x)    
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
  #  x = Conv2D(128,3)(x)
    #x = Add()([x,Conv2D(128,3,strides=2)(xp)])
    x = Dropout(.1)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Dense(output_width)(x)
    x = Activation('sigmoid')(x)
    
    model = Model(inputs=data_in,outputs=x)
    model.compile(loss='mse', metrics=['mape'],optimizer=Adam())
    
    model.fit(x_train,y_train,epochs=50,validation_data=[x_val,y_val])
    
    print(model.predict(x_test[:5]),y_test[:5])
    
    print(model.evaluate(x_test))

    #Things to do next:
    #  - Look at each class of event; see what the fraction of [0,0,0] is to the others.
    #    If there's basically no [0 1 0] for example, let's ignore it for now as
    #    it'll be basically impossible to train on.
    #  - Let's look at the output from model.predict for all events in x_test and
    #    Make distributions of the outputs.  We can compare the difference of
    #    the outputs to the truth information for events with no pions, and
    #    Those with pions.  This would also help us identify what number we
    #    Would want to be the cut in our neural network output that identifies 
    #    A pion or no pion event.
    #  - Somehow  There's a way in the x_predict to fuse in monte carlo techniques
    #    To help give you a spread on your neural network's efficiency and could
    #    Lead towards estimating uncertainties.  We should look into this.
    #  - One thing we could think about; we could add in a gaussian spread on the
    #    Time and Charge to re-shoot values for time and charge and re-train the
    #    neural network to see how much the efficiency changes based on these
    #    Fundamental factors.  This will basically quantify the systematic errors
    #    for our neural network.
