import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import json
import numpy as np
import tensorflow.keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU,Flatten
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


    x_train = input_data[0:5]
    x_val = input_data[5:8]
    x_test = input_data[9:10]

    y_train = input_data[0:5]
    y_val = input_data[5:10]

    output_width = len(output_data[0])
    
    data_in = tensorflow.keras.layers.Input(shape=(x_pixel_width,y_pixel_width,timewindow_width))
    
    x = Conv2D(32,3)(data_in)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(32,3, strides=2)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3)(data_in)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Conv2D(64,3, strides=2)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = ReLU()(x)
    x = BatchNormalization(momentum=.99)(x)
    x = Dense(output_width)(x)
    x = ReLU()(x)
    
    model = Model(inputs=data_in,outputs=x)
    model.compile(loss='binary_crossentropy',optimizer=Adam())
    
    model.fit(x_train,y_train,epochs=10,validation_data=[x_val,y_val])
    
    model.predict(x_test)
    

