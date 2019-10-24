from tensorflow.keras.layers import Dense,Activation, Conv2D, BatchNormalization, ReLU,Flatten, Dropout, Add
from tensorflow.keras.models import Model, model_from_json

def define_model_struct(data_in,output_width):
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


