import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import json
import keras
import numpy as np

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
