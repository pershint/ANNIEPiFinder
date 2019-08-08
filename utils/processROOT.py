import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import json
import numpy as np

########################################################################
#Processor for taking ANNIE PhaseII ntuples and converting the PMT time/charge
#data and pion counts into a numpy array object.  In an intermediate step, the
#ROOT data is converted to a JSON file, which can also be saved if desired.
########################################################################


mybranches = ['digitX','digitY','digitZ','digitT','digitQ','Pi0Count','PiPlusCount','PiMinusCount','digitDetID','digitType']
SAVE_JSON=False
json_filename = "./data/JSON_Data/LiliaComb_05072019.json"
SAVE_NUMPYBIN=True
numpy_filename = "./data/nparray_Data/LiliaComb_05072019.npy"


if __name__=='__main__':
    print("######## ANNIE ROOT Processor #######")

    #Test processing data
    myROOTProcessor = rp.ROOTProcessor(treename="phaseII")
    #branch = ['Pi0Count']
    myROOTProcessor.processROOTFile("./data/ROOT_Data/LiliaComb_05072019.root",branches_to_get=mybranches)
    data_injson = myROOTProcessor.getProcessedData()
    if SAVE_JSON:
        with open(json_filename,"w") as f:
            #procd_data_lists = myProcessor.removeNumpyArrays()
            #json.dump(procd_data_lists,f)
            json.dump(data_injson,f)
    myJSONProcessor = jp.JSONProcessor()
    myJSONProcessor.loadJSON(data_injson)
    myJSONProcessor.setNumDetectors(151)
    alleventdata = myJSONProcessor.processData()
    print("TESTING: printing a single event's data out")
    print(alleventdata[0])
    print("single event's time data")
    print(alleventdata[0][0])
    print("single event's charge data")
    print(alleventdata[0][1])
    print("single event's outputs (piminuscount, pi0count, pipluscount)")
    print(alleventdata[0][2])
    if SAVE_NUMPYBIN:
        np.save(numpy_filename,alleventdata)
