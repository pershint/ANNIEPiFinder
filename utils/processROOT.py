import lib.ROOTProcessor as rp
import lib.JSONProcessor as jp
import lib.PixelMapper as pm
import json
import pandas as pd
import numpy as np

########################################################################
#Processor for taking ANNIE PhaseII ntuples and converting the PMT time/charge
#data and pion counts into a numpy array object.  In an intermediate step, the
#ROOT data is converted to a JSON file, which can also be saved if desired.
########################################################################


MYBRANCHES = ['digitX','digitY','digitZ','digitT','digitQ','Pi0Count','PiPlusCount','PiMinusCount','digitDetID','digitType']
SAVE_JSON=False
INPUTFILE = "./data/ROOT_Data/LiliaComb_05072019.root"
OUTBASE = "./data/LiliaComb_05072019_pixelmap"
SAVE_NUMPYBIN=True

def process_ROOTFile(infilename,data_branches,output_filenamebase,data_branch="phaseII",save_json=False, save_numpy=True):
    #Test processing data
    myROOTProcessor = rp.ROOTProcessor(treename=data_branch)
    #branch = ['Pi0Count']
    myROOTProcessor.processROOTFile(infilename,branches_to_get=data_branches)
    data_injson = myROOTProcessor.getProcessedData()
    if SAVE_JSON:
        with open(output_filenamebase+".json","w") as f:
            #procd_data_lists = myProcessor.removeNumpyArrays()
            #json.dump(procd_data_lists,f)
            json.dump(data_injson,f)
    
    #Generate a pixel map from all PMTs hit in events in this file
    #FIXME: We could also just have the pixel map stored in a JSON file, so
    #We don't have to generate the map over and over again (plus, if a PMT was
    #Never hit in the file, then it would be left out of the map
    myPixelMapper = pm.PixelMapper(jsondata = data_injson)
    pixel_map, numxpixels, numypixels = myPixelMapper.MapPositionsToPixels(pmt_only = True, ycut = 130)
    myPixelMapper.PlotPixelMap(pixel_map)

    #Process the JSON data into our correct numpy data format
    myJSONProcessor = jp.JSONProcessor()
    myJSONProcessor.loadJSON(data_injson)
    myJSONProcessor.loadPixelMap(pixel_map,numxpixels,numypixels)
    input_data,output_data = myJSONProcessor.processData(timewindowmin=0, timewindowmax=20,
                                               numwindows=5, maxevents=10)
    print("TESTING: printing a single event's data out")
    print(input_data[0])
    print("single event's data in the x-pixel row")
    print(input_data[0][0])
    print("single event's charge data at x-pixel=0, y-pixel=3")
    print(input_data[0][0][3])
    if SAVE_NUMPYBIN:
        np.save(output_filenamebase+"_input.npy",input_data)
        np.save(output_filenamebase+"_output.npy",output_data)


if __name__=='__main__':
    print("######## ANNIE ROOT Processor #######")
    process_ROOTFile(INPUTFILE, MYBRANCHES, OUTBASE,save_json=SAVE_JSON, save_numpy=SAVE_NUMPYBIN)
