import pandas
import keras
import lib.ROOTProcessor as rp
import json

if __name__=='__main__':
    print("######## ANNIE ROOT Processor #######")

    #Test processing data
    myProcessor = rp.ROOTProcessor(treename="phaseII")
    mybranches = ['digitX','digitY','digitZ','digitT','digitQ','Pi0Count','PiPlusCount','PiMinusCount','digitDetID','digitType']
    #branch = ['Pi0Count']
    myProcessor.addROOTFile("./data/ROOT_Data/LiliaComb_05072019.root",branches_to_get=mybranches)
    with open("./data/LiliaFiles_05072019.json","w") as f:
        #procd_data_lists = myProcessor.removeNumpyArrays()
        #json.dump(procd_data_lists,f)
        json.dump(myProcessor.getProcessedData(),f)
