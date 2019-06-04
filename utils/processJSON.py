import lib.JSONProcessor as jp
import json

if __name__=='__main__':
    print("######## ANNIE ROOT Processor #######")

    #Test processing data
    myProcessor = jp.JSONProcessor(jsondatapath="./data/JSON_Data/LiliaFiles_05072019.json")
    myProcessor.processData(timewindowmin=0.0, timewindowmax=20.0,
            numwindows=5,numdetectors=150)
