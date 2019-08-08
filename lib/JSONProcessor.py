import numpy as np
import json

class JSONProcessor(object):
    '''
    This class is used to prepare JSON-format ANNIE data for loading into
    a neural network.  Specifically, charge information for PMTs is split
    into multiple channels, each channel associated with a time window.
    '''

    def __init__(self,jsondata=None):
        self.jsondata = jsondata
        self.processed_data_header = None
        self.processed_data = None
        self.num_detectors = 151

    def setNumDetectors(self,numdet):
        self.num_detectors = numdet

    def clearProcessedData(self):
        self.processed_data_header = None
        self.processed_data = None

    def loadJSON(self,jsondata):
        '''
        loads the dictionary object as this class' JSON data
        to process..
        Inputs:
            jsondata [dict]
            JSON object to load.
        '''
        self.jsondata = jsondata

    def loadJSON_fromfilepath(self,jsonfilepath):
        '''
        Load JSON data that will be processed into a text
        file.
        Inputs:
            jsonfilepath [string]
            Path to JSON file to load.
        '''
        self.jsondatapath = jsonfilepath
        with open(jsonfilepath,"r") as f:
            self.jsondata = json.load(f)

    def processData(self):
        '''
        This method loops through all events in the jsondata and
        processes PMT/LAPPD hit information for use in a multichannel
        feed-forward neural network.
        Inputs:
        '''
        all_data = []
        for j,eventhittimes in enumerate(self.jsondata["digitT"]):
            thisevent_data = []
            print("PROCESSING EVENT %i\n"%(j))
            eventhittimes = np.array(eventhittimes)
            event_hitIDdata = np.array(self.jsondata["digitDetID"][j])
            event_hitqdata = np.array(self.jsondata["digitQ"][j])
            event_hittdata = np.array(self.jsondata["digitT"][j])
            all_IDs = np.arange(0,self.num_detectors,1)
            hitPMT_IDs = np.array(self.jsondata["digitDetID"][j])
            nohit_IDs = np.setdiff1d(np.union1d(all_IDs,hitPMT_IDs),
                                         np.intersect1d(all_IDs,hitPMT_IDs))
            #Append data for hits that are not in window
            for ID in nohit_IDs:
                np.append(event_hitIDdata,ID)
                np.append(event_hittdata,0)
                np.append(event_hitqdata,0)
            #Now, sort times and charges based on PMT ID
            sorted_times = np.array([x for _,x in sorted(zip(event_hitIDdata,event_hittdata))])
            sorted_charges = np.array([x for _,x in sorted(zip(event_hitIDdata,event_hitqdata))])
            thisevent_data.append(sorted_times)
            thisevent_data.append(sorted_charges)
            # OUTPUT VALUES: Append number of pions to event data
            picounts = []
            picounts.append(self.jsondata["PiMinusCount"][j])
            picounts.append(self.jsondata["Pi0Count"][j])
            picounts.append(self.jsondata["PiPlusCount"][j])
            thisevent_data.append(np.array(picounts))
            all_data.append(np.array(thisevent_data))
        return np.array(all_data)

