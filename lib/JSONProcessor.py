import numpy as np
import json

class JSONProcessor(object):
    '''
    This class is used to prepare JSON-format ANNIE data for loading into
    a convolutional neural network.  Specifically, charge information for PMTs is split
    into multiple channels, each channel associated with a time window.
    '''

    def __init__(self,jsondata=None):
        self.jsondata = jsondata
        self.processed_data_header = None
        self.processed_data = None
        self.num_analysis_detectors = None
        self.idpixmap = None
        self.numxpixels = None
        self.numypixels = None

    def loadPixelMap(self,pixelmap, numxpixels, numypixels):
        self.idpixmap = pixelmap
        self.numxpixels = numxpixels
        self.numypixels = numypixels
        self.num_analysis_detectors = len(self.idpixmap)

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

    def processData(self,timewindowmin=0.0, timewindowmax=20.0,
            numwindows=5,maxevents=100):
        '''
        This method loops through all events in the jsondata and
        processes PMT hit information for use in a convolutional neural network.
        Inputs:
            timewindowmin float
            Gives the minimum time for which hits are to be considered/included
            in the event data.

            timewindowmax float
            Gives the maximum time for which hits are to be considered/included
            in the event data.

            numwindows int
            Number of channels that will be given to the neural network.  Charge
            information is split between the time windows.

        '''
        if timewindowmax < timewindowmin:
            print("ERROR: choose a max time window that comes after the min time.")
            return
        all_analysis_IDs = self.idpixmap["id"].values #Detector IDs to be used in CNN
        print("KEYS FROM PIXEL MAP: " + str(all_analysis_IDs))
        input_data = []
        output_data = []

        #BEGIN EVENT LOOP
        for j,event_hitIDdata in enumerate(self.jsondata["digitDetID"]):
            if j>maxevents:
                break
            print("PROCESSING EVENT %i\n"%(j))

            ##INPUT DATA##

            #Get this event's detector id, time, and charge info. for all hits.
            #Remove detector IDs that are not in the analysis pixel map.
            analysis_dets = []
            for l,DetID in enumerate(event_hitIDdata):
                if DetID in self.idpixmap["id"]:
                    analysis_dets.append(l)
            analysis_dets = np.array(analysis_dets)
            event_hitIDdata = np.array(event_hitIDdata)[analysis_dets]
            event_hittdata = np.array(self.jsondata["digitT"][j])[analysis_dets]
            event_hitqdata = np.array(self.jsondata["digitQ"][j])[analysis_dets]

            #For any PMTs that were not hit, assign time/charge zero
            nohit_IDs = np.setdiff1d(all_analysis_IDs,event_hitIDdata)
            for ID in nohit_IDs:
                event_hitIDdata = np.append(event_hitIDdata,ID)
                event_hittdata = np.append(event_hittdata,0)
                event_hitqdata = np.append(event_hitqdata,0)

            #Sort from lowest Detector ID to largest
            sorted_times = np.array([x for _,x in sorted(zip(event_hitIDdata,event_hittdata))])
            sorted_charges = np.array([x for _,x in sorted(zip(event_hitIDdata,event_hitqdata))])
            sorted_IDs = sorted(event_hitIDdata)

            #Initialize the pixel map where time/charge info will be placed
            thisevent_data = [None]*self.numxpixels
            for xrow in range(len(thisevent_data)):
                thisevent_data[xrow] = [None]*self.numypixels
            
            #Form the charge/time window array for this detector
            for k,ID in enumerate(sorted_IDs):
                print("PROCESSING PMT ID " + str(ID))
                xpixel = self.idpixmap.loc[self.idpixmap["id"] == ID,"xpixel"].iloc[0]
                ypixel = self.idpixmap.loc[self.idpixmap["id"] == ID,"ypixel"].iloc[0]
                print("XPIXEL: %i"%(xpixel))
                print("YPIXEL: %i"%(ypixel))
                tw_width = (timewindowmax-timewindowmin)/float(numwindows)
                thishit_data = np.zeros(numwindows)
                hit_time = sorted_times[k]
                hit_charge = sorted_charges[k]
                thiswinmin = timewindowmin
                for h in range(numwindows):
                    thiswinmax = thiswinmin + tw_width
                    if hit_time > thiswinmin and hit_time < thiswinmax:
                        thishit_data[h] = hit_charge
                    thiswinmin = thiswinmax
                print("THIS PMT'S HIT DATA: " + str(thishit_data))
                thisevent_data[xpixel][ypixel] = thishit_data
            
            ## OUTPUT DATA ##
            picounts = []
            picounts.append(self.jsondata["PiMinusCount"][j])
            picounts.append(self.jsondata["Pi0Count"][j])
            picounts.append(self.jsondata["PiPlusCount"][j])

            #Append this event's input/output data to the final arrays
            input_data.append(np.array(thisevent_data))
            output_data.append(np.array(picounts))
        return np.array(input_data), output_data

