import numpy as np
import json

class JSONProcessor(object):
    '''
    This class is used to prepare JSON-format ANNIE data for loading into
    a neural network.  Specifically, charge information for PMTs is split
    into multiple channels, each channel associated with a time window.
    '''

    def __init__(self,jsondatapath=None):
        self.jsondata = None
        if jsondatapath is not None:
            self.jsondatapath = jsondatapath
            with open(jsondatapath,"r") as f:
                self.jsondata = json.load(f)
        self.processed_data_header = None
        self.processed_data = None
        self.procfilepath = "./default_processed.txt"

    def clearProcessedData(self):
        self.processed_data_header = None
        self.processed_data = None

    def setFilePath(self,filepath):
        '''
        Set where to write the processed text file to.
        Input:
           filepath [string]
        '''
        self.procfilepath = filepath

    def loadJSON(self,jsonfilepath):
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
            numwindows=5,numdetectors=150,digittypes="PMT"):
        '''
        This method loops through all events in the jsondata and
        processes PMT/LAPPD hit information for use in a multichannel
        feed-forward neural network.
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

            numchannels int
            Maximum number of detectors to use in neural network. TODO: Get this
            information from the datafile itself?

            digittypes string
            either "PMT","LAPPD", or "all".
        '''
        if timewindowmax < timewindowmin:
            print("ERROR: choose a max time window that comes after the min time.")
            return
        self._appendHeaderToFile(timewindowmin, timewindowmax,
                                 numwindows,numdetectors,digittypes)
        for j,eventhittimes in enumerate(self.jsondata["digitT"]):
            inwindowindices = np.where(eventhittimes > timewindowmin &&
                                       eventhittimes < timewindowmax)[0]
            detector_data = self._getInWindowChannels(j,inwindowindices)
            all_IDs = np.arange(0,numdetectors,1)
            inwindow_IDs = self.jsondata["digitDetID"][j]
            nonwindow_IDs = np.setdiff1d(np.union1d(all_IDs,inwindow_IDs),
                                         np.intersect1d(all_IDs,inwindow_IDs))
            for ID in nonwindow_IDs:
                emptyhit = np.zeros(0,numwindows+1)
                emptyhit[0] = ID
                detector_data.append(emptyhit)
            self._appendToFile(j,detector_data)
        print("PROCESSING OF DATA COMPLETE")

    def _appendHeaderToFile(self, timewindowmin, timewindowmax,
            numwindows,numdetectors,digittypes):
        '''
        Open the filepath that is being processed and append to it the
        specifications used to process data and a key line indicating
        what each data entry in an event represents.
        Inputs:
            see description on self.processData.
        '''
        with open(self.procfilepath,"a") as f:
            f.write("timewindowmin, timewindowmax, numwindows, numdetectors, digittypes\n")
            f.write("%f,%f,%i,%i,%s\n\n"%(timewindowmin,timewindowmax,numwindows,
                numdetectors,digittypes)
            f.write("KEY_LINE\n")
            f.write("ID")
            thiswinmin = timewindowmin
            for j in xrange(numwindows):
                thiswinmax = thiswinmin + tw_width
                f.write(",%f"%(thiswinmax))
            f.write("\n\n")


    def _appendEventToFile(self,eventnum, detector_data):
        '''
        Open the filepath that is being processed and append to it with
        the new eventnumber and detector_data.
        Inputs:
            eventnum [int]
            Event number associated with the array of processed data.

            detector_data [array]
            Array of arrays, where each array is the charge of a PMT
            in each hit window.
        '''
        with open(self.procfilepath,"a") as f:
            f.write("EVENT %i"%(eventnum))
            for j,hit in detector_data:
                hitline = str(detector_data[0])
                for k in np.arange(1,len(hit),1):
                    f.write(",%s"%(str(detector_data[k])))
                f.write("\n")
            f.write("\n")
        
    
    def _getInWindowChannels(self,eventnum,inwindowindices,numwindows,
            timewindowmin, timewindowmax):
        tw_width = (timewindowmax-timewindowmin)/float(numwindows)
        event_hitIDdata = self.jsondata["digitDetID"][eventnum][inwindowindices]
        event_hitqdata = self.jsondata["digitQ"][eventnum][inwindowindices]
        event_hittdata = self.jsondata["digitT"][eventnum][inwindowindices]
        event_data = []
        thiswinmin = timewindowmin
        for j in xrange(numwindows):
            thiswinmax = thiswinmin + tw_width
            for k,ahitID in enumerate(event_hitIDdata):
                thishit_data = np.zeros(0,numwindows+1)
                thishit_data[0] = ahitID
                if event_hittdata[k] > thiswinmin and event_hittdata[k] < thiswinmax:
                    thishit_data[k] =event_hitqdata[k]
                event_data.append(thishit_data)
        return event_data
        

