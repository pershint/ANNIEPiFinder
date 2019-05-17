import uproot
import numpy as np

class ROOTProcessor(object):
    def __init__(self,treename="phaseII"):
        print("ROOTProcessor initializing")
        self.treename = treename
        self.processed_data = {}

    def clearProcessedData(self):
        '''
        Empties the current processed_data dictionary.
        '''
        self.processed_data = {}

    def setTreeName(self,treename="phaseII"):
        '''
        Set the tree of data to be accessed from loaded ROOT files.

        Input:
            treename [string]
            Tree in ROOT datafiles to be loaded.
        '''
        self.treename = treename

    def addROOTFile(self,rootfile,branches_to_get=None):
        '''
        Opens the root ntuple file at the path given and append it's data
        to the current data dictionary.
        
        Input:
            rootfile [string]
            Path to rootfile to open.

            branches_to_get [array]
            Array of data variables to load into the dictionary.  Use to
            specify specific data types in ROOT file to collect.
        '''
        f = uproot.open(rootfile)
        ftree = f.get(self.treename)
        all_data = ftree.keys()
        print(all_data)
        for dattype in all_data:
            if branches_to_get is not None:
                if dattype not in branches_to_get:
                    continue
            thistype_processed = ftree.get(dattype).array()
            self._appendProcessedEntry(dattype,thistype_processed)
            

    def _appendProcessedEntry(self, dattype, thistype_processed):
        if dattype in self.processed_data:
            thistype_proclist = list(thistype_processed)
            self.processed_data[dattype] = self.processed_data[dattype] + \
                                           thistype_proclist
        else:
            self.processed_data[dattype] = []
            thistype_proclist = list(thistype_processed)
            self.processed_data[dattype] = self.processed_data[dattype] + \
                                           thistype_proclist

    def removeNumpyArrays(self):
        '''
        returns a version of the processed data where entries are not numpy arrays.
        '''
        procdata_nonumpy = {}
        for datakey in self.processed_data:
            if type(self.processed_data[datakey][0]) == np.ndarray:
                procdata_nonumpy[datakey] = []
                for entry in self.processed_data[datakey]:
                    procdata_nonumpy[datakey].append(list(entry))
            elif type(self.processed_data[datakey][0]) == np.int64 or \
                 type(self.processed_data[datakey][0]) == np.int32:
                procdata_nonumpy[datakey] = []
                for entry in self.processed_data[datakey]:
                    procdata_nonumpy[datakey].append(int(entry))
            else:
                procdata_nonumpy[datakey] = self.processed_data[datakey]
        return procdata_nonumpy
