import uproot
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd

import lib.ROOTProcessor as rp

sns.set_context('poster')
import sys
import numpy as np

class PixelMapper(object):
    '''
    Class takes all PMT data available in a given file and
    makes a mapping of PMT IDs to pixels.  
    Inputs:
        jsondata [dict] Dictionary file output after feeding a 
        ROOT file to the ROOTPRocessor class.
    
    '''

    def __init__(self, jsondata = None):
        self.jsondata = jsondata

    def loadJSONData(self, datafile):
        self.jsondata = datafile

    def _XZ_ToTheta(self, xdata, zdata):
        theta = []
        for i in range(len(xdata)):
            thistheta = None
            x = xdata[i]
            z = zdata[i]
            isnegative = False
            print("THIS X,Z: %d,%d"%(x,z))
            if x <0:
                isnegative = True
                x = -x
            r = np.sqrt(z**2 + x**2)
            if r == 0:
                thistheta = np.arccos(0)*180.0/np.pi
            else:
                thistheta = np.arccos(z/r)*180.0/np.pi 
            if isnegative:
                thistheta = (360.0 - thistheta)
                #thistheta = (180.0 + thistheta)
            #Now, do the transormation to beam theta
            if thistheta < 180:
                thistheta = - thistheta
            else:
                thistheta =  (360.0 - thistheta)
    
            theta.append(thistheta) #yeah that's confusing labeling
            print("THIS THETA: " + str(thistheta))
        return np.array(theta)
    
    def _YVSTheta(self, pmt_only,ycut):
        ids_plotted = []
        all_xs = []
        all_zs = []
        all_ys = []
        all_thetas = []
        all_rs = []
        print(len(self.jsondata["digitDetID"]))
        for ev in range(len(self.jsondata["digitDetID"])):
            evx = self.jsondata["digitX"][ev]
            evy = self.jsondata["digitY"][ev]
            evz = self.jsondata["digitZ"][ev]
            evtype = self.jsondata["digitType"][ev]
            evid = self.jsondata["digitDetID"][ev]
            for j,hit in enumerate(evid):
                if hit in ids_plotted: 
                    continue
                if pmt_only and evtype[j]==1:
                    continue
                if ycut is not None and abs(evy[j]) > ycut:
                    continue
                else:
                    ids_plotted.append(hit)
                    theta = self._XZ_ToTheta([evx[j]],[evz[j]])
                    thisr = np.sqrt(evx[j]**2 + evz[j]**2)
                    all_ys.append(evy[j])
                    all_zs.append(evz[j])
                    all_xs.append(evx[j])
                    all_thetas.append(theta[0])
        #Let's cheese in that one missing PMT... weird...
        ids_plotted.append(-1)
        all_ys.append(103)
        all_thetas.append(-167)
        return np.array(ids_plotted), np.array(all_ys), np.array(all_thetas)
    
    def MapPositionsToPixels(self,pmt_only = True, ycut=130):
        ids, ys, thetas = self._YVSTheta(pmt_only,ycut)
        Y_ranges = [[90,120], [50,80], [10,40], [-40,-10], [-80, -50], [-120,-90]]
        numypixels = len(Y_ranges)    
        Th_ranges = [ [-180,-160],[-160,-135], [-135,-110],  [-110, -85],
                [-85, -65], [-65, -50], [-50, -20], [-20,0], [0,20], 
                 [20, 50],[50, 65], [65, 85], [85, 110], [110, 135], 
                 [135, 160],[160, 180]]
        numxpixels = len(Th_ranges)
        #Now, we need to map these IDs to pixel values
        #Start by sorting everything in terms of smallest ys
        print("SORTED THETAS")
        print(sorted(thetas))
        pixel_map = {"xpixel": [], "ypixel": [], "id": []}
        for j,yrange in enumerate(Y_ranges):
            for k,thrange in enumerate(Th_ranges):
                print("THE YRANGE: %s"%(str(yrange)))
                print("THE THRANGE: %s"%(str(thrange)))
                ranged_yind = np.where((ys>yrange[0]) & (ys<yrange[1]))[0]
                ranged_thind = np.where((thetas>thrange[0]) & (thetas<thrange[1]))[0]
                print("INDS IN YRANGE: %s"%(str(ranged_yind)))
                print("INDS IN THRANGE: %s"%(str(ranged_thind)))
                theindex = np.intersect1d(ranged_yind,ranged_thind)
                if len(theindex) > 1:
                    print("OH CRAP, YOUR Y-THETA RANGE HAS MORE THAN ONE ENTRY...")
                pixel_map["id"].append(ids[theindex[0]])
                pixel_map["xpixel"].append(k)
                pixel_map["ypixel"].append(j)
        return pd.DataFrame(pixel_map), numxpixels, numypixels

    def PlotPixelMap(self, pixel_map):
        '''
        Plots the Detector ID as a function of x and y pixel number.

        Inputs:

        pixel_map [Dictionary] 
        Pandas DataFrame output from the MapPositionsToPixels method.
        '''
        pmp = pixel_map.pivot(index="ypixel",columns="xpixel",values="id")
        ax = sns.heatmap(pmp)
        plt.show()




if __name__=='__main__':

    if str(sys.argv[1])=="--help":
        print("USAGE: python PMTMap.py [file1.root] ")
        sys.exit(0)
    f1 = str(sys.argv[1])
    #Process data into JSON objects
    mybranches = ['digitX','digitY','digitZ',
            'digitType','digitDetID']
    f1Processor = rp.ROOTProcessor(treename="phaseII")
    #branch = ['Pi0Count']
    f1Processor.addROOTFile(f1,branches_to_get=mybranches)
    f1data = f1Processor.getProcessedData()
    diX = np.array(f1data['digitX'])
    diY = np.array(f1data['digitY'])
    diZ = np.array(f1data['digitZ'])
    diType = np.array(f1data['digitType'])
    diID = np.array(f1data['digitDetID'])

    ids, ys, thetas = YVSTheta(diX,diY,diZ,diType,diID)
    #Now, create the map of IDs to pixel indices
    pixel_map = PositionsToPixels(ids, ys, thetas)
    print(pixel_map)
    #with open("./pixel_map.json","w") as f:
    #    json.dump(f,pixel_map,indent=4,sort_keys=True)
