import sys
import copy
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

import lib.Normalization as nor
from sklearn.model_selection import train_test_split

DEBUG = False

INFILE = "./data/Processed/JSON_Data/LiliaComb_05072019_pixelmap_wMuTruth.json"
TRAIN_PDFS = False 
SAVE_PDFS = False
LOAD_PDFS = True 
PLOTS = True

#THE REST GO TO THE PERFORMANCE EVALUATION
NUM_TRAINPDF_EVENTS = 2000

def GetPhiQDistribution(f1data,indices=[]): 
    DiZ = np.array(f1data['digitZ'])
    DiY = np.array(f1data['digitY'])
    DiX = np.array(f1data['digitX'])
    DiQ = np.array(f1data['digitQ'])
    DiType = np.array(f1data['digitType'])
    truDirZ = np.array(f1data['trueDirZ'])
    truDirY = np.array(f1data['trueDirY'])
    truDirX = np.array(f1data['trueDirX'])
    truX = np.array(f1data['trueVtxZ'])
    truY = np.array(f1data['trueVtxY'])
    truZ = np.array(f1data['trueVtxX'])
    #Get the phi
    allphis_pmt = []
    allQs_pmt = []
    allphis_lappd = []
    allQs_lappd = []
    totQs_pmt = []
    totQs_lappd = []
    if len(indices)==0:
        indices = range(len(DiZ))
    max_charge_pmt = 0
    max_charge_lappd = 0
    for e in indices:
        thisQ = np.array(DiQ[e])
        typedata = np.array(DiType[e])
        #For each digit, we need the dir from trueVtx to the digit
        thisDiX =np.array(DiX[e])
        thisDiY =np.array(DiY[e])
        thisDiZ =np.array(DiZ[e])
        thistruX =np.array(truX[e])
        thistruY =np.array(truY[e])
        thistruZ =np.array(truZ[e])
        magdiff = np.sqrt((thisDiX-thistruX)**2 + (thisDiY-thistruY)**2 + (thisDiZ-thistruZ)**2)
        pX = (thisDiX - thistruX)/(magdiff)
        pY = (thisDiY - thistruY)/(magdiff)
        pZ = (thisDiZ - thistruZ)/(magdiff)
        thistrudirX = truDirX[e]
        thistrudirY = truDirY[e]
        thistrudirZ = truDirZ[e]
        phi_rad = np.arccos(pX*thistrudirX + pY*thistrudirY + pZ*thistrudirZ)
        phi_deg = phi_rad * (180/np.pi)
        pmtind = np.where(typedata==0)[0]
        lappdind = np.where(typedata==1)[0]
        pmtmax, lappdmax = 0,0
        if (len(pmtind)>0): pmtmax = np.max(thisQ[pmtind])
        if (len(lappdind)>0): lappdmax = np.max(thisQ[lappdind])
        if pmtmax > max_charge_pmt: max_charge_pmt = pmtmax
        if lappdmax > max_charge_lappd: max_charge_lappd = lappdmax
        allphis_pmt.append(phi_deg[pmtind])
        allQs_pmt.append(thisQ[pmtind])
        totQs_pmt.append(np.sum(thisQ[pmtind]))
        allphis_lappd.append(phi_deg[lappdind])
        totQs_lappd.append(np.sum(thisQ[lappdind]))
    return allphis_pmt, allQs_pmt, allphis_lappd, allQs_lappd, totQs_pmt, totQs_lappd, max_charge_pmt, max_charge_lappd

if __name__ == '__main__':
    data = {}
    with open(INFILE,"r") as f:
        data = json.load(f)

    #THINGS TO DO:
    #First, process the pi counts to get the training target of 0 or 1
    #Regularize the data.  We know our ranges of interest:
    #    Hit angle: 0,180    Hit charge: 0-500
    #    Muon track length: 0,3.5 Total charge: 0-6000
    #    Or, we could try regularizing using the scipy regularization library
    #Get only the first 2000 events and then show the KDEs after regularization

    picounts = np.array(data["Pi0Count"] + data["PiPlusCount"] + data["PiMinusCount"])
    print("TOTAL PI COUNTS: " + str(picounts[0:99]))
    #Set anything with at least one pion to 1
    has_pion = np.where(picounts > 0)[0]
    picounts[has_pion] = 1
    target_data = picounts

    #Get hit charges and angles from data for PMTs and LAPPDs
    allphis_pmt, allQs_pmt, allphis_lappd, allQs_lappd ,totQs_pmt, totQs_lappd, maxQ_pmt, maxQ_lappd = GetPhiQDistribution(data)
    

    #Starting with only PMT information for now.
    max_angle = 180
    norm_hit_angles = np.array(allphis_pmt)/max_angle

    norm_hit_charges = allQs_pmt/maxQ_pmt
    print("NORMHITCHARGES: " + str(norm_hit_charges[0:5]))

    max_totcharge = np.amax(totQs_pmt)
    norm_tot_charges = totQs_pmt/max_totcharge

    track_lengths = np.array(data['trueTrackLengthInWater'])
    track_max = np.max(track_lengths)
    norm_track_lengths = track_lengths/track_max
    print("TRACK LENGTHS: " + str(norm_track_lengths[0:10]))
    input_data = []
    for j,entry in enumerate(norm_hit_angles):
        input_data.append([norm_hit_charges[j], norm_hit_angles[j], norm_tot_charges[j], norm_track_lengths[j]])
    print("X DATA IS: " + str(input_data[0:5]))
    plt.hist(track_lengths)
    plt.show()
    #Now, let's form PDFs using KDE with a hard-coded bandwidth of 0.1 to start.  Form
    #KDE with the first 2000 events.

