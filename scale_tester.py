import sys
import copy
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

import lib.ChargeScaler as cw

DEBUG = False

INFILE = "./data/Processed/JSON_Data/PMTVolume_06262019_pixelmap.json"

def GetFVEventsOnly(f1data):
    '''
    Returns the indices associated with events that originate in 
    the fiducial volume of the ANNIE tank.
    '''
    truX = np.array(f1data['trueVtxZ'])
    truY = np.array(f1data['trueVtxY'])
    truZ = np.array(f1data['trueVtxX'])
    FV_events = np.where((truZ<0) & (abs(truY)<50))[0]
    return FV_events

def GetPhiQDistribution(f1data,indices=[]): 
    DiZ = np.array(f1data['digitZ'])
    DiY = np.array(f1data['digitY'])
    DiX = np.array(f1data['digitX'])
    DiQ = np.array(f1data['digitQ'])
    DiType = np.array(f1data['digitType'])
    truDirZ = np.array(f1data['trueDirZ'])
    truDirY = np.array(f1data['trueDirY'])
    truDirX = np.array(f1data['trueDirX'])
    truX = np.array(f1data['trueVtxX'])
    truY = np.array(f1data['trueVtxY'])
    truZ = np.array(f1data['trueVtxZ'])
    #Get the phi
    allphis_pmt = []
    allQs_pmt = []
    allphis_lappd = []
    allQs_lappd = []
    totQs_pmt = []
    totQs_lappd = []
    pmt_posns = []
    pmt_dirs = []
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
        this_evt_posns = []
        for j,digit in enumerate(thisDiX):
            if j not in pmtind: continue
            this_evt_posns.append([thisDiX[j],thisDiY[j],thisDiZ[j]])
        pmt_posns.append(this_evt_posns)
        allphis_pmt.append(phi_deg[pmtind])
        allQs_pmt.append(thisQ[pmtind])
        totQs_pmt.append(np.sum(thisQ[pmtind]))
        allphis_lappd.append(phi_deg[lappdind])
        totQs_lappd.append(np.sum(thisQ[lappdind]))
    #Let's form a dictionary for some decent array management
    distributions = {"pmt_phis": {}, "pmt_charges": {}, "pmt_posns": [],
            "lappd_phis": {},"lappd_charges": {},"pmt_total_charge": {},
            "lappd_total_charge": {}}
    maxcharges = {}
    distributions["pmt_posns"] = pmt_posns 
    distributions["pmt_phis"] = allphis_pmt
    distributions["lappd_phis"] = allphis_lappd
    distributions["pmt_charges"] = allQs_pmt
    distributions["lappd_charges"] = allQs_lappd
    distributions["pmt_total_charge"] = totQs_pmt
    distributions["lappd_total_charge"] = totQs_lappd
    distributions["entrynum"] = np.arange(0,len(totQs_pmt),1)
    maxcharges["PMT"] = max_charge_pmt
    maxcharges["LAPPD"] = max_charge_lappd
    return distributions, maxcharges

if __name__ == '__main__':
    data = {}
    with open(INFILE,"r") as f:
        data = json.load(f)


    #Down-select to FV events only
    print("GETTING INDICES FOR FV EVENTS")
    FV_event_indices = GetFVEventsOnly(data)
    print("LEN OF FV EVENTS: " + str(len(FV_event_indices)))
    
    #None of this matters for this debugger 
    print("GETTING PHIQ DISTRIBUTIONS")
    distributions,maxcharges = GetPhiQDistribution(data,indices=FV_event_indices)
    print("GOT PHIQ DISTRIBUTIONS")
    #Get Muon positions and directions; down-select to only those in the FV.
    muPosX = np.array(data["trueVtxX"])[FV_event_indices]
    muPosY = np.array(data["trueVtxY"])[FV_event_indices]
    muPosZ = np.array(data["trueVtxZ"])[FV_event_indices]
    muDirX = np.array(data["trueDirX"])[FV_event_indices]
    muDirY = np.array(data["trueDirY"])[FV_event_indices]
    muDirZ = np.array(data["trueDirZ"])[FV_event_indices]

    CW = cw.ChargeScaler()
    #Let's do this the super nasty way first.  Because easy.
    all_pmt_charges = []
    all_scaled_charges = []
    all_scaled_factors = []
    all_pmt_phis = []
    for entry in range(len(distributions["entrynum"])):
        pmt_phis = distributions["pmt_phis"][entry]
        pmt_charges = distributions["pmt_charges"][entry]
        pmt_positions = distributions["pmt_posns"][entry]
        if (len(pmt_positions)==0):
            print("ENTRY %i HAS NO PMT HITS.  EVENT PROBABLY WOULD HAVE FAILED RECO"%(entry))
            print("WE'LL SKIP IT....")
            continue
        muon_pos = np.array([muPosX[entry],
                              muPosY[entry],
                              muPosZ[entry]])
        muon_dir = np.array([muDirX[entry],
                              muDirY[entry],
                              muDirZ[entry]])
        scale_factors = CW.ScaleFactors(muon_pos,muon_dir,pmt_positions)
        scaled_charges = pmt_charges / scale_factors
        all_pmt_phis.append(pmt_phis)
        all_pmt_charges.append(pmt_charges)
        all_scaled_charges.append(scaled_charges)
        all_scaled_factors.append(scale_factors)
    
    thedata = pd.DataFrame({"phi":np.concatenate(np.array(all_pmt_phis)),
        "scale_factor":np.concatenate(np.array(all_scaled_factors)),
        "scaled_charge":np.concatenate(np.array(all_scaled_charges)),
        "charge":np.concatenate(np.array(all_pmt_charges))})
    print(thedata)
    fig = plt.figure()
    g = sns.jointplot(x="phi", y="scale_factor", data=thedata,kind="hex",stat_func=None)
    g = g.set_axis_labels("PMT angle from muon (deg)", "Scale Factor")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.9,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("Scale factor calculated for PMT angles ($Q'=Q/ScaleFactor$)")
    plt.show()

    thedata = thedata[thedata["scaled_charge"] < 600]
    fig = plt.figure()
    g = sns.jointplot(x="charge", y="scaled_charge", data=thedata,kind="hex",
            stat_func=None,ylim=(0,600))
    g = g.set_axis_labels("PMT charge (pe)", "Scaled PMT charge (pe')")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.9,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("PMT charge (Q) vs. Scaled charge ($Q'=Q/ScaleFactor$)")
    plt.show()
