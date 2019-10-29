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

import lib.SimpleKDE as ske
import lib.LikelihoodCalculator as lc

DEBUG = False

INFILE = "./data/Processed/JSON_Data/PMTVolume_06262019_pixelmap.json"
TRAIN_PDFS = False 
SAVE_PDFS = False
LOAD_PDFS = True 
PLOTS = True

#THE REST GO TO THE PERFORMANCE EVALUATION
NUM_TRAINPDF_EVENTS = 3000
TRAIN_BANDWIDTHS_CHARGEANGLE = False
TRAIN_BANDWIDTHS_CHARGELENGTH = False

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
    #Let's form a dictionary for some decent array management
    distributions = {"pmt_phis": {}, "pmt_charges": {}, 
            "lappd_phis": {},"lappd_charges": {},"pmt_total_charge": {},
            "lappd_total_charge": {}}
    maxcharges = {}
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
    FV_event_indices = GetFVEventsOnly(data)
    print("LEN OF FV EVENTS: " + str(len(FV_event_indices)))
    
    
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
    target_data = picounts[FV_event_indices]
    target_data_train =  target_data[0:NUM_TRAINPDF_EVENTS]
    target_data_test = target_data[NUM_TRAINPDF_EVENTS:]

    track_lengths = np.array(data['trueTrackLengthInWater'])[FV_event_indices]
    max_tracklength = np.max(track_lengths)
    norm_track_lengths = track_lengths/max_tracklength
    
    distributions,maxcharges = GetPhiQDistribution(data,indices=FV_event_indices)
    distributions["track_length"] = track_lengths 
    
    #split distributions into training and testing.
    distributions_test = {}
    distributions_train = {}
    for key in distributions:
        distributions_train[key] = distributions[key][0:NUM_TRAINPDF_EVENTS]
        distributions_test[key] = distributions[key][NUM_TRAINPDF_EVENTS:]
    max_angle = 180
    norm_hit_angles = np.array(distributions_train["pmt_phis"])/max_angle
    
    max_hitcharge = maxcharges["PMT"]
    norm_hit_charges = np.array(distributions_train["pmt_charges"])/max_hitcharge
    max_totcharge = np.amax(distributions_train["pmt_total_charge"])
    norm_tot_charges = np.array(distributions_train["pmt_total_charge"])/max_totcharge


    #Neat.  Now, get the indices for events with only a muon and those with a pion
    pi_only = np.where(target_data_train>0)[0]
    mu_only = np.setdiff1d(np.arange(0,len(target_data_train),1),pi_only)

    #Form dataframes used to train KDEs FIXME: We need to split data fractionally
    #So some is used to train the KDE, and some is used to evaluate likelihood 
    #performance
    mu_chargeangle_df = pd.DataFrame({"charges": np.concatenate(norm_hit_charges[mu_only]),
        "angles": np.concatenate(norm_hit_angles[mu_only])})
    pi_chargeangle_df = pd.DataFrame({"charges": np.concatenate(norm_hit_charges[pi_only]),
        "angles": np.concatenate(norm_hit_angles[pi_only])})
    mu_chargelength_df = pd.DataFrame({"tot_charges": norm_tot_charges[mu_only],
        "lengths": norm_track_lengths[mu_only]})
    pi_chargelength_df = pd.DataFrame({"tot_charges": norm_tot_charges[pi_only],
        "lengths": norm_track_lengths[pi_only]})
    mu_chargeangle_KDE = ske.KernelDensityEstimator(dataframe=mu_chargeangle_df)
    pi_chargeangle_KDE = ske.KernelDensityEstimator(dataframe=pi_chargeangle_df)
    mu_chargelength_KDE = ske.KernelDensityEstimator(dataframe=mu_chargelength_df)
    pi_chargelength_KDE = ske.KernelDensityEstimator(dataframe=pi_chargelength_df)
    
    #Optimize bandwidths for charge-angle KDEs built in each direction.  Take mean
    mu_bw1 = 0.004
    mu_bw2 = 0.014
    pi_bw1 = 0.008
    pi_bw2 = 0.02
    if (TRAIN_BANDWIDTHS_CHARGEANGLE):
        bandlims = [0.002,0.02]
        numbands = 10
        mu_bw1 = mu_chargeangle_KDE.GetOptimalBandwidth("charges",bandlims,numbands)
        mu_bw2 = mu_chargeangle_KDE.GetOptimalBandwidth("angles",bandlims,numbands)
        pi_bw1 = pi_chargeangle_KDE.GetOptimalBandwidth("charges",bandlims,numbands)
        pi_bw2 = pi_chargeangle_KDE.GetOptimalBandwidth("angles",bandlims,numbands)
        print("OPTIMAL MUON KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(mu_bw1,mu_bw2))
        print("OPTIMAL PION KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(pi_bw1,pi_bw2))
    print("USING MEAN BANDWIDTH OF THE OPTIMAL BANDWIDTHS FOUND IN EACH DIMENSION")

    #Optimize bandwidths for charge-length KDEs built in each direction.  Take mean
    mu_bw3 = 0.006
    mu_bw4 = 0.012
    pi_bw3 = 0.062
    pi_bw4 = 0.03
    if (TRAIN_BANDWIDTHS_CHARGELENGTH):
        bandlims = [0.05,0.09]
        numbands = 30
        #mu_bw3 = mu_chargelength_KDE.GetOptimalBandwidth("tot_charges",bandlims,numbands)
        #mu_bw4 = mu_chargelength_KDE.GetOptimalBandwidth("lengths",bandlims,numbands)
        pi_bw3 = pi_chargelength_KDE.GetOptimalBandwidth("tot_charges",bandlims,numbands)
        #pi_bw4 = pi_chargelength_KDE.GetOptimalBandwidth("lengths",bandlims,numbands)
        print("OPTIMAL CHARGE-LENGTH MUON KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(mu_bw3,mu_bw4))
        print("OPTIMAL CHARGE-LENGTH PION KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(pi_bw3,pi_bw4))

    print("USING MEAN BANDWIDTH OF THE OPTIMAL BANDWIDTHS FOUND IN EACH DIMENSION")
    mu_bw_ca = (mu_bw1 + mu_bw2)/2.0
    pi_bw_ca = (pi_bw1 + pi_bw2)/2.0
    mu_bw_cl = (mu_bw3 + mu_bw4)/2.0
    pi_bw_cl = (pi_bw3 + pi_bw4)/2.0

    if PLOTS:
        #Make plots of each KDE for debugging
        mx,my,mz = mu_chargeangle_KDE.KDEEstimate2D(mu_bw_ca,"angles","charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,500./max_hitcharge],kern='gaussian')
        mx = mx*max_angle
        my = my*max_hitcharge
        mz=mz/np.max(mz)
        sns.set_style("whitegrid")
        sns.axes_style("darkgrid")
        xkcd_colors =  [ 'slate blue', 'green', 'grass','pink']
        sns.set_context("poster")
        sns.set_palette(sns.xkcd_palette(xkcd_colors))
        cbar = plt.contourf(mx,my,mz,40,cmap='inferno')
        plt.colorbar()
        plt.title("Hit charge vs. hit angle KDE for muon only, normalized to max of 1")
        plt.show()
    
        px,py,pz = pi_chargeangle_KDE.KDEEstimate2D(pi_bw_ca,"angles","charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,500./max_hitcharge],kern='gaussian')
        px = px*max_angle
        py = py*max_hitcharge
        pz=pz/np.max(pz)
        sns.set_style("whitegrid")
        sns.axes_style("darkgrid")
        xkcd_colors =  [ 'slate blue', 'green', 'grass','pink']
        sns.set_context("poster")
        sns.set_palette(sns.xkcd_palette(xkcd_colors))
        cbar = plt.contourf(px,py,pz,40,cmap='inferno')
        plt.colorbar()
        plt.title("Hit charge vs. hit angle KDE for muon + pions, normalized to max of 1")
        plt.show()
    
        mx_cl,my_cl,mz_cl = mu_chargelength_KDE.KDEEstimate2D(mu_bw_cl,"lengths","tot_charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,6000./max_totcharge],kern='gaussian')
        mx_cl = mx_cl*max_tracklength
        my_cl = my_cl*max_totcharge
        mz_cl=mz_cl/np.max(mz_cl)
        sns.set_style("whitegrid")
        sns.axes_style("darkgrid")
        xkcd_colors =  [ 'slate blue', 'green', 'grass','pink']
        sns.set_context("poster")
        sns.set_palette(sns.xkcd_palette(xkcd_colors))
        cbar = plt.contourf(mx_cl,my_cl,mz_cl,40,cmap='inferno')
        plt.colorbar()
        plt.title("Total charge vs. Track length KDE for muon only, normalized to max of 1")
        plt.show()
    
        px_cl,py_cl,pz_cl = pi_chargelength_KDE.KDEEstimate2D(pi_bw_cl,"lengths","tot_charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,6000./max_totcharge],kern='gaussian')
        px_cl = px_cl*max_tracklength
        py_cl = py_cl*max_totcharge
        pz_cl=pz_cl/np.max(pz_cl)
        sns.set_style("whitegrid")
        sns.axes_style("darkgrid")
        xkcd_colors =  [ 'slate blue', 'green', 'grass','pink']
        sns.set_context("poster")
        sns.set_palette(sns.xkcd_palette(xkcd_colors))
        cbar = plt.contourf(px_cl,py_cl,pz_cl,40,cmap='inferno')
        plt.colorbar()
        plt.title("Total charge vs. Track length KDE for muon + pions, normalized to max of 1")
        plt.show()

    #Now, we used KDE to estimate the charge-angle and charge-length PDFs.  From here,
    #we can calculate the likelihood that an event is a muon only or has muons + pions
    LikelihoodFunc = lc.LikelihoodFunction()
    LikelihoodFunc.Add2DPDF("Muon_ChargeAngle","S","pmt_phis","pmt_charges",mx,my,mz,weight=0.1)
    LikelihoodFunc.Add2DPDF("MuPi_ChargeAngle","B","pmt_phis","pmt_charges",px,py,pz,weight=0.1)
    LikelihoodFunc.Add2DPDF("Muon_TotChargeLength","S","track_length",
                            "pmt_total_charge",mx_cl,my_cl,mz_cl)
    LikelihoodFunc.Add2DPDF("MuPi_TotChargeLength","B","track_length",
                            "pmt_total_charge",px_cl,py_cl,pz_cl)
    Signal_likelihoods = LikelihoodFunc.GetLikelihoods(distributions_test)
    #Now, let's plot the likelihoods for Muon only and Muon + pions
    pi_only_test = np.where(target_data_test>0)[0]
    mu_only_test = np.setdiff1d(np.arange(0,len(target_data_test),1),pi_only_test)
    pi_likelihoods = Signal_likelihoods[pi_only_test]
    mu_likelihoods = Signal_likelihoods[mu_only_test]

    sns.set_style("whitegrid")
    sns.axes_style("darkgrid")
    xkcd_colors = ['red', 'grass',  'black', 'blue', 'cobalt']
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('Single ring likelihood')
    plt.hist(pi_likelihoods, bins=30, 
                linewidth=4,label='Muon + Pions',histtype="step")
    plt.hist(mu_likelihoods, bins=30, 
                linewidth=4,label='Muon only',histtype="step")
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("OH SHIIIIIT")
    plt.show()
    
    lcut = 0.3
    print("CUT ON PARAMETER AT %f"%(lcut))
    mu_pass = np.where(mu_likelihoods>lcut)[0]
    pi_pass = np.where(pi_likelihoods>lcut)[0]
    mu_acceptance = float(len(mu_pass))/len(mu_likelihoods)
    mu_acc_unc = np.sqrt(len(mu_pass))/len(mu_likelihoods)
    pi_acceptance = float(len(pi_pass))/len(pi_likelihoods)
    pi_acc_unc = np.sqrt(len(pi_pass))/len(pi_likelihoods)
    print("MUON ONLY ACCEPTANCE: %f PM %f"%(mu_acceptance,mu_acc_unc))
    print("MUON + PION ACCEPTANCE: %f PM %f"%(pi_acceptance,pi_acc_unc))
