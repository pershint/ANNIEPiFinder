import sys
import copy
import json
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("poster")

from sklearn.model_selection import train_test_split

import lib.Normalization as nor
import lib.SimpleKDE as ske
import lib.LikelihoodCalculator as lc
import lib.Plots.LikelihoodPlots as lpl

DEBUG = False

INFILE = "./data/Processed/JSON_Data/PMTVolume_06262019_pixelmap.json"

#FIXME: Implement these
TRAIN_PDFS = False 
SAVE_PDFS = False
LOAD_PDFS = True 

PLOTS = True

#THE REST GO TO THE PERFORMANCE EVALUATION
NUM_TRAINPDF_EVENTS = 2500
NUM_DIMENSIONS = 1
TRAIN_BANDWIDTHS_CHARGEANGLE = False
TRAIN_BANDWIDTHS_CHARGELENGTH = False

def GetFVEventsOnly(f1data):
    '''
    Returns the indices associated with events that originate in 
    the fiducial volume of the ANNIE tank.
    '''
    truX = np.array(f1data['trueVtxX'])
    truY = np.array(f1data['trueVtxY'])
    truZ = np.array(f1data['trueVtxZ'])
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
    track_lengths = np.array(data['trueTrackLengthInWater'])
    
    #Initialize angle, charge, and track length arrays used in likelihood analysis 
    allphis_pmt = []
    allQs_pmt = []
    allphis_lappd = []
    allQs_lappd = []
    totQs_pmt = []
    totQs_lappd = []
    alltrack_lengths = []
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
        thistruTL =track_lengths[e]
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
        alltrack_lengths.append(thistruTL)
    #Let's form a dictionary for some decent array management
    distributions = {}
    maxcharges = {}
    distributions["track_length"] = alltrack_lengths
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
    
    #Massage training data: 0 = muon only event, 1 = muon + at least one pion
    picounts = np.array(data["Pi0Count"]) + np.array(data["PiPlusCount"]) + \
    np.array(data["PiMinusCount"])
    print("LEN OF TOTAL PI COUNTS: " + str(len(picounts)))
    #Set anything with at least one pion to 1
    has_pion = np.where(picounts > 0)[0]
    picounts[has_pion] = 1
    target_data = picounts[FV_event_indices]
    target_data_train =  target_data[0:NUM_TRAINPDF_EVENTS]
    target_data_test = target_data[NUM_TRAINPDF_EVENTS:]

    #Select down the input data to data used for likelihood analysis 
    distributions,maxcharges = GetPhiQDistribution(data,indices=FV_event_indices)
    
    #split distributions into training and testing.
    distributions_test = {}
    distributions_train = {}
    for key in distributions:
        distributions_train[key] = distributions[key][0:NUM_TRAINPDF_EVENTS]
        distributions_test[key] = distributions[key][NUM_TRAINPDF_EVENTS:]


    #Normalize data to range from [0,1]
    max_angle = 180
    norm_hit_angles = np.array(distributions_train["pmt_phis"])/max_angle
    max_hitcharge = maxcharges["PMT"]
    norm_hit_charges = np.array(distributions_train["pmt_charges"])/max_hitcharge
    max_totcharge = np.amax(distributions_train["pmt_total_charge"])
    norm_tot_charges = np.array(distributions_train["pmt_total_charge"])/max_totcharge
    max_tracklength = np.max(distributions["track_length"])
    norm_track_lengths = np.array(distributions_train["track_length"])/max_tracklength

    #Split training data into signal (muon only) and background (muon + pion) datasets
    #Background DataFrames
    pi_only = np.where(target_data_train>0)[0]
    pi_chargeangle_df = pd.DataFrame({"charges": np.concatenate(norm_hit_charges[pi_only]),
        "angles": np.concatenate(norm_hit_angles[pi_only])})
    pi_chargelength_df = pd.DataFrame({"tot_charges": norm_tot_charges[pi_only],
        "lengths": norm_track_lengths[pi_only]})
    #Signal DataFrames
    mu_only = np.setdiff1d(np.arange(0,len(target_data_train),1),pi_only)
    mu_chargelength_df = pd.DataFrame({"tot_charges": norm_tot_charges[mu_only],
        "lengths": norm_track_lengths[mu_only]})
    mu_chargeangle_df = pd.DataFrame({"charges": np.concatenate(norm_hit_charges[mu_only]),
        "angles": np.concatenate(norm_hit_angles[mu_only])})
    
    mu_chargeangle_KDE = ske.KernelDensityEstimator(dataframe=mu_chargeangle_df)
    pi_chargeangle_KDE = ske.KernelDensityEstimator(dataframe=pi_chargeangle_df)
    mu_chargelength_KDE = ske.KernelDensityEstimator(dataframe=mu_chargelength_df)
    pi_chargelength_KDE = ske.KernelDensityEstimator(dataframe=pi_chargelength_df)
    
    #Optimize bandwidths for charge-angle KDEs built in each direction.  Take mean
    mu_bandwidths,pi_bandwidths = {},{}
    mu_bandwidths["charges"] = 0.006
    mu_bandwidths["angles"] = 0.012
    pi_bandwidths["charges"] = 0.006
    pi_bandwidths["angles"] = 0.012
    if (TRAIN_BANDWIDTHS_CHARGEANGLE):
        bandlims = [0.002,0.015]
        numbands = 15
        mu_bandwidths["charges"] = mu_chargeangle_KDE.GetOptimalBandwidth("charges",bandlims,numbands)
        mu_bandwidths["angles"] = mu_chargeangle_KDE.GetOptimalBandwidth("angles",bandlims,numbands)
        pi_bandwidths["charges"] = pi_chargeangle_KDE.GetOptimalBandwidth("charges",bandlims,numbands)
        pi_bandwidths["angles"] = pi_chargeangle_KDE.GetOptimalBandwidth("angles",bandlims,numbands)
        print("OPTIMAL MUON KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(mu_bandwidths["charges"],
                                                                        mu_bandwidths["angles"]))
        print("OPTIMAL PION KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(pi_bandwidths["charges"],
                                                                        pi_bandwidths["angles"]))
    print("USING MEAN BANDWIDTH OF THE OPTIMAL BANDWIDTHS FOUND IN EACH DIMENSION")

    #Optimize bandwidths for charge-length KDEs built in each direction.  Take mean
    mu_bandwidths["tot_charges"] = 0.012
    mu_bandwidths["lengths"] = 0.017
    pi_bandwidths["tot_charges"] = 0.03
    pi_bandwidths["lengths"] = 0.022
    
    if (TRAIN_BANDWIDTHS_CHARGELENGTH):
        bandlims = [0.001,0.03]
        numbands = 30
        mu_bandwidths["tot_charges"] = mu_chargelength_KDE.GetOptimalBandwidth("tot_charges",bandlims,numbands)
        mu_bandwidths["lengths"] = mu_chargelength_KDE.GetOptimalBandwidth("lengths",bandlims,numbands)
        pi_bandwidths["tot_charges"] = pi_chargelength_KDE.GetOptimalBandwidth("tot_charges",bandlims,numbands)
        pi_bandwidths["lengths"] = pi_chargelength_KDE.GetOptimalBandwidth("lengths",bandlims,numbands)
        print("OPTIMAL CHARGE-LENGTH MUON KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(mu_bandwidths["tot_charges"],mu_bandwidths["lengths"]))
        print("OPTIMAL CHARGE-LENGTH PION KDE BANDWIDTHS USING FULL DATASET: %f,%f "%(pi_bandwidths["tot_charges"],pi_bandwidths["lengths"]))

    if NUM_DIMENSIONS == 1:
        print("#### USING ONE-DIMENSIONAL PDFS TO CONSTRUCT LIKELIHOOD FUNCTIONS ####")

        #Construct 1D PDFs using hit angle and hit charge for signal & background
        mx_angle,my_angle = mu_chargeangle_KDE.KDEEstimate1D(mu_bandwidths["angles"],
                "angles",x_range=[0,1],bins=100,kern='gaussian')
        mx_angle = mx_angle*max_angle
        my_angle = my_angle/np.max(my_angle)

        mx_charge,my_charge = mu_chargeangle_KDE.KDEEstimate1D(mu_bandwidths["charges"],
                "charges",x_range=[0,1],bins=100,kern='gaussian')
        mx_charge = mx_charge*max_hitcharge
        my_charge = my_charge/np.max(my_charge)

        px_angle,py_angle = pi_chargeangle_KDE.KDEEstimate1D(pi_bandwidths["angles"],
                "charges",x_range=[0,1],bins=100,kern='gaussian')
        px_angle = px_angle*max_angle
        py_angle = py_angle/np.max(py_angle)

        px_charge,py_charge = pi_chargeangle_KDE.KDEEstimate1D(pi_bandwidths["charges"],
                "charges",x_range=[0,1],bins=100,kern='gaussian')
        px_charge = px_charge*max_hitcharge
        py_charge = py_charge/np.max(py_charge)

        #Construct 1D PDFs using track length and total charge for signal & background
        mx_length,my_length = mu_chargelength_KDE.KDEEstimate1D(mu_bandwidths["lengths"],
                "lengths",x_range=[0,1],bins=100,kern='gaussian')
        mx_length = mx_length*max_tracklength
        my_length = my_length/np.max(my_length)

        mx_totcharge,my_totcharge = mu_chargelength_KDE.KDEEstimate1D(mu_bandwidths["tot_charges"],
                "tot_charges",x_range=[0,1],bins=100,kern='gaussian')
        mx_totcharge = mx_totcharge*max_totcharge
        my_totcharge = my_totcharge/np.max(my_totcharge)

        px_length,py_length = pi_chargelength_KDE.KDEEstimate1D(pi_bandwidths["lengths"],
                "lengths",x_range=[0,1],bins=100,kern='gaussian')
        px_length = px_length*max_tracklength
        py_length = py_length/np.max(py_length)

        px_totcharge,py_totcharge = pi_chargelength_KDE.KDEEstimate1D(pi_bandwidths["tot_charges"],
                "tot_charges",x_range=[0,1],bins=100,kern='gaussian')
        px_totcharge = px_totcharge*max_totcharge
        py_totcharge = py_totcharge/np.max(py_totcharge)

    if NUM_DIMENSIONS == 2:
        print("#### USING TWO-DIMENSIONAL PDFS TO CONSTRUCT LIKELIHOOD FUNCTIONS ####")
        print("USING MEAN BANDWIDTH OF THE OPTIMAL BANDWIDTHS FOUND IN EACH DIMENSION")
        mu_bw_ca = (mu_bandwidths["charges"] + mu_bandwidths["angles"])/2.0
        pi_bw_ca = (pi_bandwidths["charges"] + pi_bandwidths["angles"])/2.0
        mu_bw_cl = (mu_bandwidths["tot_charges"] + mu_bandwidths["lengths"])/2.0
        pi_bw_cl = (pi_bandwidths["tot_charges"] + pi_bandwidths["lengths"])/2.0

        #Develop two-dimensional PDFs using Kernel Density Estimation, then re-scale
        mx,my,mz = mu_chargeangle_KDE.KDEEstimate2D(mu_bw_ca,"angles","charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,500./max_hitcharge],kern='gaussian')
        mx = mx*max_angle
        my = my*max_hitcharge
        mz=mz/np.max(mz)

        px,py,pz = pi_chargeangle_KDE.KDEEstimate2D(pi_bw_ca,"angles","charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,500./max_hitcharge],kern='gaussian')
        px = px*max_angle
        py = py*max_hitcharge
        pz=pz/np.max(pz)

        mx_cl,my_cl,mz_cl = mu_chargelength_KDE.KDEEstimate2D(mu_bw_cl,"lengths","tot_charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,6000./max_totcharge],kern='gaussian')
        mx_cl = mx_cl*max_tracklength
        my_cl = my_cl*max_totcharge
        mz_cl=mz_cl/np.max(mz_cl)

        px_cl,py_cl,pz_cl = pi_chargelength_KDE.KDEEstimate2D(pi_bw_cl,"lengths","tot_charges",xbins=100j,
                ybins=100j,x_range=[0,1],y_range=[0,6000./max_totcharge],kern='gaussian')
        px_cl = px_cl*max_tracklength
        py_cl = py_cl*max_totcharge
        pz_cl=pz_cl/np.max(pz_cl)


    #Training done: re-scale data now that KDE training is complete
    mu_chargeangle_df["angles"] = mu_chargeangle_df["angles"]*max_angle 
    mu_chargeangle_df["charges"] = mu_chargeangle_df["charges"]*max_hitcharge
    pi_chargeangle_df["angles"] = pi_chargeangle_df["angles"]*max_angle 
    pi_chargeangle_df["charges"] = pi_chargeangle_df["charges"]*max_hitcharge
    mu_chargelength_df["lengths"] = mu_chargelength_df["lengths"]*max_tracklength
    mu_chargelength_df["tot_charges"] = mu_chargelength_df["tot_charges"]*max_totcharge
    pi_chargelength_df["lengths"] = pi_chargelength_df["lengths"]*max_tracklength
    pi_chargelength_df["tot_charges"] = pi_chargelength_df["tot_charges"]*max_totcharge

    #Plot data distributions and PDFs constructed using KDE with data
    if PLOTS and NUM_DIMENSIONS == 2:
        xtitle,ytitle = "PMT Phi from vertex dir. (deg)","PMT Charge (pe)"

        title = ("Distribution of PMT charges relative to muon direction \n" + 
                 "Muon stops in MRD, muon only")
        lpl.Plot2DDataDistribution(mu_chargeangle_df,"angles","charges",
                                   xtitle,ytitle,title)
        
        title = ("Hit charge vs. hit angle KDE for muon only")
        lpl.PlotKDE(mx,my,mz,40,xtitle,ytitle,title)

        title = ("Distribution of PMT charges relative to muon direction \n" + 
                 "Muon stops in MRD, muon + pions")
        lpl.Plot2DDataDistribution(pi_chargeangle_df,"angles","charges",
                                   xtitle,ytitle,title)
        title = ("Hit charge vs. hit angle KDE for muon + pions")
        lpl.PlotKDE(px,py,pz,40,xtitle,ytitle,title)

        #### TRACK LENGTH VS. MUON ANGLE DATA AND PDFS ####
        xtitle,ytitle = "Muon track length in tank (m)","Total PMT Charge (pe)"
        title = ("Total PMT charge vs. true muon track length \n" + 
                 "Muon stops in MRD, muon only")
        lpl.Plot2DDataDistribution(mu_chargelength_df,"lengths","tot_charges",
                                   xtitle,ytitle,title)
        
        title = ("Track length vs. total charge KDE for muon only")
        lpl.PlotKDE(mx_cl,my_cl,mz_cl,40,xtitle,ytitle,title)

        title = ("Total PMT charge vs. true muon track length \n" + 
                 "Muon stops in MRD, muon + pions")
        lpl.Plot2DDataDistribution(pi_chargelength_df,"lengths","tot_charges",
                                   xtitle,ytitle,title)
        title = ("Track length vs. total charge KDE for muon + pions")
        lpl.PlotKDE(px_cl,py_cl,pz_cl,40,xtitle,ytitle,title)

    
    LikelihoodFunc = lc.LikelihoodCalculator()
    #Load developed PDFs for use in the likelihood function
    if NUM_DIMENSIONS == 1:
        LikelihoodFunc.Add1DPDF("Muon_HitCharge","S","pmt_charges",mx_charge,my_charge,weight=1.0)
        LikelihoodFunc.Add1DPDF("Muon_HitAngle","S","pmt_phis",mx_angle,my_angle,weight=1.0)
        LikelihoodFunc.Add1DPDF("Muon_TotCharge","S","pmt_total_charge",mx_totcharge,my_totcharge,weight=1.0)
        LikelihoodFunc.Add1DPDF("Muon_TrackLength","S","track_length",mx_length,my_length,weight=1.0)
        LikelihoodFunc.Add1DPDF("MuPi_HitCharge","B","pmt_charges",px_charge,py_charge,weight=1.0)
        LikelihoodFunc.Add1DPDF("MuPi_HitAngle","B","pmt_phis",px_angle,py_angle,weight=1.0)
        LikelihoodFunc.Add1DPDF("MuPi_TotCharge","B","pmt_total_charge",px_totcharge,py_totcharge,weight=1.0)
        LikelihoodFunc.Add1DPDF("MuPi_TrackLength","B","track_length",px_length,py_length,weight=1.0)

    if NUM_DIMENSIONS == 2:
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
    leg = ax.legend(loc=2,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Likelihood ratio for Muon and Muon + Pion test data")
    plt.show()
   
    lcut_optimal, lcuts_checked, efficiencies, purities = LikelihoodFunc.OptimizeCut(mu_likelihoods,pi_likelihoods)

    print("OPTIMAL CUT ON PARAMETER AT %f"%(lcut_optimal))
    mu_pass = np.where(mu_likelihoods>lcut_optimal)[0]
    pi_pass = np.where(pi_likelihoods>lcut_optimal)[0]
    mu_acceptance = float(len(mu_pass))/len(mu_likelihoods)
    mu_acc_unc = np.sqrt(len(mu_pass))/len(mu_likelihoods)
    pi_acceptance = float(len(pi_pass))/len(pi_likelihoods)
    pi_acc_unc = np.sqrt(len(pi_pass))/len(pi_likelihoods)
    print("MUON ONLY ACCEPTANCE: %f PM %f"%(mu_acceptance,mu_acc_unc))
    print("MUON + PION ACCEPTANCE: %f PM %f"%(pi_acceptance,pi_acc_unc))

    sns.set_style("whitegrid")
    sns.axes_style("darkgrid")
    xkcd_colors = ['black', 'blue', 'purple']
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('Likelihood parameter cut')
    plt.plot(np.array(lcuts_checked),np.array(efficiencies), 
                linewidth=4,label='Signal Efficiency = $S_{acc}/S_{tot}$')
    plt.plot(np.array(lcuts_checked),np.array(purities), 
                linewidth=4,label='Signal Purity = $S_{acc}/(S_{acc}+B_{acc})$')
    plt.plot(np.array(lcuts_checked),np.array(purities)*np.array(efficiencies), 
                linewidth=4,label='Purity*Efficiency')
    leg = ax.legend(loc=3,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Efficiency and Purity of muons for varying classifier cut")
    plt.show()
   
