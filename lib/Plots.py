import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
sns.set_context('poster')
sns.set(font_scale=3.0)

def ShowChargeDistributions(in_data,out_data):
    '''Gets the indices for events with a pion and without a pion and
    plots the histogram of summed charges for the two distributions'''
    nopion_sums = {'xpixel':[], 'ypixel':[], 'channel_sum':[]}
    pion_sums = {'xpixel':[], 'ypixel':[], 'channel_sum':[]}
    num_xpixels = len(in_data[0])
    num_ypixels = len(in_data[0][0])
    num_time_chans = len(in_data[0][0][0])
    nopion_inds = None
    pion_inds = None
    if type(out_data[0]) == np.ndarray:
        if len(out_data[0]) > 0:
            nopion_inds = np.where(np.sum(out_data,axis=1) == 0)[0]
            pion_inds = np.where(np.sum(out_data,axis=1) > 0)[0]
    else:
        nopion_inds = np.where(out_data == 0)[0]
        pion_inds = np.where(out_data > 0)[0]
    nopion_channel_vals = in_data[nopion_inds,0:num_xpixels,0:num_ypixels,0:num_time_chans]
    print("BEFORE SUM")
    print(nopion_channel_vals)
    #Now, sum the charges in all time windows in all pixels for each event
    num_collapses = len(nopion_channel_vals.shape)-1
    for x in range(num_collapses):
        nopion_channel_vals = np.sum(nopion_channel_vals,axis=1)
    print("AFTER SUM")
    print(nopion_channel_vals)

    pion_channel_vals = in_data[pion_inds,0:num_xpixels,0:num_ypixels,0:num_time_chans]
    print("BEFORE SUM")
    print(pion_channel_vals)
    #Now, sum the charges in all time windows in all pixels for each event
    num_collapses = len(pion_channel_vals.shape)-1
    for x in range(num_collapses):
        pion_channel_vals = np.sum(pion_channel_vals,axis=1)
    print("AFTER SUM")
    print(pion_channel_vals)

    NBINS = 40
    sns.set_style("whitegrid")
    sns.axes_style("darkgrid")
    xkcd_colors =  [ 'warm pink', 'slate blue', 'green', 'grass']
    sns.set_palette(sns.xkcd_palette(xkcd_colors))
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(nopion_channel_vals,bins=NBINS,label="No pion",normed=True)
    plt.hist(pion_channel_vals,bins=NBINS,label="One pion",alpha=0.6,normed=True)
    leg = ax.legend(fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    for t in ax.yaxis.get_major_ticks(): t.label.set_fontsize(20)
    for t in ax.xaxis.get_major_ticks(): t.label.set_fontsize(20)
    plt.ylabel("Probability of total charge",fontsize=30)
    plt.xlabel("Total Charge (Photoelectrons)",fontsize=30)
    plt.title("Probability of total charge in PMTs\n"+\
            "GENIE-simulated CCQE events, WCSim detector simulation",fontsize=34)
    plt.show()
    #for xpixel in range(len(in_data[0])):
    #    for ypixel in range(len(in_data[0][xpixel])):
    #        #Get this pixels charge array (each element is a time interval in event)
    #        nopion_channel_vals = input_data[nopion_inds,xpixel,ypixel,0:pixel_chans]
    #        #Sum this channel's charge
    #        nopion_channel_sums = np.sum(nopion_channel_vals,axis=1)
    #        pion_channel_vals = input_data[pion_inds,xpixel,ypixel,0:pixel_chans]
    #        pion_channel_sums = np.sum(pion_channel_vals,axis=1)
    #        nopion_sums['channel_sum'].append(nopion_channel_sums)
    #        pion_mean['channel_avg'].append(pion_channel_average)
    #        nopion_mean['channel_stdev'].append(nopion_channel_stdev)
    #        pion_mean['channel_stdev'].append(pion_channel_stdev)

def ShowMeanMaps(in_data,out_data):
    nopion_mean = {'xpixel':[], 'ypixel':[], 'channel_avg':[], 'channel_stdev':[]}
    pion_mean = {'xpixel':[], 'ypixel':[], 'channel_avg':[], 'channel_stdev':[]}
     
    nopion_inds = None
    pion_inds = None
    if type(out_data[0]) == np.ndarray:
        if len(out_data[0]) > 0:
            nopion_inds = np.where(np.sum(out_data,axis=1) == 0)[0]
            pion_inds = np.where(np.sum(out_data,axis=1) > 0)[0]
    else:
        nopion_inds = np.where(out_data == 0)[0]
        pion_inds = np.where(out_data > 0)[0]
    for xpixel in range(len(in_data[0])):
        for ypixel in range(len(in_data[0][xpixel])):
            pixel_chans = len(in_data[0][xpixel][ypixel])
            nopion_mean['xpixel'].append(xpixel)
            nopion_mean['ypixel'].append(ypixel)
            pion_mean['xpixel'].append(xpixel)
            pion_mean['ypixel'].append(ypixel)
            nopion_channel_vals = in_data[nopion_inds,xpixel,ypixel,0:pixel_chans]
            nopion_channel_sums = np.sum(nopion_channel_vals,axis=1)
            nopion_channel_average = np.sum(nopion_channel_sums)/len(nopion_channel_sums)
            nopion_channel_stdev = np.std(nopion_channel_sums)
            pion_channel_vals = in_data[pion_inds,xpixel,ypixel,0:pixel_chans]
            pion_channel_sums = np.sum(pion_channel_vals,axis=1)
            pion_channel_average = np.sum(pion_channel_sums)/len(pion_channel_sums)
            pion_channel_stdev = np.std(pion_channel_sums)
            nopion_mean['channel_avg'].append(nopion_channel_average)
            pion_mean['channel_avg'].append(pion_channel_average)
            nopion_mean['channel_stdev'].append(nopion_channel_stdev)
            pion_mean['channel_stdev'].append(pion_channel_stdev)
    diff_mean = copy.deepcopy(pion_mean)
    diff_mean["channel_avg"] = np.array(pion_mean["channel_avg"]) - np.array(nopion_mean["channel_avg"])
    diff_mean["channel_stdev"] = np.sqrt(np.array(pion_mean["channel_stdev"])**2 + np.array(nopion_mean["channel_stdev"])**2)
    nopion_mean = pd.DataFrame(nopion_mean)
    pion_mean = pd.DataFrame(pion_mean)
    diff_mean = pd.DataFrame(diff_mean)
    
    nopm = nopion_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    pm = pion_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    diffm = diff_mean.pivot(index='ypixel',columns='xpixel',values='channel_avg')
    nops = nopion_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    ps = pion_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    diffs = diff_mean.pivot(index='ypixel',columns='xpixel',values='channel_stdev')
    sns.heatmap(nopm)
    plt.title("Average charge distribution of events \nwith no pion (channels summed)")
    plt.show()
    sns.heatmap(nops)
    plt.title("Std. dev. of charge distributions in events \nwith no pion (channels summed)")
    plt.show()
    sns.heatmap(pm)
    plt.title("Average charge distribution of events \nwith a pion (channels summed)")
    plt.show()
    sns.heatmap(ps)
    plt.title("Std. dev. of charge distributions in events \nwith a pion (channels summed)")
    plt.show()
    sns.heatmap(diffm)
    plt.title("Pion - No Pion average charge distribution (channels summed)")
    plt.show()
    sns.heatmap(diffs)
    plt.title("Pion - No Pion errors Std. Devs. added in quadrature (channels summed)")
    plt.show()

def ShowSingleEvent(input_data,event_index):
    #Let's make the first event's pixel map, summed over all the channels
    single_event = {'xpixel':[], 'ypixel':[], 'channel_sum':[]}
    for xpixel,val in enumerate(input_data[event_index]):
        for ypixel,pixel_chans in enumerate(val):
            single_event['xpixel'].append(xpixel)
            single_event['ypixel'].append(ypixel)
            single_event['channel_sum'].append(np.sum(val[ypixel]))
    single_event = pd.DataFrame(single_event)
    se = single_event.pivot(index='ypixel',columns='xpixel',values='channel_sum')
    sns.heatmap(se)
    plt.title("Example event (channels summed)")
    plt.show()

def ShowPionValidationPlots(predictions,truths):
    diffs = predictions-truths
    #pidiffs = {"piplus_diff": diffs[0:len(diffs),0], "pi0_diff": diffs[0:len(diffs),1],
    #        "piminus_diff": diffs[0:len(diffs),2], "piplus_truth": truths[0:len(truths),0],
    #        "pi0_truth": truths[0:len(truths),1],"piminus_truth": truths[0:len(truths),2]}
    pidiffs = {"pi0_diff": diffs[0:len(diffs),1],"piminus_diff": diffs[0:len(diffs),2], 
            "pi0_truth": truths[0:len(truths),1],"piminus_truth": truths[0:len(truths),2],
            "pi0_predict": predictions[0:len(predictions),1],
            "piminus_predict": predictions[0:len(predictions),2]}
    pidiffs = pd.DataFrame(pidiffs)

   
    #Let's first get indices of entries with no pions
    nopion_ind = np.where(pidiffs["pi0_truth"] + pidiffs["piminus_truth"] == 0)[0]
    
    pion_ind = np.where(pidiffs["pi0_truth"] + pidiffs["piminus_truth"] > 0)[0]
    piminus_ind = np.where(pidiffs["piminus_truth"] > 0)[0]

    pi0_count = pidiffs["pi0_truth"]
    pi0_ind = np.where(pi0_count > 0)[0]

    total_pioncountdiff = pidiffs["piminus_diff"] + pidiffs["pi0_diff"]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(pidiffs["pi0_predict"][nopion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{0}$ prediction', histtype="step", color='red')
    plt.hist(pidiffs["piminus_predict"][nopion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{-}$ prediction', histtype="step", color='blue')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction of pion presence with no pions produced")
    plt.show()

    #Let's first get indices of entries with pions

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(pidiffs["pi0_predict"][pion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{0}$ prediction', histtype="step", color='red')
    plt.hist(pidiffs["piminus_predict"][pion_ind], range=(0,1), bins=50,
            linewidth=4, label='$\pi^{-}$ prediction', histtype="step", color='blue')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction of pion presence for events with a pion")
    plt.show()

    predicts_wpions = pd.DataFrame({"pi0p":pidiffs["pi0_predict"][pion_ind],
        "pimp":pidiffs["piminus_predict"][pion_ind]})
    fig = plt.figure()
    g = sns.jointplot(x="pi0p", y="pimp", data=predicts_wpions,kind="hex",stat_func=None)
    g = g.set_axis_labels("$\pi^{0}$ count prediction", "$\pi^{-}$ count prediction")
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.9,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.85,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle("Distribution of $\pi^{0}$ and $\pi^{-}$ predictions for events with a pion")
    plt.show()

    plt.hist(pidiffs["pi0_diff"][pi0_ind],bins=30,color='red',alpha=0.8)
    plt.title("Prediction-Truth for count of $\pi^{0}$ in events with $\pi^{0}$")
    plt.show()

    plt.hist(pidiffs["piminus_diff"][piminus_ind],bins=30,color='green',alpha=0.8)
    plt.title("Prediction-Truth for count of $\pi^{-}$ in events with $\pi^{-}$")
    plt.show()

    plt.hist(total_pioncountdiff[nopion_ind],bins=30,color='purple',alpha=0.8)
    plt.title("Sum of $\pi^{0}$ and $\pi^{-}$ Prediction-Truth for no-$\pi$ events")
    plt.show()

def ShowRecoValidationPlots(predictions,truths):
    diffs = predictions-truths
    thediffs = {"posX": diffs[0:len(diffs),0],"posY": diffs[0:len(diffs),1], 
            "posZ": diffs[0:len(diffs),2],"dirX": diffs[0:len(diffs),3],
            "dirY": diffs[0:len(diffs),4],
            "dirZ": diffs[0:len(diffs),5]}
            #"time": diffs[0:len(diffs),6]}
    thediffs = pd.DataFrame(thediffs)
    plt.hist(thediffs["posX"],bins=30,color='red',alpha=0.8)
    plt.title("Prediction-Truth for muon X position in detector (m)")
    plt.show()
    plt.hist(thediffs["posY"],bins=30,color='red',alpha=0.8)
    plt.title("Prediction-Truth for muon Y position in detector (m)")
    plt.show()
    plt.hist(thediffs["posZ"],bins=30,color='red',alpha=0.8)
    plt.title("Prediction-Truth for muon Z position in detector (m)")
    plt.show()
    plt.hist(thediffs["dirX"],bins=30,color='blue',alpha=0.8)
    plt.title("Prediction-Truth for muon direction's X-component in detector")
    plt.show()
    plt.hist(thediffs["dirY"],bins=30,color='blue',alpha=0.8)
    plt.title("Prediction-Truth for muon direction's Y-component in detector")
    plt.show()
    plt.hist(thediffs["dirZ"],bins=30,color='blue',alpha=0.8)
    plt.title("Prediction-Truth for muon directions' Z-component in detector")
    plt.show()
    #plt.hist(thediffs["time"],bins=30,color='green',alpha=0.8)
    #plt.title("Pblueiction-Truth for muon vertex time in detector (ns)")
    #plt.show()
   

def ShowRingValidationPlots(predictions,truths):
    diffs = predictions-truths
    #pidiffs = {"piplus_diff": diffs[0:len(diffs),0], "pi0_diff": diffs[0:len(diffs),1],
    #        "piminus_diff": diffs[0:len(diffs),2], "piplus_truth": truths[0:len(truths),0],
    #        "pi0_truth": truths[0:len(truths),1],"piminus_truth": truths[0:len(truths),2]}
    print(diffs)
    ringdiffs = {"ring_diff": diffs,"ring_truth": truths, 
            "ring_predict": predictions}
    ringdiffs = pd.DataFrame(ringdiffs)

   
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(ringdiffs["ring_truth"],  bins=50,
            linewidth=4, label='Truth', histtype="step", color='red')
    plt.hist(ringdiffs["ring_predict"], bins=50,
            linewidth=4, label='Prediction', histtype="step", color='blue')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Truth vs. Prediction of Single Ring (0) vs. Multi Ring (1) ")
    plt.xlabel("Single ring classifier (0=single ring, 1= multi-ring)")
    plt.show()

    #Let's first get indices of entries with pions
    has_pion_inds = np.where(ringdiffs["ring_truth"]>0)[0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(ringdiffs["ring_diff"][has_pion_inds],  bins=50,
            linewidth=4, label='Prediction-Truth', histtype="step", color='green')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction-Truth of Single Ring (0) vs. Multi Ring (1) for events with pions")
    plt.show()

    #Let's first get indices of entries with pions
    nopion_inds = np.where(ringdiffs["ring_truth"]==0)[0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.hist(ringdiffs["ring_diff"][nopion_inds],  bins=50,
            linewidth=4, label='Prediction-Truth', histtype="step", color='green')
    leg = ax.legend(loc=4,fontsize=24)
    leg.set_frame_on(True)
    leg.draw_frame(True)
    plt.title("Prediction-Truth of Single Ring (0) vs. Multi Ring (1) for events without pions")
    plt.show()


