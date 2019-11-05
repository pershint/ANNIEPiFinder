import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style("whitegrid")
sns.axes_style("darkgrid")
xkcd_colors =  [ 'slate blue', 'green', 'grass','pink']
sns.set_context("poster")
sns.set_palette(sns.xkcd_palette(xkcd_colors))

def Plot2DDataDistribution(data_frame,xlabel,ylabel,xtitle,ytitle,title):
    g = sns.jointplot(xlabel,ylabel,data=data_frame,kind="hex",
            joint_kws=dict(gridsize=80),
            stat_func=None).set_axis_labels(xtitle,ytitle)
    plt.subplots_adjust(left=0.2,right=0.8,
            top=0.85,bottom=0.2)
    cbar_ax = g.fig.add_axes([0.84,0.2,0.05,0.62])
    plt.colorbar(cax=cbar_ax)
    g.fig.suptitle(title)
    plt.show()

def PlotKDE(xx,yy,zz,numcontours,xtitle,ytitle,title):
    cbar = plt.contourf(xx,yy,zz,40,cmap='inferno')
    plt.colorbar()
    plt.ylabel(xtitle)
    plt.xlabel(ytitle)
    plt.title(title)
    plt.show()

