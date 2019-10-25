import sklearn.neighbors as skn
import sklearn.grid_search as sgs
import sklearn.cross_validation as cv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#for example run
import AnnieHeatMap as ahm

class KernelDensityEstimator(object):
    def __init__(self,dataframe=None):
        self.df = dataframe
        self.bandwidths = {}

    def SetDataFrame(self,dataf):
        self.df = dataf

    def ClearBandwidths(self):
        self.bandwidths={}

    def SetBandwidth(self,datalabel,bw):
        '''Set the bandwidth to use for the input datalabel.
        Args:
            datalabel: string
                string for the label of the dataframe data to associate bandwidth to
            bw: float
                bandwidth that will be used by KDE methods
        '''

        self.bandwidths[datalabel] = bw

    def GetOptimalBandwidth(self,datalabel,bandlims,numbands):
        '''Optimize the bandwidth using leave-one-out cross-validation.
        Example follows that at jakevdp.github.io/PythonDataScienceHandbook.
        Args
            datalabel: string
                string describing which datalabel in the dataframe to find
                the bandwidth for
            bandlims: array (length 2)
                limits to search for the optimal bandwidth in
            numbands: int
                ints 
        '''
        if bandlims[1] < 0 or bandlims[0] < 0:
            print("Bandwidth must be greater than zero")
            return
        bandwidths = np.linspace(bandlims[0],bandlims[1],numbands)
        data = self.df[datalabel]
        if isinstance(self.df[datalabel][0],np.ndarray):
            print("WERE IN HERE")
            data_arr = []
            for i in range(len(self.df[datalabel])):
                data_arr = data_arr + list(self.df[datalabel][0])
            data=np.array(data_arr)
        if len(data)>500:
            print("This may take some time depending on your data length.")
            print("numbands > 10 with len(data)>500 starts to take a bit")
        grid = sgs.GridSearchCV(skn.KernelDensity(kernel='gaussian'),
                            {'bandwidth': bandwidths},
                            cv=cv.LeaveOneOut(len(data)))
        grid.fit(data[:,None])
        thebandwidth = grid.best_params_['bandwidth']
        return thebandwidth

    def KDEEstimate1D(self,datalabel,xlims=None,kern='gaussian'):
        bandwidth = None
        data = None
        try:
            bandwidth = self.bandwidths[datalabel]
            data = self.df[datalabel]
        except KeyError:
            print("No bandwidth or data found for this datalabel.")
            return
        print(data)
        if isinstance(self.df[datalabel][0],np.ndarray):
            print("WERE IN HERE")
            data_arr = []
            for i in range(len(self.df[datalabel])):
                data_arr = data_arr + list(self.df[datalabel][0])
            data=np.array(data_arr)
        if xlims is None:
            xlims = [np.min(data),np.max(data)]
        linspace = np.linspace(xlims[0],xlims[1], (xlims[1]-xlims[0])*100.)
        kde = skn.KernelDensity(bandwidth=bandwidth, kernel=kern)
        kde.fit(self.df[datalabel][:,None])
        logp = kde.score_samples(linspace[:,None])
        return linspace, np.exp(logp)

    def KDEEstimate2D(self,bandwidth,datalabelx,datalabely,xbins=100j,ybins=100j,kern='gaussian'):
        datax = None
        datay = None
        try:
            datax = self.df[datalabelx]
            datay = self.df[datalabely]
        except KeyError:
            print("No data found for one of these datalabels.")
            return
        if isinstance(self.df[datalabelx][0],np.ndarray):
            print("WERE IN HERE")
            datax_arr = []
            for i in range(len(self.df[datalabelx])):
                datax_arr = datax_arr + list(self.df[datalabelx][0])
            datax=np.array(datax_arr)
        if isinstance(self.df[datalabely][0],np.ndarray):
            print("WERE IN HERE")
            datay_arr = []
            for i in yrange(len(self.df[datalabely])):
                datay_arr = datay_arr + list(self.df[datalabely][0])
            datay=np.array(datay_arr)
        xx, yy = np.mgrid[datax.min():datax.max():xbins,
                datay.min():datay.max():ybins]

        xy_grid = np.vstack([yy.ravel(),xx.ravel()]).T
        xy_dataset = np.vstack([datay,datax]).T
        TwoDKDE = skn.KernelDensity(bandwidth=bandwidth,kernel=kern)
        TwoDKDE.fit(xy_dataset)

        z = np.exp(TwoDKDE.score_samples(xy_grid))

        return xx,yy,np.reshape(z,xx.shape)

if __name__=='__main__':
    print("WOO")
    mymap = ahm.AnnieHeatMapMaker(rootfiles=['PMTReco_05202019.root'])
    mymap.load_dataframe()
    miniframe = mymap.MakeYThetaDataFrame()
    sns.kdeplot(miniframe["Theta"],miniframe["Y"], shade=True)
    plt.show()
    myestimator = KernelDensityEstimator(dataframe=miniframe)
    xx,yy,zz = myestimator.KDEEstimate2D(300,'Y','Theta',xbins=100j,ybins=100j)
    #print("WOO")
    #mymap = ahm.AnnieHeatMapMaker(rootfiles=['RPTest_100.root'])
    #mymap.load_dataframe()
    #mymap.AddHitAnglesToDataFrame()
    #miniframe = mymap.MakePMTHitChargeDataFrame()
    #myestimator = KernelDensityEstimator(dataframe=miniframe)
    #xx,yy,zz = myestimator.KDEEstimate2D(8.0,'hitCharges','hitAngles',xbins=1000j,
    #                                     ybins=1000j)
