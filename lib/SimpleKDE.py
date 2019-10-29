import sklearn.neighbors as skn
import sklearn.grid_search as sgs
import sklearn.cross_validation as cv
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#for example run
#import AnnieHeatMap as ahm

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
        Essentially, for a single bandwidth, a PDF is made with all 
        points except one, and the unused point is tested against the model. 
        This is done many times, and an average error is computed.  This is
        done for each bandwidth, and the bandwidth with the lowest average
        error is returned.
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
                            cv=cv.LeaveOneOut(len(data)),
                            verbose=1)
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

    def KDEEstimate2D(self,bandwidth,datalabelx,datalabely,xbins=100j,ybins=100j,
            x_range=[0,1], y_range=[0,1],kern='gaussian'):
        '''
        Performs a 2D Kernel density estimation using data from the two variables
        specified in datalabelx and datalabely.  x and y-ranges assume data has
        been normalized and have the full range from [0,1].
        '''
        datax = None
        datay = None
        try:
            datax = self.df[datalabelx]
            datay = self.df[datalabely]
        except KeyError:
            print("No data found for one of these datalabels.")
            return
        range_x_ind = np.where((datax>x_range[0]) & (datax<x_range[1]))[0]
        range_y_ind = np.where((datay>y_range[0]) & (datay<y_range[1]))[0]
        print("RANGE_X: " + str(range_x_ind))
        print("RANGE_Y_IND: " + str(range_y_ind))
        range_indices = np.array(list(set(np.concatenate((range_x_ind,range_y_ind)))))
        print("RANGE_INDICES: " + str(range_indices))
        datax = datax[range_indices]
        datay = datay[range_indices]
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
    #mymap = ahm.AnnieHeatMapMaker(rootfiles=['PMTReco_05202019.root'])
    #mymap.load_dataframe()
    #miniframe = mymap.MakeYThetaDataFrame()
    #sns.kdeplot(miniframe["Theta"],miniframe["Y"], shade=True)
    #plt.show()
    #myestimator = KernelDensityEstimator(dataframe=miniframe)
    #xx,yy,zz = myestimator.KDEEstimate2D(300,'Y','Theta',xbins=100j,ybins=100j)
