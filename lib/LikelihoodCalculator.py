import numpy as np
from scipy import interpolate

class LikelihoodFunction(object):
    def __init__(self):
        self.PDFs = {}

    def Remove2DPDF(self,PDF_name):
        '''
        Remove this PDF from those used to calculate the likelihood.
        Input: PDF_name [string]
        '''
        if PDF_name not in self.PDFs.keys():
            print("PDF not loaded into likelihood function.")
        else:
            del self.PDFs[PDF_name]

    def Add2DPDF(self,PDF_name,datatype,xlabel,ylabel,xx,yy,zz,weight=1.0):
        '''
        Adds information for a two-dimensional PDF for use in the
        likelihood function.  xlabel and ylabel must correspond to
        labels in the DataFrame given to the GetLikelihood() method.
    
        Inputs:
            PDF_name [string]
                Label for this particular PDF
            datatype [string]
                PDF is associated with either signal (muon only) or background
                (muon + pions).  Allowed inputs are "S" or "B".
            xlabel,ylabel [string]
                Labels associated with data in dataset that will be evaluated 
                in GetLikelihood() method.
            xx,yy,zz [array]
                Meshgrid relating x-axis and y-axis points to the height of the 
                contour
        '''
        if PDF_name in self.PDFs.keys():
            print("PDF name already exists! Please pick a new name or remove the present PDF.")
            return
        self.PDFs[PDF_name] = {}
        self.PDFs[PDF_name]['x_label'] = xlabel
        self.PDFs[PDF_name]['y_label'] = ylabel
        self.PDFs[PDF_name]['PDF_data'] = [xx,yy,zz]
        self.PDFs[PDF_name]["S_or_B"] = datatype
        self.PDFs[PDF_name]["weight"] = weight

    def GetLikelihoods(self,test_data,PDF_interpolation='linear'):
        '''
        Use data PDFs loaded and calculate the likelihood each
        event has only a muon or a muon + pions
        '''
        likelihoods = []
        S_likelihood = np.zeros(len(test_data["entrynum"]))
        B_likelihood = np.zeros(len(test_data["entrynum"]))
        #Loop through 2D PDFs in class and calculate their likelihood contributions.
        for PDF in self.PDFs:
            PDF_data = self.PDFs[PDF]['PDF_data']
            PDF_weight = self.PDFs[PDF]['weight']
            f = interpolate.interp2d(PDF_data[0],PDF_data[1],PDF_data[2],kind=PDF_interpolation)
            x_data = test_data[self.PDFs[PDF]['x_label']]
            y_data = test_data[self.PDFs[PDF]['y_label']]
            if type(x_data[0]) == np.ndarray and type(y_data[0]) == np.ndarray:
                #Loop through data entries and for each entry, calculate this
                #PDF's contribution to the S or B likelihood
                for j,entry in enumerate(x_data):
                    ev_xdata = x_data[j]
                    ev_ydata = y_data[j]
                    if len(ev_xdata) == 0: continue
                    likelihood = f(ev_xdata,ev_ydata)
                    likelihood = np.sum(likelihood)/len(likelihood)
                    if self.PDFs[PDF]["S_or_B"] == "S":
                        S_likelihood[j]+=likelihood*PDF_weight
                    elif self.PDFs[PDF]["S_or_B"] == "B":
                        B_likelihood[j]+=likelihood*PDF_weight
            else:
                #Loop through data entries and for each entry, calculate this
                #PDF's contribution to the S or B likelihood
                for k,entry in enumerate(x_data):
                    ev_x= x_data[k]
                    ev_y = y_data[k]
                    likelihood = f(ev_x,ev_y)
                    if self.PDFs[PDF]["S_or_B"] == "S":
                        S_likelihood[k]+=likelihood*PDF_weight
                    elif self.PDFs[PDF]["S_or_B"] == "B":
                        B_likelihood[k]+=likelihood*PDF_weight
        Signal_likelihood = (S_likelihood)/(S_likelihood + B_likelihood)
        return Signal_likelihood
