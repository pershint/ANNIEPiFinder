import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns

class LikelihoodCalculator(object):
    def __init__(self):
        self.PDFs = {}

    def RemovePDF(self,PDF_name):
        '''
        Remove this PDF from those used to calculate the likelihood.
        Input: PDF_name [string]
        '''
        if PDF_name not in self.PDFs.keys():
            print("PDF not loaded into likelihood function.")
        else:
            del self.PDFs[PDF_name]

    def Add1DPDF(self,PDF_name,datatype,label,x,y,weight=1.0):
        '''
        Adds information for a on3-dimensional PDF for use in the
        likelihood function.  xlabel and ylabel must correspond to
        labels in the DataFrame given to the GetLikelihood() method.
    
        Inputs:
            PDF_name [string]
                Label for this particular PDF
            datatype [string]
                PDF is associated with either signal (muon only) or background
                (muon + pions).  Allowed inputs are "S" or "B".
            label, [string]
                Labels associated with data in dataset that will be evaluated 
                in GetLikelihood() method.
            x,y [array]
                Meshgrid relating data values (x) to the height of the KDE contour (y)
        '''
        if PDF_name in self.PDFs.keys():
            print("PDF name already exists! Please pick a new name or remove the present PDF.")
            return
        self.PDFs[PDF_name] = {}
        self.PDFs[PDF_name]['x_label'] = xlabel
        self.PDFs[PDF_name]['PDF_data'] = [x,y]
        self.PDFs[PDF_name]["S_or_B"] = datatype
        self.PDFs[PDF_name]["weight"] = weight
        self.PDFs[PDF_name]["dimension"] = 1

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
        self.PDFs[PDF_name]["dimension"] = 2

    def _Calculate2DLikelihoods(self,data,PDF_info,S_likelihood,B_likelihood,PDF_interpolation):
        PDF_data = PDF_info['PDF_data']
        PDF_weight = PDF_info['weight']
        f = interpolate.interp2d(PDF_data[0],PDF_data[1],PDF_data[2],kind=PDF_interpolation)
        x_data = data[PDF_info['x_label']]
        y_data = data[PDF_info['y_label']]
        if type(x_data[0]) == np.ndarray and type(y_data[0]) == np.ndarray:
            #Loop through data entries and for each entry, calculate this
            #PDF's contribution to the S or B likelihood
            for j,entry in enumerate(x_data):
                ev_xdata = x_data[j]
                ev_ydata = y_data[j]
                if len(ev_xdata) == 0: continue
                likelihood = f(ev_xdata,ev_ydata)
                likelihood = np.sum(likelihood)/len(likelihood)
                if PDF_info["S_or_B"] == "S":
                    S_likelihood[j]+=likelihood*PDF_weight
                elif PDF_info["S_or_B"] == "B":
                    B_likelihood[j]+=likelihood*PDF_weight
        else:
            #Loop through data entries and for each entry, calculate this
            #PDF's contribution to the S or B likelihood
            for k,entry in enumerate(x_data):
                ev_x= x_data[k]
                ev_y = y_data[k]
                likelihood = f(ev_x,ev_y)
                if PDF_info["S_or_B"] == "S":
                    S_likelihood[k]+=likelihood*PDF_weight
                elif PDF_info["S_or_B"] == "B":
                    B_likelihood[k]+=likelihood*PDF_weight
        return S_likelihood,B_likelihood

    def _Calculate1DLikelihoods(self,data,PDF_info,S_likelihood,B_likelihood,PDF_interpolation):
        PDF_data = PDF_info['PDF_data']
        PDF_weight = PDF_info['weight']
        f = interpolate.interp1d(PDF_data[0],PDF_data[1],kind=PDF_interpolation)
        x_data = data[PDF_info['x_label']]
        if type(x_data[0]) == np.ndarray and type(y_data[0]) == np.ndarray:
            #Loop through data entries and for each entry, calculate this
            #PDF's contribution to the S or B likelihood
            for j,entry in enumerate(x_data):
                ev_xdata = x_data[j]
                if len(ev_xdata) == 0: continue
                likelihood = f(ev_xdata)
                likelihood = np.sum(likelihood)/len(likelihood)
                if PDF_info["S_or_B"] == "S":
                    S_likelihood[j]+=likelihood*PDF_weight
                elif PDF_info["S_or_B"] == "B":
                    B_likelihood[j]+=likelihood*PDF_weight
        else:
            #Loop through data entries and for each entry, calculate this
            #PDF's contribution to the S or B likelihood
            for k,entry in enumerate(x_data):
                ev_x= x_data[k]
                likelihood = f(ev_x)
                if PDF_info["S_or_B"] == "S":
                    S_likelihood[k]+=likelihood*PDF_weight
                elif PDF_info["S_or_B"] == "B":
                    B_likelihood[k]+=likelihood*PDF_weight
        return S_likelihood,B_likelihood


    def GetLikelihoods(self,test_data,PDF_interpolation='linear'):
        '''
        Use data PDFs loaded and calculate the likelihood each
        event has only a muon or a muon + pions
        '''
        likelihoods = []
        S_likelihood = np.zeros(len(test_data["entrynum"]))
        B_likelihood = np.zeros(len(test_data["entrynum"]))
        #Loop through 2D PDFs in class and calculate their likelihood contributions.
        #TODO: We should be able to restructure our inputs in a way to generalize to N-Dim. PDFs
        for PDF in self.PDFs:
            if self.PDFs[PDF]["dimension"] == 1:
                S_likelihood,B_likelihood = self._Calculate1DLikelihoods(test_data,self.PDFs[PDF],S_likelihood,B_likelihood,PDF_interpolation)
            if self.PDFs[PDF]["dimension"] == 2:
                S_likelihood,B_likelihood = self._Calculate2DLikelihoods(test_data,self.PDFs[PDF],S_likelihood,B_likelihood,PDF_interpolation)
        Signal_likelihood = (S_likelihood)/(S_likelihood + B_likelihood)
        return Signal_likelihood

    def OptimizeCut(self,S_likelihoods,B_likelihoods,l_range=[0,1],l_interval=0.01):
        l_cuts = np.arange(l_range[0],l_range[1],l_interval)
        efficiencies = []
        purities = []
        lcuts_checked = []
        max_significance = 0
        max_sig_cut = None
        for l_cut in l_cuts:
            S_pass = float(len(np.where(S_likelihoods>l_cut)[0]))
            B_pass = float(len(np.where(B_likelihoods>l_cut)[0]))
            N0 = S_pass + B_pass
            if (N0 == 0):
                print("NO SIGNAL OR BACKGROUND AT CUT %f.  CONTINUING"%(l_cut))
                continue
            lcuts_checked.append(l_cut)
            eff = S_pass/len(S_likelihoods)
            purity = S_pass/(S_pass + B_pass)
            efficiencies.append(eff)
            purities.append(purity)

            #significance = eff * (purity**2)
            #significance = np.sqrt(2*N0*np.log(1+(S_pass/B_pass))-2*S_pass) 
            significance = S_pass / np.sqrt(N0)
            print("SIGNIFICANCE AT CUT %f IS %f "%(l_cut,significance))
            if max_sig_cut is None:
                max_sig_cut = l_cut
                max_significance = significance
            elif significance > max_significance:
                max_sig_cut = l_cut
                max_significance = significance
        print("USING EFFICIENCY*PURITY^2, MAX SIGNIFICANCE HAPPENS AT CUT %f"%(max_sig_cut))
        return max_sig_cut, lcuts_checked, efficiencies, purities


