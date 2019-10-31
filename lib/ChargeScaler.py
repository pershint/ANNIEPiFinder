import numpy as np

class ChargeScaler(object):
    '''
    Class is designed to re-scale any PMT's total charge by the 
    total exposed surface area that would be visible 
    from the muon track position that Cherenkov light would be emitted from to 
    hit that PMT.
    '''

    def __init__(self):
        print("WELCOME TO THE SCALER.  IT'S PARTY TIME")
        self.thetac = (42*np.pi)/180  #Cherenkov angle in water in radians
        self.y_cut = 130 #In centimeters, assume all PMTs at abs(Y) > y_cut face towards the 
                    #center of the cylinder

    def _EstimateDirection(self,pmt_positions):
        '''
        Assuming PMTs are mounted on a cylindrical structure facing 
        normal to cylinder & inward, use PMT positions to estimate
        the PMT directions.  PMT coordinates must be given relative
        to the center of the cyliner.
        '''
        pmt_directions = []
        pmt_ys = np.array(pmt_positions)[:,1]
        bot_ind = np.where(abs(pmt_ys)<-self.y_cut)[0]
        top_ind = np.where(abs(pmt_ys)>self.y_cut)[0]
        for j in range(len(pmt_positions)):
            if j in bot_ind:
                pmt_directions.append([0,1,0])
            elif j in top_ind:
                pmt_directions.append([0,-1,0])
            else:
                pos_mag = self._mag(np.array(pmt_positions[j]))
                norm_position = np.array(pmt_positions[j])/pos_mag
                pmt_dir_unnorm = np.array([-1*norm_position[0], 0, -1*norm_position[2]])
                pmt_dir_norm = pmt_dir_unnorm/self._mag(pmt_dir_unnorm)
                pmt_directions.append(list(pmt_dir_norm))
        return np.array(pmt_directions)


    def ScaleFactors(self, mu_position, mu_direction,pmt_positions):
        #Change coordinates so muon position is at origin
        pmt_directions = self._EstimateDirection(pmt_positions)
        pmt_positions = pmt_positions - mu_position
        mag_pmtpos = self._mag(pmt_positions)
        dir_topmt = pmt_positions/(mag_pmtpos.reshape(len(mag_pmtpos),1))

        #Get the angle between the muon and 
        alphas = self._anglebw(mu_direction,dir_topmt)
        mag_mutracks = self._mag(pmt_positions)*np.sin(self.thetac-alphas)/(np.sin(np.pi-self.thetac))
        mag_ltracks = self._mag(pmt_positions)*np.sin(alphas)/(np.sin(np.pi-self.thetac))
        lhats = (1/mag_ltracks.reshape(len(mag_ltracks),1))*(pmt_positions - 
                mag_mutracks.reshape(len(mag_mutracks),1)*mu_direction)
        cosine_phis = -self._dot(lhats,pmt_directions)
        scale_factor = (0.5*(1 + cosine_phis))
        noscaling_inds = np.where(alphas>52)[0] #Buffer of 10 degrees uncertainty
        scale_factor[noscaling_inds] = 1
        return scale_factor

    def _mag(self,vec):
        if len(vec.shape) == 1:
            return np.sqrt(np.sum(vec**2))
        else: 
            return np.sqrt(np.sum(vec**2,axis=1))

    def _dot(self,vec1,vec2):
        return np.sum((vec1*vec2),axis=1)

    def _anglebw(self,a,b):
        dotprod = self._dot(a,b)
        mag_a = self._mag(a)
        mag_b = self._mag(b)
        angle = np.arccos(dotprod/(mag_a*mag_b))
        return angle
