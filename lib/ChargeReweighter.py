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

    def ScaleCharge(self, charges, mu_positions, mu_directions,pmt_positions, pmt_directions):
        #Change coordinates so muon position is at origin
        pmt_positions = pmt_positions - mu_positions
        dirs_to_pmt = 
        #oh jebus
        alphas = self._dot(mu_directions,pmt_positions)

        scaled_charges = charges/(0.5(1 + cosine_phis))
        return scaled_charges

