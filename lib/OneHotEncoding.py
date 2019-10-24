import numpy as np

def OneHotEncodeBinaryArgs(output_data):
    '''
    Takes in an array of outputs and one-hot encodes the
    output data.  A single output is assumed to be of arbitrary length,
    but each entry in the output could be either zero or one.  

    Example where an entry in output data has pi0 and/or piminus only:

    0 0             1 0               0 1             1 1
     |               |                 |               |
     V               V                 V               V
    1 0 0 0        0 1 0 0           0 0 1 0         0 0 0 1
    '''
    numoutputs = len(output_data[0])
    num_onehots = (numoutputs)**2
    onehot_output = []
    for j,entry in enumerate(output_data): #For each entry, put a 1 in the proper onehot position
        onehot_ind = None
        this_onehotoutput = np.zeros(num_onehots)
        entry = np.array(entry)
        num_pions = np.sum(entry)
        if num_pions > numoutputs:
            print("OH SHOOT, THIS EVENT HAD TWO PIONS OF A SINGLE TYPE")
            print("TRAIN IT AS IF ONE OF EACH TYPE FOR NOW...")
            onehot_ind = num_onehots - 1
        elif num_pions == 0:
           onehot_ind = 0 
        elif num_pions == numoutputs:
            onehot_ind = num_onehots-1
        else:  #Only one pion was produced of some type, and we must identify which
            onehot_mask = np.zeros(numoutputs)
            onehot_ind = 1
            onehot_mask[0] = 1
            num_rolls = 0
            have_match = False
            while not have_match:
                if np.sum(onehot_mask*entry)== np.sum(entry):
                    have_match = True
                elif num_rolls == numoutputs - 1:
                    onehot_ind +=1
                    onehot_mask[0] = 1
                    num_rolls = 0
                else:
                    onehot_ind +=1
                    onehot_mask = np.roll(onehot_mask,1)
                    num_rolls+=1
        this_onehotoutput[onehot_ind] = 1
        onehot_output.append(this_onehotoutput)
    return np.array(onehot_output)

