import numpy as np

def ReverseRegularizePosDir(output_data):
    '''Reverses regularization of position and direction information'''
    positions = output_data[:,0:3]
    directions = output_data[:,3:6]
    #time = output_data[:,6:7]
    positions = (positions*100)-100.
    directions = (2*directions) - 1.
    #time = time - 0.5
    procd_output = np.append(positions,directions,axis=1)
    return procd_output

def RegularizePosDir(output_data):
    '''Assumes seven inputs in the following order: trueVtxX,
    trueVtxY, trueVtxZ, trueDirX, trueDirY, trueDirZ, trueVtxTime.
    Transforms variables to reside basically between zero and one.'''
    positions = output_data[:,0:3]
    directions = output_data[:,3:6]
    #time = output_data[:,6:7]
    positions = (positions+100.)/100.
    directions = (directions+1.)/2.
    #time = time + 0.5
    procd_output = np.append(positions,directions,axis=1)
    #procd_output = np.append(procd_output,time,axis=1)
    return procd_output
            
