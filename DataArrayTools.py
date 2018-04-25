#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#


import numpy as np

def TrimArray(Inputarray, TrimAmount):
    #TrimAmount sign determines direction
    #Trim array to remove elements that should have been shifted off the end into nothing. 
    if(TrimAmount > 0):
        Inputarray = Inputarray[TrimAmount:]
    elif(TrimAmount < 0):
       Inputarray = Inputarray[:TrimAmount] 
    return Inputarray


def ShitftAmount(Inputarray, shiftamount):
    #This will barrel shift the input array by the given amount. It will shift only the other array! 
    #The shifted array is then trimed by the shiftamount 
    #Do a barrel roll!
    #Outarray = np.roll(Inputarray,shiftamount,axis=0)

    return TrimArray(Inputarray, shiftamount)