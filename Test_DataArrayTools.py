#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#

import numpy as np
from DataArrayTools import ShitftAmount,TrimArray


TestList = np.array([[0,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9]]);

print("Data Start\n")
print(TestList)
print("\n")

TestList2 = ShitftAmount(TestList,3)

print("DataShifted and Trimed  Y\n")
print(TestList2)
print("\n")


print("Data Trimed   X \n")
TestList = TrimArray(TestList,-3)
print(TestList)
print("\n")
