import numpy as np

w=np.array((1,2,3,7))
c=np.concatenate([w,[w[-1]+1]])
print(c)