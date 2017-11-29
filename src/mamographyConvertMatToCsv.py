import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('Data/Mamography/Input/mammography.mat')

data = mat_contents['X']
labels = mat_contents['y'].astype(int)

fUnder = open("Data/Mamography/Input/mammography.csv", "w+")

for samples in np.column_stack((data, labels)):
    fUnder.write((',').join(str(l) for l in samples) + '\n')
fUnder.close()

