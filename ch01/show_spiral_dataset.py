import sys
sys.path.append('..')
from dataset import spiral
import matplotlib.pyplot as plt
import numpy as np

x, t = spiral.load_data()
#print('x', x.shape)
#print('t', t.shape)
t = np.argmax(t, axis=1)
for i in range(3):
    plt.scatter(x[t==i][:,0], x[t==i][:,1])
plt.show()