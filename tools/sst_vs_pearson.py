import banpei
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

matplotlib.use('TKAgg')
data1 = pd.read_csv('tools\periodic_waves.csv')['y']
n = len(data1)
model = banpei.SST(w=50)
results1 = model.detect(data1)
ax1 = plt.subplot(8, 1, 1)
plt.sca(ax1)
plt.plot([t for t in range(0, n)], data1)
ax2 = plt.subplot(8, 1, 3)
plt.sca(ax2)
plt.plot([t for t in range(0, n)], results1)
data2 = [t for t in range(0, 530, 1)] + [530] * 500 + [t for t in range(530, -410, -2)]
ax3 = plt.subplot(8, 1, 5)
plt.sca(ax3)
plt.plot([t for t in range(0, n)], data2)
model = banpei.SST(w=50)
results2 = model.detect(data2)
ax4 = plt.subplot(8, 1, 7)
plt.sca(ax4)
plt.plot([t for t in range(0, n)], results2)
r = pearsonr(data1, data2)
print(r)
plt.show()
