import pandas as pd
import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

df = pd.read_csv("CSV/0001.CSV")
density, bins = np.histogram(df.iloc[:,0], bins=30)
peaks, properties = find_peaks(density)
troughs = argrelextrema(density, np.greater)
density = list(density)
x_peaks = []
y_peaks = []
for i in range(0,len(troughs)):
    y_peaks.append(density[peaks[i]])

for i in range(0,len(y_peaks)):
    x_peaks.append(bins[density.index(y_peaks[i])])

counts, bins, _ = plt.hist(df.iloc[:,0], bins=30, histtype='step', color='white')
d = gaussian_kde(counts)
bins = list(bins)
del bins[-1]
plt.plot(bins, counts)
for i in range(0,len(x_peaks)):
    plt.axvline(x_peaks[i], linestyle='--')
plt.show()

