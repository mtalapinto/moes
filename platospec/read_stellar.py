import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stellarpath = 'C:\\Users\\marce\\Documents\\fondecyt_uai\\fideos_moes\\stellar_spectrum\\'
filename = 'lte05700-3.00+1.0.7.dat'

#hdu = open(stellarpath+filename,'r')
data = pd.read_csv(stellarpath+filename, sep=',')

wavemin = 3500.
wavemax = 9000.
data = data.loc[data['wave'].values < wavemax]
data = data.loc[data['wave'].values > wavemin]
waves = data['wave'].values
#print(data.diff())
#print(3500/0.048)
#print(9000/0.080)
plt.figure(figsize=(12, 4))
plt.plot(data['wave'], data['flux'], 'k.')
plt.show()
#data['wave'] = data['wave'].values[:].replace('D', 'E')

#data = data.loc[data['wave'].values.astype(np.float) > wavemin]
#data = data.loc[data['wave'].values.astype(np.float) < wavemax]
#fout = open(stellarpath+'lte05700-3.00+1.0.7.dat', 'w')
#fout.write('wave,flux\n')
#for k in range(len(data)):
#    wave = float(data['wave'].values[k].replace('D', 'E'))
#    flux = float(data['flux'].values[k].replace('D', 'E'))
#    fout.write('%.10f,%.10f\n' %(wave, flux))

#fout.close()