import numpy as np
import os
import sys
from astropy.io import ascii


# full sample
os.system('cp -r DISKS.csv temp.csv')
db = ascii.read('temp.csv', format='csv', fast_reader=True)

# Taurus only
tau = ((db['SFR'] == 'Lup') & (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'HJ') & 
       (db['FL_MULT'] != 'WJ') & (db['FL_MULT'] != 'BCB') & 
       (db['FL_MULT'] != 'HJB') & (db['SED'] != 'III') & (db['SED'] != 'I'))
print(len(db[tau]))


# SpT earlier than M5 (Teff >= 3000 K)
temp = ((db['logTeff'] >= np.log10(3000.)) & (db['FL_Teff'] == 0))
print(len(db[tau & temp])) 


# mm detection in either B6 or B7 (a "risk" criterion)
mmdet = np.logical_or((db['FL_B6'] == 0), (db['FL_B7'] == 0))
print(len(db[tau & temp & mmdet]))


# no known binaries (circumbinaries ok)
mults = ((db['FL_MULT'] != 'B'))
print(len(db[tau & temp & mmdet & mults]))


# combination
remainder = tau & temp & mmdet & mults


# and write these out to a text file
f = open('lup_sample_remainder.txt', 'w')
for i in range(len(db[remainder])):
    f.write(f'{db["NAME"][remainder][i]:15s}  {db["SPT"][remainder][i]:5s}  {db["logLs"][remainder][i]:.2f}  {0.5*(db["elogLs_H"][remainder][i] + db["elogLs_L"][remainder][i]):.2f}  {10**db["logTeff"][remainder][i]:.0f}\n')
f.close()

#print(db['NAME'][tau & temp & mmdet & mults])


#& (db['SFR'] != 'IC348') & (db['FL_MULT'] != 'J') & (db['FL_MULT'] != 'B') & (db['FL_MULT'] != 'HJ') & (db['FL_MULT'] != 'HJB') & (db['FL_MULT'] != 'HC') & (db['FL_MULT'] != 'HCB') & (db['FL_MULT'] != 'CB') & (db['FL_Teff'] == 0) & (db['SED'] != 'I'))
