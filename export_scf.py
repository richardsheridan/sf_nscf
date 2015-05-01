# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import, unicode_literals
from scf1d import SCFcache
from matplotlib import pyplot as plt
import numpy as np
import csv
import time

starttime=time.time()

parameter_tuples, polymer_arrays, failures = [], [], []
count = 0
for r in (20, 50, 100, 200, 500):
    for sigma in (.05,.1,.2,.4):
        for pdi in (1,1.1,1.2,1.5,2):
            for chi_s in np.linspace(0,.3,10):
                for chi in np.linspace(0,.9,10):
                    parameters=(chi,chi_s,pdi,sigma,r)
                    count+=1
                    if not count % 1000:
                        print("Parameter set",count)
                    try:
                        phi=SCFcache(*parameters,disp=False)
                        parameter_tuples.append(parameters)
                        polymer_arrays.append(phi.copy())
                    except KeyboardInterrupt:
                        import pdb
                        pdb.set_trace()
                    except:
                        print(parameters)
                        print("^^^^these parameters did not work^^^")
                        failures.append(parameters)

print(count,'solutions done in',time.time()-starttime,'seconds.')
with open('scf_arrays.txt','wb') as arrayfile:
    writer=csv.writer(arrayfile)
    for polymer_array in polymer_arrays:
        writer.writerow(polymer_array)
with open('scf_parameters.txt','wb') as paramfile:
    writer=csv.writer(paramfile)
    for parameter_tuple in parameter_tuples:
        writer.writerow(parameter_tuple)
print('saved')