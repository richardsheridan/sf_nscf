# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:38:03 2014

@author: rjs3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scf1d import SCFcache

def update(val):
    chi=chi_sl.val
    chi_s = chi_s_sl.val
    pdi = pdi_sl.val
    sigma = 10**sigma_sl.val
    nn = nn_sl.val
    phi = SCFcache(chi,chi_s,pdi,sigma,nn)
    z = np.arange(1,len(phi)+1)
    mainplot.set_ydata(phi)
    mainplot.set_xdata(z)
    ax.axis([0,np.max(z),0,np.max(phi)])
    fig.canvas.draw_idle()

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.10, bottom=0.4)
mainplot,=plt.plot(SCFcache(0,0,1,.1,200))
sw = .75
sh = .03
sx = .1
chi_ax = plt.axes([sx, 0.1, sw, sh])
chi_s_ax = plt.axes([sx, 0.15, sw, sh])
pdi_ax = plt.axes([sx, 0.20, sw, sh])
sigma_ax = plt.axes([sx, 0.25, sw, sh])
nn_ax = plt.axes([sx, 0.30, sw, sh])
chi_sl = Slider(chi_ax,r'$\chi$',0,1,valinit=.0)
chi_s_sl = Slider(chi_s_ax,r'$\chi_s$',-0.5,.5,valinit=.0)
pdi_sl = Slider(pdi_ax,r'$PDI$',1,2,valinit=1.0)
sigma_sl = Slider(sigma_ax,r'$\sigma$',-4,0,valinit=-1)
nn_sl = Slider(nn_ax,r'$MW_n$',50,1000,valinit=100.)
chi_sl.on_changed(update)
chi_s_sl.on_changed(update)
pdi_sl.on_changed(update)
sigma_sl.on_changed(update)
nn_sl.on_changed(update)
plt.show(block=True)