# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 16:38:03 2014

@author: rjs3
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scf1d import SCFprofile

def update(val):
    chi = chi_sl.val
    chi_s = chi_s_sl.val
    pdi = pdi_sl.val
    h_dry = h_dry_sl.val
    phi_b = phi_b_sl.val
    mn = mn_sl.val
    z = np.linspace(0,500,5000)
    phi = SCFprofile(z, chi, chi_s, h_dry, l_lat=16.1 , m_lat=449.436 ,#l_lat=14.178 , m_lat=589.8 ,
               mn=mn, phi_b=phi_b, pdi=pdi, disp=False)
    mainplot.set_ydata(phi)
    mainplot.set_xdata(z)
    ax.axis([0,z.max(),0,phi.max()*1.1])
    annotation.set_text('$h_{{RMS}}$={:.3g}'.format(rms(z,phi)))
    fig.canvas.draw_idle()

def rms(z,phi):
    theta=np.sum(phi)
    squared=z*z*phi
    ms=np.sum(squared)/theta
    return np.sqrt(ms)

fig, ax = plt.subplots(figsize=(8,10))
fig.subplots_adjust(left=0.10, bottom=0.5)

mainplot,=ax.plot([0,1])
ax.set_ylabel(r'$\phi$')
ax.set_xlabel('z (nm)')

sw = .75
sh = .03
sx = .1

chi_ax = fig.add_axes([sx, 0.1, sw, sh])
chi_s_ax = fig.add_axes([sx, 0.15, sw, sh])
pdi_ax = fig.add_axes([sx, 0.20, sw, sh])
h_dry_ax = fig.add_axes([sx, 0.25, sw, sh])
phi_b_ax = fig.add_axes([sx, 0.30, sw, sh])
mn_ax = fig.add_axes([sx, 0.35, sw, sh])

chi_sl = Slider(chi_ax,r'$\chi$',0,1,valinit=.0)
chi_s_sl = Slider(chi_s_ax,r'$\chi_s$',-1,1,valinit=.0)
pdi_sl = Slider(pdi_ax,r'$PDI$',1,2,valinit=1.0)
h_dry_sl = Slider(h_dry_ax,r'$h_{dry}$',1,100,valinit=25)
phi_b_sl = Slider(phi_b_ax,r'$\phi_b$',0,1,valinit=0)
mn_sl = Slider(mn_ax,r'$MW_n$',1000,100000,valinit=25000.)

chi_sl.on_changed(update)
chi_s_sl.on_changed(update)
pdi_sl.on_changed(update)
h_dry_sl.on_changed(update)
phi_b_sl.on_changed(update)
mn_sl.on_changed(update)

annotation=ax.annotate('null',xy=(0.8, 0.95), xycoords='axes fraction',)

update(0)

plt.show(block=True)
