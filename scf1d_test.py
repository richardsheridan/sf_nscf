# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 21:08:34 2014

@author: Richard Sheridan

These tests were generated using the code as of 5/15/15.
No guarantee is made regarding absolute correctness, only that the results
haven't changed since that version.
"""

from __future__ import division, print_function
import numpy as np

from util import schultz_zimm
from scf1d import (SCFprofile, SCFsqueeze, BasicSystem, VaporSwollenSystem,
                   SCFsolve, SCFeqns, SCFeqns_multi, Propagator, NoConvergence)

g_zs_data=np.array((
   ( 0.90000000,  0.67833333,  0.53457407,  0.43538321,  0.36346252,
     0.30925318,  0.26714431,  0.23365571,  0.20652162,  0.18420452,
     0.16562391,  0.14999788,  0.13674637,  0.12542991,  0.11570948,
     0.10731936,  0.10004837,  0.0937266 ,  0.08821584,  0.08340252),
   ( 0.92222222,  0.85049383,  0.76425846,  0.68155058,  0.60783778,
     0.54394935,  0.4891275 ,  0.44218798,  0.40194363,  0.36734135,
     0.33749022,  0.31165092,  0.28921394,  0.26967689,  0.25262446,
     0.23771168,  0.22465053,  0.21319918,  0.20315348,  0.19434016),
   ( 0.94444444,  0.89197531,  0.84257659,  0.7930217 ,  0.74413393,
     0.69722471,  0.65323138,  0.61264733,  0.57563553,  0.54214957,
     0.5120236 ,  0.4850313 ,  0.46092225,  0.43944331,  0.42035059,
     0.4034157 ,  0.38842854,  0.37519809,  0.36355202,  0.3533357 ),
   ( 0.96666667,  0.93444444,  0.90345542,  0.87380169,  0.84505407,
     0.81709233,  0.79005942,  0.76420135,  0.73975859,  0.71691616,
     0.69579069,  0.67643584,  0.65885458,  0.64301244,  0.62884938,
     0.61628918,  0.6052467 ,  0.59563309,  0.58735937,  0.58033893),
   ( 0.98888889,  0.97790123,  0.96719845,  0.95693471,  0.94725739,
     0.93822305,  0.92984965,  0.92215235,  0.91515455,  0.90888661,
     0.90338093,  0.89866754,  0.89477117,  0.89170994,  0.88949506,
     0.88813126,  0.8876176 ,  0.88794842,  0.88911423,  0.89110265),
   ( 1.01111111,  1.02234568,  1.03387151,  1.04586384,  1.05850568,
     1.07179988,  1.08562363,  1.09981238,  1.11421611,  1.12872507,
     1.14327513,  1.1578427 ,  1.17243568,  1.18708411,  1.20183196,
     1.2167308 ,  1.23183506,  1.24719889,  1.26287398,  1.27890843),
   ( 1.03333333,  1.06777778,  1.10354047,  1.14085245,  1.17885107,
     1.21673962,  1.25401498,  1.29043   ,  1.32591   ,  1.36048312,
     1.39423345,  1.42727231,  1.45972146,  1.49170372,  1.52333786,
     1.55473604,  1.58600256,  1.6172335 ,  1.64851681,  1.67993261),
   ( 1.05555556,  1.11419753,  1.17627115,  1.23566805,  1.29102979,
     1.34261892,  1.39113583,  1.43729989,  1.48172328,  1.52489154,
     1.56717868,  1.60887008,  1.65018392,  1.69128862,  1.7323162 ,
     1.77337216,  1.81454277,  1.85590018,  1.8975062 ,  1.93941493),
   ( 1.07777778,  1.16160494,  1.21517227,  1.25496048,  1.28875838,
     1.32029364,  1.35135685,  1.38279751,  1.41500143,  1.44812428,
     1.48220808,  1.51724019,  1.55318343,  1.58999174,  1.62761807,
     1.66601842,  1.70515369,  1.74499047,  1.78550116,  1.82666377),
   ( 1.1       ,  1.00425926,  0.9494177 ,  0.91902123,  0.90402499,
     0.89922403,  0.90148479,  0.9088376 ,  0.91999378,  0.93407904,
     0.95048074,  0.96875736,  0.98858277,  1.00971099,  1.03195321,
     1.05516234,  1.07922242,  1.10404129,  1.1295452 ,  1.15567502)))

g_zs_ta_data = np.array((
       (  9.00000000e-01,   5.40000000e-01,   3.44750000e-01,
          2.32057407e-01,   1.63182762e-01,   1.18909613e-01,
          8.91942729e-02,   6.85112896e-02,   5.36702862e-02,
          4.27458873e-02,   3.45293323e-02,   2.82349142e-02,
          2.33361729e-02,   1.94708968e-02,   1.63840546e-02,
          1.38924051e-02,   1.18619166e-02,   1.01929871e-02,
          8.81054699e-03,   7.65729535e-03),
       (  0.00000000e+00,   1.38333333e-01,   1.68049383e-01,
          1.59655453e-01,   1.39999701e-01,   1.18990036e-01,
          9.99648391e-02,   8.37567498e-02,   7.02914374e-02,
          5.92119992e-02,   5.01154320e-02,   4.26348293e-02,
          3.64612874e-02,   3.13434433e-02,   2.70798154e-02,
          2.35098237e-02,   2.05055808e-02,   1.79650316e-02,
          1.58064477e-02,   1.39640932e-02),
       (  0.00000000e+00,   0.00000000e+00,   2.17746914e-02,
          4.01622085e-02,   5.09704753e-02,   5.55038957e-02,
          5.58698032e-02,   5.37794965e-02,   5.03986580e-02,
          4.64583240e-02,   4.23921664e-02,   3.84437829e-02,
          3.47398749e-02,   3.13372615e-02,   2.82521634e-02,
          2.54781029e-02,   2.29966886e-02,   2.07840162e-02,
          1.88143888e-02,   1.70624124e-02),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          3.50814472e-03,   8.73138241e-03,   1.39319543e-02,
          1.82139259e-02,   2.13049739e-02,   2.32609305e-02,
          2.42696439e-02,   2.45469939e-02,   2.42904207e-02,
          2.36629921e-02,   2.27918432e-02,   2.17724199e-02,
          2.06743749e-02,   1.95472967e-02,   1.84256090e-02,
          1.73324995e-02,   1.62829586e-02),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   5.78194222e-04,   1.82024107e-03,
          3.51226222e-03,   5.37879716e-03,   7.19678933e-03,
          8.82375170e-03,   1.01879864e-02,   1.12682785e-02,
          1.20747695e-02,   1.26345241e-02,   1.29818629e-02,
          1.31524460e-02,   1.31799736e-02,   1.30945794e-02,
          1.29222527e-02,   1.26848446e-02),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   9.74364337e-05,
          3.72423702e-04,   8.45750250e-04,   1.48928204e-03,
          2.25009829e-03,   3.07039056e-03,   3.89911195e-03,
          4.69684009e-03,   5.43641967e-03,   6.10148618e-03,
          6.68422006e-03,   7.18306621e-03,   7.60075285e-03,
          7.94271547e-03,   8.21591641e-03),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.67807191e-05,   7.56996886e-05,   1.98314090e-04,
          3.95755195e-04,   6.68038713e-04,   1.00663252e-03,
          1.39794287e-03,   1.82641850e-03,   2.27679495e-03,
          2.73546224e-03,   3.19113019e-03,   3.63500643e-03,
          4.06067510e-03,   4.46381663e-03),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   2.95216355e-06,   1.53949862e-05,
          4.58153916e-05,   1.02417597e-04,   1.91445606e-04,
          3.16399051e-04,   4.78006078e-04,   6.74663168e-04,
          9.03073979e-04,   1.15890201e-03,   1.43733487e-03,
          1.73351944e-03,   2.04286531e-03),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   5.30296046e-07,
          3.14642320e-06,   1.05080289e-05,   2.60638735e-05,
          5.35481090e-05,   9.64846722e-05,   1.57814991e-04,
          2.39684868e-04,   3.43375784e-04,   4.69343535e-04,
          6.17323631e-04,   7.86469082e-04),
       (  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          9.72209417e-08,   6.48139611e-07,   2.40177435e-06,
          6.53967800e-06,   1.46129172e-05,   2.84049958e-05,
          4.97630787e-05,   8.04351502e-05,   1.21938004e-04,
          1.75467518e-04,   2.41852179e-04)))

def calc_g_zs_ta_test():
    layers=10
    segments=20
    g_z = np.linspace(.9,1.1,layers)
    g_zs = Propagator(g_z,segments)
    assert np.allclose(g_zs.ta(), g_zs_ta_data, atol=1e-14)


def calc_g_zs_ngts_u_test():
    layers=10
    segments=20
    g_z = np.linspace(.9,1.1,layers)
    c = 1.0
    g_zs = Propagator(g_z,segments)
    assert np.allclose(g_zs.ngts_u(c), g_zs_data, atol=1e-14)

def calc_g_zs_ngts_test():
    layers=10
    segments=20
    g_z = np.linspace(.9,1.1,layers)
    c_i = np.zeros(segments)
    c_i[-1]= 1.0
    g_zs = Propagator(g_z,segments)
    assert np.allclose(g_zs.ngts(c_i), g_zs_data, atol=1e-14)

def calc_g_zs_free_test():
    layers=10
    segments=20
    g_z = np.linspace(.9,1.1,layers)
    g_zs = Propagator(g_z,segments)
    assert np.allclose(g_zs.free(), g_zs_data, atol=1e-14)

def SCFeqns_test():

    # away from solution
    phi_z = np.linspace(.5,0,50)
    chi = 0.1
    chi_s = 0.05
    sigma = .1
    navgsegments = 95.5
    pdi = 1.2
    p_i = schultz_zimm(pdi,navgsegments)
    data = np.array([  4.56885753e+00,   1.27473567e+01,   1.29799866e+01,
         1.25941193e+01,   1.20030366e+01,   1.54353170e+00,
         8.20490367e-01,   5.85342697e-01,   4.84869467e-01,
         4.36478200e-01,   4.09949567e-01,   3.92782245e-01,
         3.79629333e-01,   3.68190711e-01,   3.57478065e-01,
         3.47068739e-01,   3.36783786e-01,   3.26548662e-01,
         3.16332987e-01,   3.06124696e-01,   2.95919128e-01,
         2.85714537e-01,   2.75510285e-01,   2.65306148e-01,
         2.55102048e-01,   2.44897961e-01,   2.34693878e-01,
         2.24489796e-01,   2.14285714e-01,   2.04081633e-01,
         1.93877551e-01,   1.83673469e-01,   1.73469388e-01,
         1.63265306e-01,   1.53061224e-01,   1.42857143e-01,
         1.32653061e-01,   1.22448980e-01,   1.12244898e-01,
         1.02040816e-01,   9.18367347e-02,   8.16326531e-02,
         7.14285714e-02,   6.12244898e-02,   5.10204082e-02,
         4.08163265e-02,   3.06122449e-02,   2.04081633e-02,
         1.02040816e-02,  -2.60349244e-25])
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)

    assert np.allclose(result, data, atol=1e-14)

    # at solution
    phi_z = np.array([  3.65555778e-01,   3.97530942e-01,   3.95177593e-01,
         3.88755392e-01,   3.80784804e-01,   3.72128254e-01,
         3.63090353e-01,   3.53797489e-01,   3.44313620e-01,
         3.34678411e-01,   3.24920384e-01,   3.15062002e-01,
         3.05122091e-01,   2.95117245e-01,   2.85062724e-01,
         2.74973026e-01,   2.64862253e-01,   2.54744317e-01,
         2.44633057e-01,   2.34542304e-01,   2.24485900e-01,
         2.14477710e-01,   2.04531625e-01,   1.94661559e-01,
         1.84881453e-01,   1.75205269e-01,   1.65646998e-01,
         1.56220661e-01,   1.46940309e-01,   1.37820027e-01,
         1.28873934e-01,   1.20116183e-01,   1.11560959e-01,
         1.03222487e-01,   9.51150272e-02,   8.72528915e-02,
         7.96504609e-02,   7.23222205e-02,   6.52828140e-02,
         5.85471226e-02,   5.21303666e-02,   4.60482169e-02,
         4.03168798e-02,   3.49530831e-02,   2.99738486e-02,
         2.53958999e-02,   2.12345582e-02,   1.75020558e-02,
         1.42053630e-02,   1.13438569e-02,   8.90736329e-03,
         6.87516042e-03,   5.21636994e-03,   3.89180122e-03,
         2.85690535e-03,   2.06521082e-03,   1.47156065e-03,
         1.03463569e-03,   7.18524401e-04,   4.93361836e-04,
         3.35231541e-04,   2.25588058e-04,   1.50440247e-04,
         9.94770155e-05,   6.52498334e-05,   4.24697365e-05,
         2.74367658e-05,   1.75962019e-05,   1.12044893e-05,
         6.43768632e-07,   3.90023255e-07,   2.34559779e-07,
         1.40008161e-07,   8.29272726e-08,   4.87244589e-08,
         2.83845206e-08,   1.63805733e-08,   9.35038310e-09,
         5.26436221e-09,   2.90675760e-09,   1.55475537e-09,
         7.81883338e-10,   3.39857917e-10,   9.31247294e-11])
    data = np.array([  1.38858813e+00,   1.22712170e+01,   1.27152996e+01,
         1.25948180e+01,   1.21308339e+01,   2.77822712e+00,
         1.00037091e+00,   6.14461712e-01,   4.57514843e-01,
         3.84265846e-01,   3.46503169e-01,   3.24346931e-01,
         3.09058361e-01,   2.96756860e-01,   2.85731743e-01,
         2.75239734e-01,   2.64965933e-01,   2.54783573e-01,
         2.44647525e-01,   2.34547494e-01,   2.24487712e-01,
         2.14478326e-01,   2.04531829e-01,   1.94661625e-01,
         1.84881474e-01,   1.75205275e-01,   1.65647000e-01,
         1.56220662e-01,   1.46940309e-01,   1.37820027e-01,
         1.28873934e-01,   1.20116183e-01,   1.11560959e-01,
         1.03222487e-01,   9.51150272e-02,   8.72528915e-02,
         7.96504609e-02,   7.23222205e-02,   6.52828140e-02,
         5.85471226e-02,   5.21303666e-02,   4.60482169e-02,
         4.03168798e-02,   3.49530831e-02,   2.99738486e-02,
         2.53958999e-02,   2.12345582e-02,   1.75020558e-02,
         1.42053630e-02,   1.13438569e-02,   8.90736329e-03,
         6.87516042e-03,   5.21636994e-03,   3.89180122e-03,
         2.85690535e-03,   2.06521082e-03,   1.47156065e-03,
         1.03463569e-03,   7.18524401e-04,   4.93361836e-04,
         3.35231541e-04,   2.25588058e-04,   1.50440247e-04,
         9.94770155e-05,   6.52498334e-05,   4.24697365e-05,
         2.74367658e-05,   1.75962019e-05,   1.12044893e-05,
         6.43768632e-07,   3.90023255e-07,   2.34559779e-07,
         1.40008161e-07,   8.29272726e-08,   4.87244589e-08,
         2.83845206e-08,   1.63805733e-08,   9.35038310e-09,
         5.26436221e-09,   2.90675760e-09,   1.55475537e-09,
         7.81883338e-10,   3.39857917e-10,   9.31247294e-11])
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)

    assert np.allclose(result, data, atol=1e-14)

    #TODO: check float overflow handling

def SCFeqns_multi_test():
    u_jz0 = np.zeros((3,50))
    chi_jk = (1-np.eye(4))*0.10
    sigma_j = np.array((0,0,.01,0))
    phi_b_j = np.array((0,0.1,0,.9))
    n_avg_j = np.array((0,1,75,1))
    data = np.array([[ -5.52314290e-02,  -1.03134227e-01,  -1.23902379e-01,
         -1.22509546e-01,  -1.06273052e-01,  -8.37906912e-02,
         -6.14775076e-02,  -4.26557546e-02,  -2.82764492e-02,
         -1.80107679e-02,  -1.10521437e-02,  -6.53981797e-03,
         -3.73196876e-03,  -2.05342896e-03,  -1.08910584e-03,
         -5.56654539e-04,  -2.74100305e-04,  -1.29996454e-04,
         -5.93673960e-05,  -2.61011669e-05,  -1.10451439e-05,
         -4.49767739e-06,  -1.76204598e-06,  -6.63994071e-07,
         -2.40621051e-07,  -8.38353786e-08,  -2.80765437e-08,
         -9.03600415e-09,  -2.79392144e-09,  -8.29734542e-10,
         -2.36606623e-10,  -6.47656499e-11,  -1.70117891e-11,
         -4.28644351e-12,  -1.03569370e-12,  -2.39799723e-13,
         -5.33117863e-14,  -1.14174698e-14,  -2.33501732e-15,
         -4.69846300e-16,  -7.89946904e-18,  -4.46512879e-18,
         -3.83374067e-18,  -3.72284436e-18,  -3.70424847e-18,
         -3.70127350e-18,  -3.70081979e-18,  -3.70075388e-18,
         -3.70074477e-18,   3.88888889e-03],
       [ -4.80798201e-02,  -9.35171085e-02,  -1.12258255e-01,
         -1.10885581e-01,  -9.60858691e-02,  -7.56790137e-02,
         -5.54733754e-02,  -3.84583301e-02,  -2.54757702e-02,
         -1.62161669e-02,  -9.94440779e-03,  -5.88034163e-03,
         -3.35319684e-03,  -1.84357314e-03,  -9.76975949e-04,
         -4.98890083e-04,  -2.45417330e-04,  -1.16271723e-04,
         -5.30403986e-05,  -2.32918008e-05,  -9.84387025e-06,
         -4.00313149e-06,  -1.56606621e-06,  -5.89252096e-07,
         -2.13194325e-07,  -7.41538364e-08,  -2.47897057e-08,
         -7.96306979e-09,  -2.45723773e-09,  -7.28200323e-10,
         -2.07187834e-10,  -5.65785055e-11,  -1.48240256e-11,
         -3.72528658e-12,  -8.97582123e-13,  -2.07195619e-13,
         -4.59321732e-14,  -9.81671965e-15,  -2.00240574e-15,
         -4.03677261e-16,   4.69670784e-18,  -2.17197266e-18,
         -3.43474891e-18,  -3.65654152e-18,  -3.69373330e-18,
         -3.69968325e-18,  -3.70059067e-18,  -3.70072248e-18,
         -3.70074070e-18,   5.55555556e-03],
       [ -6.85647624e-02,  -1.03134227e-01,  -1.23902379e-01,
         -1.22509546e-01,  -1.06273052e-01,  -8.37906912e-02,
         -6.14775076e-02,  -4.26557546e-02,  -2.82764492e-02,
         -1.80107679e-02,  -1.10521437e-02,  -6.53981797e-03,
         -3.73196876e-03,  -2.05342896e-03,  -1.08910584e-03,
         -5.56654539e-04,  -2.74100305e-04,  -1.29996454e-04,
         -5.93673960e-05,  -2.61011669e-05,  -1.10451439e-05,
         -4.49767739e-06,  -1.76204598e-06,  -6.63994071e-07,
         -2.40621051e-07,  -8.38353786e-08,  -2.80765437e-08,
         -9.03600414e-09,  -2.79392143e-09,  -8.29734531e-10,
         -2.36606612e-10,  -6.47656388e-11,  -1.70117780e-11,
         -4.28643240e-12,  -1.03568259e-12,  -2.39788621e-13,
         -5.33006841e-14,  -1.14063675e-14,  -2.32391509e-15,
         -4.58744069e-16,   3.20276120e-18,   6.63710145e-18,
          7.26848958e-18,   7.37938588e-18,   7.39798177e-18,
          7.40095675e-18,   7.40141046e-18,   7.40147636e-18,
          7.40148547e-18,  -9.44444444e-03]])
    result=SCFeqns_multi(u_jz0,chi_jk, sigma_j, phi_b_j, n_avg_j)

    assert np.allclose(result, data, atol=1e-14)

def BasicSystem_test():
    bs = BasicSystem()
    p = (0,0,0,.1,.1,.1)
    pu = bs.unscale_parameters(p)
    ps = bs.scale_parameters(pu)
    assert p == ps

    data = np.array([[ -3.19411156e-01,  -3.89647496e-01,  -3.90270490e-01,
         -3.80437095e-01,  -3.66248523e-01,  -3.49113842e-01,
         -3.29405818e-01,  -3.07163773e-01,  -2.82352428e-01,
         -2.54947843e-01,  -2.24954033e-01,  -1.92409097e-01,
         -1.57423105e-01,  -1.20353697e-01,  -8.24002663e-02,
         -4.67742483e-02,  -1.89552925e-02,  -3.13814481e-03,
          2.20142759e-03,   2.33601299e-03,   1.27362900e-03,
          4.78180334e-04,   1.01462098e-04,  -2.28747647e-05,
         -4.03110074e-05,  -2.72163308e-05,  -1.29297425e-05,
         -4.36319481e-06,  -6.45616055e-07,   4.48961165e-07,
          5.39871059e-07,   4.77780221e-07,   6.32951664e-07,
          1.10337109e-06,   1.66086017e-06,   1.34678920e-06,
         -2.32710760e-06,  -1.44262092e-05,  -4.31263812e-05,
         -9.72718737e-05,  -1.73314694e-04,  -2.06006614e-04,
          1.04529329e-04,   1.86380529e-03,   8.63386677e-03,
          2.93905435e-02,   7.27777687e-02]])
    result = bs.from_cache((0,0,0,.1,.1,.1))

    assert np.allclose(result, data, atol=1e-14)

def VaporSwollenSystem_test():
    vs = VaporSwollenSystem()

    data = np.array([[ -3.19411156e-01,  -3.89647496e-01,  -3.90270490e-01,
         -3.80437095e-01,  -3.66248523e-01,  -3.49113842e-01,
         -3.29405818e-01,  -3.07163773e-01,  -2.82352428e-01,
         -2.54947843e-01,  -2.24954033e-01,  -1.92409097e-01,
         -1.57423105e-01,  -1.20353697e-01,  -8.24002663e-02,
         -4.67742483e-02,  -1.89552925e-02,  -3.13814481e-03,
          2.20142759e-03,   2.33601299e-03,   1.27362900e-03,
          4.78180334e-04,   1.01462098e-04,  -2.28747647e-05,
         -4.03110074e-05,  -2.72163308e-05,  -1.29297425e-05,
         -4.36319481e-06,  -6.45616055e-07,   4.48961165e-07,
          5.39871059e-07,   4.77780221e-07,   6.32951664e-07,
          1.10337109e-06,   1.66086017e-06,   1.34678920e-06,
         -2.32710760e-06,  -1.44262092e-05,  -4.31263812e-05,
         -9.72718737e-05,  -1.73314694e-04,  -2.06006614e-04,
          1.04529329e-04,   1.86380529e-03,   8.63386677e-03,
          2.93905435e-02,   7.27777687e-02]])
    result = vs.from_cache((1, -.6, 2.5, .01, .1, 75, 1))

    assert np.allclose(result, data, atol=1e-14)

def SCFsolve_test():

    #find the solution used in the previous test without an initial guess
    chi = 0.1
    chi_s = 0.05
    sigma = .1
    phi_b = 0
    navgsegments = 95.5
    pdi = 1.2
    parameters = (chi,chi_s,pdi,sigma,phi_b,navgsegments)
    bs = BasicSystem()
    data = np.array([[ -3.43014103e-01,  -4.28357066e-01,  -4.23920516e-01,
         -4.14558595e-01,  -4.03168321e-01,  -3.91006421e-01,
         -3.78517904e-01,  -3.65889209e-01,  -3.53215004e-01,
         -3.40553175e-01,  -3.27943914e-01,  -3.15417276e-01,
         -3.02996865e-01,  -2.90702002e-01,  -2.78549093e-01,
         -2.66552517e-01,  -2.54725175e-01,  -2.43078835e-01,
         -2.31624336e-01,  -2.20371723e-01,  -2.09330327e-01,
         -1.98508832e-01,  -1.87915327e-01,  -1.77557356e-01,
         -1.67441965e-01,  -1.57575750e-01,  -1.47964905e-01,
         -1.38615267e-01,  -1.29532359e-01,  -1.20721433e-01,
         -1.12187511e-01,  -1.03935418e-01,  -9.59698262e-02,
         -8.82952844e-02,  -8.09162623e-02,  -7.38371911e-02,
         -6.70625154e-02,  -6.05967549e-02,  -5.44445837e-02,
         -4.86109270e-02,  -4.31010768e-02,  -3.79208134e-02,
         -3.30765047e-02,  -2.85751248e-02,  -2.44241017e-02,
         -2.06308746e-02,  -1.72020484e-02,  -1.41420894e-02,
         -1.14516460e-02,  -9.12575058e-03,  -7.15232184e-03,
         -5.51142391e-03,  -4.17560784e-03,  -3.11137786e-03,
         -2.28150620e-03,  -1.64770258e-03,  -1.17310810e-03,
         -8.24217002e-04,  -5.72046149e-04,  -3.92576813e-04,
         -2.66625236e-04,  -1.79346082e-04,  -1.19557416e-04,
         -7.90287552e-05,  -5.18204963e-05,  -3.37185291e-05,
         -2.17767885e-05,  -1.39622374e-05,  -8.35685560e-07,
         -5.09857802e-07,  -3.08832388e-07,  -1.85707330e-07,
         -1.10847522e-07,  -6.56692743e-08,  -3.86076772e-08,
         -2.25201375e-08,  -1.30293677e-08,  -7.47352630e-09,
         -4.24640856e-09,  -2.38664486e-09,  -1.32321915e-09,
         -7.19741059e-10,  -3.79408720e-10,  -1.88179452e-10,
         -8.05951811e-11,  -2.10539864e-11]])

    result = SCFsolve(bs.field_equations(parameters),np.zeros((1,40)))

    assert np.allclose(result, data, atol=1e-14)

    # try a very hard one using the answer as an initial guess
    chi = 1
    chi_s = .5
    parameters = (chi,chi_s,pdi,sigma,phi_b,navgsegments)

    try:
        SCFsolve(bs.field_equations(parameters),np.zeros((1,40)))
    except NoConvergence:
        pass
    else: # Belongs to try, executes if no exception is raised
        assert False, 'should not arrive here'

    data = np.array([[  3.07254929e-01,   1.41259529e-01,   1.62795764e-01,
          1.68991420e-01,   1.76513700e-01,   1.83949395e-01,
          1.91169043e-01,   1.98034330e-01,   2.04468663e-01,
          2.10498735e-01,   2.16628713e-01,   2.25451759e-01,
          2.46568079e-01,   2.60299445e-01,   1.53151751e-01,
          4.00451631e-02,   6.78029664e-03,   1.22648571e-03,
          2.88017146e-04,   8.06983334e-05,   2.42960996e-05,
          7.48556220e-06,   2.32031419e-06,   7.19743365e-07,
          2.23051957e-07,   6.90243465e-08,   2.13222654e-08,
          6.57099636e-09,   2.01620328e-09,   6.11667304e-10,
          1.78586804e-10,   4.58150217e-11]])
    result = SCFsolve(bs.field_equations(parameters),data)

    assert np.allclose(result, data)#, atol=1e-14)

def walk_test():
    sigma = .1
    phi_b = 0
    navgsegments = 95.5
    pdi = 1.2
    chi = 1
    chi_s = .5
    parameters = (chi,chi_s,pdi,sigma,phi_b,navgsegments)

    BasicSystem._cache.clear()
    bs = BasicSystem()
    data = np.array([[  3.07254929e-01,   1.41259529e-01,   1.62795764e-01,
          1.68991420e-01,   1.76513700e-01,   1.83949395e-01,
          1.91169043e-01,   1.98034330e-01,   2.04468663e-01,
          2.10498735e-01,   2.16628713e-01,   2.25451759e-01,
          2.46568079e-01,   2.60299446e-01,   1.53151751e-01,
          4.00451629e-02,   6.78029659e-03,   1.22648570e-03,
          2.88017145e-04,   8.06983330e-05,   2.42960995e-05,
          7.48556221e-06,   2.32031422e-06,   7.19743315e-07,
          2.23052035e-07,   6.90242937e-08,   2.13221809e-08,
          6.57094402e-09,   2.01630121e-09,   6.11666288e-10,
          1.78584481e-10,   4.58154653e-11]])
    result = bs.walk(parameters)
    assert np.allclose(result, data)

    # check that the cache is holding items
    assert bs._cache

    # check high pdi solutions converge without too many layers
    parameters = (.47,0,1.75,.1,.1,100)
    result = bs.walk(parameters)

    assert result.shape <= (1,150)

    # check vapor swollen too
    vs = VaporSwollenSystem()
    param = (1, -1.5, 2.5, .01, .1, 75, 1)
    result = vs.walk(param)
    data = np.array([[ -1.52118963e+00,  -1.38452635e+00,  -8.13458121e-01,
         -1.90503082e-01,  -2.48189020e-02,  -2.80714305e-03,
         -3.11807636e-04,  -3.46076656e-05,  -3.84715908e-06,
         -4.28602497e-07,  -4.78067911e-08,  -4.90215584e-09,
         -1.08622606e-09,  -2.35384428e-10,  -7.48124674e-10,
          1.84496467e-10,   1.05030500e-10,  -1.02179807e-06,
         -9.42283970e-06,  -8.69154836e-05,  -8.01458554e-04,
         -7.36536171e-03,  -6.56689503e-02,  -4.69992867e-01],
       [ -9.41579696e-01,  -8.23304003e-01,  -5.89680968e-01,
         -1.45219368e-01,  -1.91983327e-02,  -2.17836225e-03,
         -2.42051082e-04,  -2.68633581e-05,  -2.98573081e-06,
         -3.32673197e-07,  -3.70463328e-08,  -3.92349194e-09,
         -8.41015870e-10,  -1.18288900e-10,  -4.80341830e-10,
          5.95140525e-11,  -1.73955150e-10,  -7.94810273e-07,
         -7.32873919e-06,  -6.76007680e-05,  -6.23348797e-04,
         -5.72794301e-03,  -5.10211525e-02,  -2.78862797e-01],
       [  1.73564354e+00,   1.14044803e+00,   2.18743362e-01,
          2.59693844e-02,   2.88315791e-03,   3.19408774e-04,
          3.54379768e-05,   3.93865902e-06,   4.38599467e-07,
          4.87036670e-08,   5.54966913e-09,   5.14995921e-10,
         -6.39609001e-11,   1.45624585e-10,   5.08469660e-12,
         -7.05947607e-11,  -4.30620886e-10,   1.13250803e-07,
          1.04704417e-06,   9.65791907e-06,   8.90905377e-05,
          8.21732223e-04,   7.57004032e-03,   6.89908215e-02]])

    assert np.allclose(result, data)


def SCFsqueeze_test():

    # squeeze the easy solution substantially
    chi = 0.1
    chi_s = 0.05
    sigma = .1
    phi_b = 0
    navgsegments = 95.5
    pdi = 1.2
    layers = 65
    data = np.array([  3.65558082e-01,   3.97533333e-01,   3.95179978e-01,
         3.88757819e-01,   3.80787284e-01,   3.72130794e-01,
         3.63092959e-01,   3.53800165e-01,   3.44316370e-01,
         3.34681240e-01,   3.24923297e-01,   3.15065003e-01,
         3.05125186e-01,   2.95120440e-01,   2.85066025e-01,
         2.74976440e-01,   2.64865786e-01,   2.54747977e-01,
         2.44636853e-01,   2.34546245e-01,   2.24489996e-01,
         2.14481972e-01,   2.04536066e-01,   1.94666194e-01,
         1.84886295e-01,   1.75210336e-01,   1.65652309e-01,
         1.56226235e-01,   1.46946170e-01,   1.37826200e-01,
         1.28880447e-01,   1.20123068e-01,   1.11568257e-01,
         1.03230243e-01,   9.51233029e-02,   8.72617647e-02,
         7.96600346e-02,   7.23326302e-02,   6.52942345e-02,
         5.85597723e-02,   5.21445036e-02,   4.60641230e-02,
         4.03348257e-02,   3.49732711e-02,   2.99963349e-02,
         2.54205052e-02,   2.12607873e-02,   1.75290516e-02,
         1.42319226e-02,   1.13685232e-02,   8.92858342e-03,
         6.89147965e-03,   5.22661400e-03,   3.89520205e-03,
         2.85314433e-03,   2.05437856e-03,   1.45405351e-03,
         1.01101862e-03,   6.89388676e-04,   4.59200254e-04,
         2.96354450e-04,   1.82124633e-04,   1.02535872e-04,
         4.79564181e-05,   1.33585282e-05])
    result = SCFsqueeze(chi,chi_s,pdi,sigma,phi_b,navgsegments,layers)

    assert np.allclose(result, data)

def SCFprofile_test():

    # basically checking that numpy interp hasn't changed
    data = np.array([ 0.50131233,  0.48796373,  0.47461512,  0.46161515,  0.45407189,
        0.44652863,  0.43950727,  0.43630926,  0.43311124,  0.43002774,
        0.42746534,  0.42490294,  0.42237239,  0.41994257,  0.41751275,
        0.41509461,  0.41270371,  0.4103128 ,  0.40793082,  0.40556467,
        0.40319852,  0.40084175,  0.39849793,  0.39615411,  0.3938202 ,
        0.39149703,  0.38917385,  0.38686069,  0.38455603,  0.38225138,
        0.37995647,  0.37766804,  0.37537961,  0.37310045,  0.37082607,
        0.36855168,  0.36628602,  0.36402374,  0.36176146,  0.35950736,
        0.35725557,  0.35500377,  0.35275965,  0.35051699,  0.34827433,
        0.34603894,  0.34380434,  0.34156975,  0.33934209,  0.33711472,
        0.33488748,  0.33266668,  0.33044589,  0.32822558,  0.32601089,
        0.3237962 ,  0.32158232,  0.31937341,  0.3171645 ,  0.31495671,
        0.31275338,  0.31055005,  0.30834816,  0.30615033,  0.30395249,
        0.30175644,  0.29956413,  0.29737182,  0.29518166,  0.29299499,
        0.29080832,  0.28862423,  0.28644341,  0.2842626 ,  0.28208487,
        0.27991021,  0.27773555,  0.27556458,  0.27339648,  0.27122839,
        0.26906468,  0.26690365,  0.26474262,  0.26258682,  0.26043346,
        0.25828011,  0.25613298,  0.25398802,  0.25184306,  0.24970552,
        0.24756981,  0.2454341 ,  0.24330721,  0.24118174,  0.23905627,
        0.23694129,  0.23482721,  0.23271312,  0.23061151,  0.22851014,
        0.22640935,  0.2243222 ,  0.22223504,  0.22014951,  0.21807832,
        0.21600712,  0.21393883,  0.21188558,  0.20983232,  0.20778356,
        0.20575051,  0.20371746,  0.20169083,  0.19968055,  0.19767026,
        0.19566877,  0.19368416,  0.19169955,  0.18972659,  0.18777093,
        0.18581527,  0.18387469,  0.18195165,  0.18002861,  0.17812473,
        0.1762384 ,  0.17435206,  0.17248972,  0.17064459,  0.16879946,
        0.16698396,  0.16518495,  0.16338595,  0.16162305,  0.15987545,
        0.15812785,  0.15642371,  0.15473311,  0.15304251,  0.15140355,
        0.14977574,  0.14814792,  0.14658068,  0.1450215 ,  0.14346232,
        0.14197324,  0.14048841,  0.1390038 ,  0.13759872,  0.13619364,
        0.13479386,  0.13347334,  0.13215281,  0.13084316,  0.1296112 ,
        0.12837924,  0.12716403,  0.12602362,  0.12488322,  0.12376549,
        0.12271842,  0.12167135,  0.1206527 ,  0.11969942,  0.11874615,
        0.11782659,  0.1169662 ,  0.1161058 ,  0.11528374,  0.11451395,
        0.11374416,  0.11301644,  0.11233374,  0.11165104,  0.11101315,
        0.11041294,  0.10981273,  0.10925898,  0.10873578,  0.10821259,
        0.10773643,  0.10728415,  0.10683186,  0.10642618,  0.10603832,
        0.10565047,  0.10530788,  0.10497781,  0.10464774,  0.10436088,
        0.10408203,  0.10380317,  0.10356487,  0.1033309 ,  0.10309693,
        0.10290046,  0.1027054 ,  0.10251113,  0.10234949,  0.10218784,
        0.10202858,  0.10189536,  0.10176214,  0.10163237,  0.10152315,
        0.10141392,  0.10130879,  0.10121965,  0.10113051,  0.10104578,
        0.10097335,  0.10090092,  0.10083296,  0.10077434,  0.10071572,
        0.10066145,  0.10061418,  0.1005669 ,  0.10052375,  0.10048576,
        0.10044776,  0.10041357,  0.10038313,  0.10035269,  0.1003257 ,
        0.10030138,  0.10027705,  0.10025582,  0.10023644,  0.10021706,
        0.1002004 ,  0.100185  ,  0.10016959,  0.10015656,  0.10014435,
        0.10013214,  0.10012198,  0.10011232,  0.10010265,  0.10009475,
        0.10008712,  0.1000795 ,  0.10007336,  0.10006735,  0.10006134,
        0.1000566 ,  0.10005187,  0.10004718,  0.10004347,  0.10003976,
        0.10003612,  0.10003321,  0.10003029,  0.10002747,  0.10002518,
        0.10002289,  0.1000207 ,  0.10001889,  0.10001709,  0.10001538,
        0.10001395,  0.10001251,  0.10001117,  0.10001002,  0.10000887,
        0.1000078 ,  0.10000685,  0.1000059 ,  0.10000502,  0.10000422,
        0.10000341,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ])
    result = SCFprofile(np.linspace(0,120,350), chi=.5, chi_s=.3, h_dry=15,
                        l_lat=1, mn=200, m_lat=1, phi_b=0.1, pdi=1.5, disp=False)

    assert np.allclose(result, data, atol=1e-14)


def benchmark():
    from time import clock
    BasicSystem._cache.clear()
    start=clock()
    bs = BasicSystem()
    p = (0,0,1,.1,0,160.52)
    bs.walk(p)
    p = (0,0,1.75,.1,0,360.52)
    bs.walk(p)
    print('Benchmark time:', clock()-start, 'seconds.')


def main():
    from time import clock
    start=clock()
    calc_g_zs_ta_test()
    calc_g_zs_free_test()
    calc_g_zs_ngts_u_test()
    calc_g_zs_ngts_test()
    calc_g_zs_ngts_test()
    BasicSystem_test()
    SCFsolve_test()
    SCFeqns_multi_test()
    walk_test()
    SCFsqueeze_test()
    SCFprofile_test()
    stop=clock()
    print('All tests passed in {:.3g} seconds!'.format(stop-start))
    benchmark()


if __name__ == '__main__':
    main()
