# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 21:08:34 2014

@author: Richard Sheridan

These tests were generated using the code as of 11/7/14.
No guarantee is made regarding absolute correctness, only that the results
haven't changed since that version.
"""

from __future__ import division, print_function
import numpy as np

from scf1d import (SCFprofile, SCFcache, SCFsolve, SCFeqns, SCFeqns_u, SZdist,
                   calc_g_zs, NoConvergence)


def calc_g_zs_test():
    layers=10
    segments=20
    g_z = np.linspace(.9,1.1,layers)
    data=np.array((
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

    # free chains
    c_i = np.zeros(segments)
    c_i[-1]= 1.0
    assert np.allclose(calc_g_zs(g_z,c_i,layers,segments), data, atol=1e-14)

    # uniform chains
    c_i = 1.0
    assert np.allclose(calc_g_zs(g_z,c_i,layers,segments), data, atol=1e-14)

    # end-tethered chains
    c_i = -1
    data = np.array((
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
    assert np.allclose(calc_g_zs(g_z,c_i,layers,segments), data, atol=1e-14)

    # TODO: Test free chains c_i = 0


def SZdist_test():

    # uniform
    pdi=1
    nn=100
    data = np.array(((
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.)))
    assert np.allclose(SZdist(pdi,nn), data, atol=1e-14)

    # too narrow
    pdi=1.0000000001
    nn=100
    assert np.allclose(SZdist(pdi,nn), data, atol=1e-14)

    # broad
    pdi=2
    nn=30
    data = np.array(((
          3.22405367e-02,   3.11835662e-02,   3.01612473e-02,
          2.91724440e-02,   2.82160575e-02,   2.72910251e-02,
          2.63963189e-02,   2.55309446e-02,   2.46939407e-02,
          2.38843770e-02,   2.31013540e-02,   2.23440015e-02,
          2.16114780e-02,   2.09029695e-02,   2.02176887e-02,
          1.95548740e-02,   1.89137890e-02,   1.82937212e-02,
          1.76939817e-02,   1.71139040e-02,   1.65528435e-02,
          1.60101767e-02,   1.54853007e-02,   1.49776321e-02,
          1.44866070e-02,   1.40116795e-02,   1.35523220e-02,
          1.31080240e-02,   1.26782919e-02,   1.22626480e-02,
          1.18606306e-02,   1.14717929e-02,   1.10957028e-02,
          1.07319424e-02,   1.03801075e-02,   1.00398071e-02,
          9.71066304e-03,   9.39230964e-03,   9.08439310e-03,
          8.78657127e-03,   8.49851320e-03,   8.21989880e-03,
          7.95041846e-03,   7.68977274e-03,   7.43767200e-03,
          7.19383611e-03,   6.95799411e-03,   6.72988393e-03,
          6.50925209e-03,   6.29585343e-03,   6.08945080e-03,
          5.88981486e-03,   5.69672376e-03,   5.50996294e-03,
          5.32932487e-03,   5.15460882e-03,   4.98562064e-03,
          4.82217255e-03,   4.66408293e-03,   4.51117611e-03,
          4.36328216e-03,   4.22023676e-03,   4.08188094e-03,
          3.94806097e-03,   3.81862813e-03,   3.69343861e-03,
          3.57235329e-03,   3.45523762e-03,   3.34196146e-03,
          3.23239893e-03,   3.12642829e-03,   3.02393178e-03,
          2.92479550e-03,   2.82890930e-03,   2.73616662e-03,
          2.64646441e-03,   2.55970299e-03,   2.47578594e-03,
          2.39462002e-03,   2.31611504e-03,   2.24018376e-03,
          2.16674180e-03,   2.09570755e-03,   2.02700209e-03,
          1.96054905e-03,   1.89627461e-03,   1.83410734e-03,
          1.77397814e-03,   1.71582022e-03,   1.65956895e-03,
          1.60516180e-03,   1.55253834e-03,   1.50164008e-03,
          1.45241046e-03,   1.40479478e-03,   1.35874013e-03,
          1.31419533e-03,   1.27111088e-03,   1.22943891e-03,
          1.18913311e-03,   1.15014869e-03,   1.11244233e-03,
          1.07597213e-03,   1.04069757e-03,   1.00657945e-03,
          9.73579848e-04,   9.41662104e-04,   9.10790748e-04,
          8.80931476e-04,   8.52051107e-04,   8.24117549e-04,
          7.97099762e-04,   7.70967724e-04,   7.45692395e-04,
          7.21245691e-04,   6.97600444e-04,   6.74730382e-04,
          6.52610088e-04,   6.31214985e-04,   6.10521296e-04,
          5.90506027e-04,   5.71146937e-04,   5.52422513e-04,
          5.34311949e-04,   5.16795120e-04,   4.99852561e-04,
          4.83465445e-04,   4.67615562e-04,   4.52285300e-04,
          4.37457625e-04,   4.23116058e-04,   4.09244663e-04,
          3.95828028e-04,   3.82851241e-04,   3.70299885e-04,
          3.58160010e-04,   3.46418129e-04,   3.35061191e-04,
          3.24076579e-04,   3.13452085e-04,   3.03175903e-04,
          2.93236615e-04,   2.83623175e-04,   2.74324902e-04,
          2.65331462e-04,   2.56632862e-04,   2.48219436e-04,
          2.40081835e-04,   2.32211016e-04,   2.24598233e-04,
          2.17235027e-04,   2.10113216e-04,   2.03224886e-04,
          1.96562381e-04,   1.90118300e-04,   1.83885481e-04,
          1.77856998e-04,   1.72026152e-04,   1.66386464e-04,
          1.60931666e-04,   1.55655699e-04,   1.50552698e-04,
          1.45616994e-04,   1.40843101e-04,   1.36225715e-04,
          1.31759704e-04,   1.27440108e-04,   1.23262124e-04,
          1.19221111e-04,   1.15312578e-04,   1.11532182e-04,
          1.07875722e-04,   1.04339135e-04,   1.00918492e-04,
          9.76099898e-05,   9.44099537e-05,   9.13148273e-05,
          8.83211712e-05,   8.54256588e-05,   8.26250726e-05,
          7.99163005e-05,   7.72963325e-05,   7.47622573e-05,
          7.23112590e-05,   6.99406139e-05,   6.76476879e-05,
          6.54299329e-05,   6.32848845e-05,   6.12101592e-05,
          5.92034515e-05,   5.72625315e-05,   5.53852424e-05,
          5.35694982e-05,   5.18132812e-05,   5.01146398e-05,
          4.84716865e-05,   4.68825956e-05,   4.53456013e-05,
          4.38589956e-05,   4.24211267e-05,   4.10303968e-05,
          3.96852604e-05,   3.83842228e-05,   3.71258383e-05,
          3.59087085e-05,   3.47314810e-05,   3.35928476e-05,
          3.24915431e-05,   3.14263436e-05,   3.03960655e-05,
          2.93995640e-05,   2.84357316e-05,   2.75034974e-05,
          2.66018255e-05,   2.57297140e-05,   2.48861936e-05,
          2.40703271e-05,   2.32812080e-05,   2.25179592e-05,
          2.17797327e-05,   2.10657081e-05,   2.03750920e-05,
          1.97071171e-05,   1.90610409e-05,   1.84361457e-05,
          1.78317369e-05,   1.72471431e-05,   1.66817144e-05,
          1.61348228e-05,   1.56058604e-05,   1.50942394e-05,
          1.45993914e-05,   1.41207664e-05,   1.36578326e-05,
          1.32100756e-05,   1.27769978e-05,   1.23581180e-05,
          1.19529707e-05,   1.15611057e-05,   1.11820876e-05,
          1.08154952e-05,   1.04609211e-05,   1.01179713e-05,
          9.78626472e-06,   9.46543280e-06,   9.15511900e-06,
          8.85497850e-06,   8.56467777e-06,   8.28389424e-06,
          8.01231588e-06,   7.74964092e-06,   7.49557747e-06,
          7.24984321e-06,   7.01216508e-06,   6.78227897e-06,
          6.55992941e-06,   6.34486935e-06,   6.13685979e-06,
          5.93566959e-06,   5.74107520e-06,   5.55286037e-06,
          5.37081595e-06,   5.19473966e-06,   5.02443584e-06,
          4.85971524e-06,   4.70039482e-06,   4.54629755e-06,
          4.39725219e-06,   4.25309311e-06,   4.11366014e-06,
          3.97879832e-06,   3.84835779e-06,   3.72219362e-06,
          3.60016559e-06,   3.48213813e-06,   3.36798006e-06,
          3.25756454e-06,   3.15076887e-06,   3.04747438e-06,
          2.94756629e-06,   2.85093357e-06,   2.75746885e-06,
          2.66706827e-06,   2.57963137e-06,   2.49506100e-06,
          2.41326317e-06,   2.33414699e-06,   2.25762455e-06,
          2.18361081e-06,   2.11202354e-06,   2.04278317e-06,
          1.97581277e-06,   1.91103792e-06,   1.84838665e-06,
          1.78778933e-06,   1.72917862e-06,   1.67248940e-06,
          1.61765868e-06,   1.56462552e-06,   1.51333099e-06,
          1.46371810e-06,   1.41573171e-06,   1.36931851e-06,
          1.32442691e-06,   1.28100703e-06,   1.23901062e-06,
          1.19839102e-06,   1.15910309e-06,   1.12110317e-06,
          1.08434904e-06,   1.04879985e-06,   1.01441610e-06)))
    assert np.allclose(SZdist(pdi,nn), data, atol=1e-14)


easy_phi_z = np.array([
         3.65550131e-01,   3.97525425e-01,   3.95172134e-01,
         3.88749940e-01,   3.80779349e-01,   3.72122793e-01,
         3.63084887e-01,   3.53792021e-01,   3.44308153e-01,
         3.34672949e-01,   3.24914931e-01,   3.15056561e-01,
         3.05116668e-01,   2.95111844e-01,   2.85057350e-01,
         2.74967684e-01,   2.64856948e-01,   2.54739052e-01,
         2.44627839e-01,   2.34537137e-01,   2.24480790e-01,
         2.14472662e-01,   2.04526645e-01,   1.94656653e-01,
         1.84876626e-01,   1.75200527e-01,   1.65642347e-01,
         1.56216107e-01,   1.46935857e-01,   1.37815685e-01,
         1.28869707e-01,   1.20112077e-01,   1.11556982e-01,
         1.03218644e-01,   9.51113248e-02,   8.72493367e-02,
         7.96470616e-02,   7.23189855e-02,   6.52797529e-02,
         5.85442457e-02,   5.21276853e-02,   4.60457440e-02,
         4.03146298e-02,   3.49510720e-02,   2.99720919e-02,
         2.53944097e-02,   2.12333392e-02,   1.75011012e-02,
         1.42046521e-02,   1.13433550e-02,   8.90702465e-03,
         6.87493337e-03,   5.21620335e-03,   3.89165005e-03,
         2.85673384e-03,   2.06499215e-03,   1.47127167e-03,
         1.03424720e-03,   7.18028131e-04,   4.92777056e-04,
         3.34589594e-04,   2.24917559e-04,   1.49760747e-04,
         9.87989677e-05,   6.45770566e-05,   4.18019610e-05,
         2.67713428e-05,   1.69288546e-05,   1.05292942e-05,
         6.39309703e-06,   3.73012439e-06,   2.01768258e-06,
         9.21302206e-07,   2.55824637e-07])


def SCFeqns_test():

    # away from solution
    phi_z = np.linspace(.5,0,50)
    chi = 0.1
    chi_s = 0.05
    sigma = .1
    navgsegments = 95.5
    pdi = 1.2
    p_i = SZdist(pdi,navgsegments)
    data = np.array((
        0.24810378,  0.19945097,  0.16973725,  0.14729956,  0.13006686,
        0.11658225,  0.10586535,  0.0972781 ,  0.09039446,  0.08491051,
        0.08059274,  0.07725048,  0.07472126,  0.07286278,  0.07154816,
        0.0706631 ,  0.07010401,  0.06977679,  0.06959604,  0.06948434,
        0.06937178,  0.0691954 ,  0.06889878,  0.06843147,  0.06774859,
        0.06681028,  0.06558131,  0.06403057,  0.06213075,  0.05985788,
        0.05719101,  0.05411175,  0.05060381,  0.04665231,  0.04224292,
        0.03736045,  0.03198701,  0.02609946,  0.01966667,  0.01264807,
        0.00499856, -0.00330835, -0.01221432, -0.02142874, -0.03017032,
       -0.036859  , -0.03906598, -0.03438694, -0.02291   , -0.01063772))
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)
    assert np.allclose(result, data, atol=1e-14)

    # at solution
    phi_z = easy_phi_z.copy()
    data = np.array([
         2.15679141e-09,   1.79636311e-09,   1.14144011e-09,
        -4.26718882e-10,   2.19736107e-09,   3.25258487e-09,
         2.82894724e-09,   3.92600941e-09,   4.52902954e-09,
         3.92317240e-09,   3.90727395e-09,   2.16293861e-09,
         3.01917874e-09,   2.40970921e-09,   4.05688316e-09,
         4.39389802e-09,   5.85036347e-09,   4.67507322e-09,
         4.21623264e-09,   3.60206598e-09,   4.70427380e-09,
         4.63695898e-09,   5.49120407e-09,   6.10241882e-09,
         7.80735607e-09,   8.91483148e-09,   9.42704295e-09,
         1.04590603e-08,   1.00006105e-08,   1.24722999e-08,
         1.41677424e-08,   1.64147597e-08,   1.91621784e-08,
         2.18245532e-08,   2.42455538e-08,   2.74230632e-08,
         3.22772100e-08,   3.92602619e-08,   4.79658271e-08,
         5.73868632e-08,   6.65241277e-08,   7.46713040e-08,
         8.18733290e-08,   8.87611609e-08,   9.54999615e-08,
         1.01632030e-07,   1.05586531e-07,   1.05323089e-07,
         9.90404951e-08,   8.57449149e-08,   6.61943253e-08,
         4.24110654e-08,   1.74404836e-08,  -5.70322180e-09,
        -2.52843270e-08,  -4.24516584e-08,  -6.40530091e-08,
        -1.05451432e-07,  -1.52929123e-07,  -1.83977922e-07,
        -1.89611585e-07,  -1.74002548e-07,  -1.46423278e-07,
        -1.15342273e-07,  -8.62539708e-08,  -6.17858872e-08,
        -4.25946188e-08,  -2.82625129e-08,  -1.79312797e-08,
        -1.06252390e-08,  -5.74556937e-09,  -2.78403094e-09,
        -1.13988252e-09,  -2.92913286e-10])
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)
    assert np.allclose(result, data, atol=1e-14)

    # check that penalty penalizes
    phi_z[0]=.999
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)
    below = np.linalg.norm(result,np.inf)
    phi_z[0] = 1.0
    result = SCFeqns(phi_z,chi,chi_s,sigma,navgsegments,p_i)
    above = np.linalg.norm(result,np.inf)
    assert above > (below + 1e5*(phi_z[0]-.99999))

    #TODO: check float overflow handling


def SCFeqns_u_test():
    chi=.1
    chi_s=.05
    sigma=.1
    pdi=1.2
    navgsegments=95.5
    p_i = SZdist(pdi,navgsegments)
    u_z = np.array([  3.43006993e-01,   4.28349935e-01,   4.23913496e-01,
         4.14551684e-01,   4.03161526e-01,   3.90999746e-01,
         3.78511354e-01,   3.65882790e-01,   3.53208719e-01,
         3.40547028e-01,   3.27937909e-01,   3.15411416e-01,
         3.02991155e-01,   2.90696445e-01,   2.78543693e-01,
         2.66547276e-01,   2.54720097e-01,   2.43073923e-01,
         2.31619594e-01,   2.20367154e-01,   2.09325934e-01,
         1.98504619e-01,   1.87911298e-01,   1.77553515e-01,
         1.67438316e-01,   1.57572297e-01,   1.47961654e-01,
         1.38612223e-01,   1.29529528e-01,   1.20718822e-01,
         1.12185125e-01,   1.03933266e-01,   9.59679162e-02,
         8.82936270e-02,   8.09148704e-02,   7.38360816e-02,
         6.70617103e-02,   6.05962844e-02,   5.44444877e-02,
         4.86112577e-02,   4.31018993e-02,   3.79222036e-02,
         3.30785426e-02,   2.85778823e-02,   2.44276240e-02,
         2.06351574e-02,   1.72070143e-02,   1.41475718e-02,
         1.14573860e-02,   9.13141341e-03,   7.15753117e-03,
         5.51580836e-03,   4.17884819e-03,   3.11324354e-03,
         2.28187351e-03,   1.64655283e-03,   1.17050981e-03,
         8.20299026e-04,   5.66968571e-04,   3.86504485e-04,
         2.59706126e-04,   1.71694503e-04,   1.11241489e-04,
         7.00629404e-05,   4.21699968e-05,   2.33378564e-05,
         1.07420296e-05,   2.84148909e-06])

    result = SCFeqns_u(u_z,chi,chi_s,sigma,navgsegments,p_i)

    data = np.array([ -3.07771919e-10,  -3.76982612e-10,  -1.01695752e-09,
         2.46801968e-10,   1.10221610e-09,   6.05490103e-11,
        -3.05329262e-10,   1.66717934e-09,   1.87079907e-09,
         1.68562697e-09,   3.91757959e-10,  -1.38017120e-09,
        -1.91659938e-09,  -1.23429472e-09,  -5.81238391e-10,
        -9.46159373e-10,  -2.92230962e-10,   8.05277539e-10,
         1.02631467e-09,   1.20857008e-09,   1.01646969e-10,
        -3.22862320e-10,  -1.85071958e-10,   5.57896729e-10,
         7.60392777e-10,  -6.95533769e-10,  -7.44130008e-10,
        -6.53599008e-10,  -3.95680794e-10,   2.88328694e-10,
        -2.76271950e-10,  -3.76709802e-10,  -5.08009051e-11,
        -4.35606828e-11,  -3.80524778e-11,   4.87524882e-11,
        -1.43034612e-11,   8.58460733e-11,   7.79860551e-11,
        -1.72794140e-11,  -3.77273629e-11,   1.06229747e-11,
         5.50300500e-11,   4.57448419e-11,   4.03607217e-11,
         1.40921060e-11,   2.14250249e-11,   3.29307120e-11,
         1.01080394e-11,   7.61183651e-12,  -1.48409236e-12,
        -2.02647794e-12,   3.48382781e-12,   2.23308438e-12,
        -2.29546417e-12,   2.45060482e-12,   8.11895039e-13,
         3.92474139e-13,  -1.40297604e-13,  -7.67181999e-14,
         4.29475899e-13,   7.74513917e-14,   1.67485044e-13,
        -8.99905421e-15,   1.96386486e-14,  -4.70833259e-14,
         3.80390333e-14,   2.27622698e-15])

    assert np.allclose(result,data,atol=1e-14)

    # both SCFeqns should agree
    result = SCFeqns_u(u_z,chi,chi_s,sigma,navgsegments,p_i,dump_phi = True)
    data = easy_phi_z.copy()[:68]

    assert np.allclose(result, data, atol = 2e-5)


def SCFsolve_test():

    #find the solution used in the previous test without an initial guess
    chi = 0.1
    chi_s = 0.05
    sigma = .1
    phi_b = 0
    navgsegments = 95.5
    pdi = 1.2
    data = easy_phi_z.copy()
    result = SCFsolve(chi,chi_s,pdi,sigma,phi_b,navgsegments)
    assert np.allclose(result, data, atol=1e-14)

    # try a very hard one using the answer as an initial guess
    chi = 1
    chi_s = .5
    try:
        SCFsolve(chi,chi_s,pdi,sigma,phi_b,navgsegments)
    except NoConvergence:
        pass
    else: # Belongs to try, executes if no exception is raised
        assert False, 'should not arrive here'

    phi0 = np.array((
         7.68622748e-01,   7.38403430e-01,   7.24406743e-01,
         7.18854113e-01,   7.13805025e-01,   7.08721605e-01,
         7.03592422e-01,   6.98483104e-01,   6.93373096e-01,
         6.87807938e-01,   6.79307808e-01,   6.56674507e-01,
         5.77590583e-01,   3.58036148e-01,   1.00802863e-01,
         1.68381666e-02,   2.86654637e-03,   6.37708606e-04,
         1.74095080e-04,   5.19490850e-05,   1.59662700e-05,
         4.94738039e-06,   1.53508370e-06,   4.75950448e-07,
         1.47353950e-07))
    data = np.array([
         7.68622748e-01,   7.38403430e-01,   7.24406744e-01,
         7.18854113e-01,   7.13805025e-01,   7.08721605e-01,
         7.03592422e-01,   6.98483104e-01,   6.93373097e-01,
         6.87807939e-01,   6.79307808e-01,   6.56674508e-01,
         5.77590583e-01,   3.58036145e-01,   1.00802861e-01,
         1.68381663e-02,   2.86654627e-03,   6.37708520e-04,
         1.74094994e-04,   5.19489979e-05,   1.59661825e-05,
         4.94729291e-06,   1.53499630e-06,   4.75863078e-07,
         1.47266530e-07,   4.54466899e-08,   1.39424805e-08,
         4.20762450e-09,   1.19832073e-09,   2.57594313e-10])
    result = SCFsolve(chi,chi_s,pdi,sigma,phi_b,navgsegments,phi0=phi0)
    assert np.allclose(result, data, atol=1e-14)


def SCFcache_test():

    # check that the hard solution can be found by walking
    chi = 1
    chi_s = 0.5
    sigma = .1
    phi_b  = 0
    navgsegments = 95.5
    pdi = 1.2
    from collections import OrderedDict
    cache = OrderedDict()
    data = np.array([
         7.68622748e-01,   7.38403430e-01,   7.24406744e-01,
         7.18854113e-01,   7.13805025e-01,   7.08721605e-01,
         7.03592422e-01,   6.98483104e-01,   6.93373097e-01,
         6.87807939e-01,   6.79307808e-01,   6.56674508e-01,
         5.77590583e-01,   3.58036145e-01,   1.00802861e-01,
         1.68381664e-02,   2.86654639e-03,   6.37708645e-04,
         1.74095120e-04,   5.19491242e-05,   1.59663088e-05,
         4.94741910e-06,   1.53512234e-06,   4.75989066e-07,
         1.47392564e-07,   4.55729455e-08,   1.40695041e-08,
         4.33706730e-09,   1.33497413e-09,   4.10311966e-10,
         1.25932129e-10,   3.85969529e-11,   1.18134453e-11,
         3.61092206e-12,   1.10226030e-12,   3.36024510e-13,
         1.02293126e-13,   3.10880552e-14,   9.42377326e-15,
         2.84090350e-15,   8.43003506e-16,   2.36439378e-16,
         5.02832752e-17])
    result = SCFcache(chi,chi_s,pdi,sigma,phi_b,navgsegments,False,cache)
    assert np.allclose(result, data, atol=1e-14)

    # check that the cache is holding items
    assert cache

    # check that cache is reordered on hits and misses
    cache_keys = list(cache)
    oldest_key = cache_keys[0]
    newest_key = cache_keys[-1]
    SCFcache(0,0,1,.1,.1,50,False,cache)

    assert oldest_key == list(cache)[-1]
    SCFcache(chi,chi_s,pdi+.1,sigma,phi_b,navgsegments,False,cache)
    assert newest_key == list(cache)[-2]


long_profile = np.array([
        0.50129252,  0.49016694,  0.47904136,  0.46791578,  0.45907616,
        0.45278886,  0.44650156,  0.44021426,  0.43734935,  0.43468384,
        0.43201832,  0.42957398,  0.42743832,  0.42530266,  0.423167  ,
        0.4211297 ,  0.41910456,  0.41707943,  0.41506606,  0.41307337,
        0.41108069,  0.409088  ,  0.40711254,  0.40514049,  0.40316843,
        0.4012021 ,  0.39924866,  0.39729522,  0.39534177,  0.39340175,
        0.39146552,  0.38952929,  0.38759695,  0.38567615,  0.38375536,
        0.38183456,  0.37992356,  0.37801629,  0.37610902,  0.37420406,
        0.37230849,  0.37041292,  0.36851735,  0.36662854,  0.36474307,
        0.36285759,  0.36097336,  0.35909662,  0.35721988,  0.35534315,
        0.35347109,  0.35160197,  0.34973285,  0.34786431,  0.34600191,
        0.34413951,  0.34227712,  0.34041808,  0.33856171,  0.33670533,
        0.33484912,  0.33299823,  0.33114733,  0.32929644,  0.32744811,
        0.3256023 ,  0.32375649,  0.32191069,  0.32006959,  0.3182286 ,
        0.31638761,  0.31454872,  0.31271238,  0.31087605,  0.30903971,
        0.30720761,  0.30537586,  0.30354411,  0.30171417,  0.29988703,
        0.29805988,  0.29623273,  0.29440966,  0.29258722,  0.29076477,
        0.28894399,  0.28712642,  0.28530886,  0.28349129,  0.2816779 ,
        0.27986547,  0.27805304,  0.27624217,  0.27443521,  0.27262825,
        0.27082129,  0.26901879,  0.26721773,  0.26541666,  0.26361706,
        0.26182239,  0.26002772,  0.25823305,  0.2564433 ,  0.25465563,
        0.25286797,  0.25108164,  0.24930169,  0.24752173,  0.24574178,
        0.24396735,  0.24219593,  0.24042451,  0.23865423,  0.2368923 ,
        0.23513038,  0.23336845,  0.2316128 ,  0.22986147,  0.22811014,
        0.22635957,  0.2246201 ,  0.22288063,  0.22114116,  0.21940883,
        0.21768267,  0.21595651,  0.21423048,  0.21251928,  0.21080808,
        0.20909687,  0.20739378,  0.20569943,  0.20400507,  0.20231071,
        0.20063447,  0.19895909,  0.19728372,  0.19561748,  0.19396351,
        0.19230955,  0.19065559,  0.18902332,  0.18739349,  0.18576366,
        0.18414394,  0.18254131,  0.18093868,  0.17933605,  0.17775925,
        0.17618722,  0.17461519,  0.17305403,  0.17151637,  0.1699787 ,
        0.16844103,  0.1669337 ,  0.16543449,  0.16393528,  0.16244726,
        0.16099091,  0.15953456,  0.15807821,  0.15665673,  0.15524791,
        0.15383908,  0.15244105,  0.15108457,  0.14972809,  0.14837161,
        0.14705396,  0.1457547 ,  0.14445544,  0.14316556,  0.14192829,
        0.14069102,  0.13945375,  0.13825794,  0.13708714,  0.13591635,
        0.13475233,  0.13365203,  0.13255172,  0.13145141,  0.13039309,
        0.12936661,  0.12834013,  0.12731679,  0.12636662,  0.12541645,
        0.12446629,  0.12355607,  0.1226837 ,  0.12181133,  0.12093896,
        0.12014367,  0.11934947,  0.11855527,  0.11779656,  0.11707976,
        0.11636297,  0.11564617,  0.11499967,  0.11435838,  0.11371709,
        0.11310508,  0.11253636,  0.11196765,  0.11139893,  0.11089042,
        0.11039043,  0.10989045,  0.10941282,  0.10897701,  0.10854119,
        0.10810538,  0.10771806,  0.10734133,  0.1069646 ,  0.10660361,
        0.10628055,  0.10595749,  0.10563443,  0.10534824,  0.10507333,
        0.10479841,  0.10453366,  0.10430141,  0.10406916,  0.10383691,
        0.10363123,  0.10343637,  0.1032415 ,  0.10305258,  0.10289012,
        0.10272767,  0.10256522,  0.10242102,  0.1022864 ,  0.10215178,
        0.10202019,  0.10190925,  0.10179831,  0.10168736,  0.10158842,
        0.10149746,  0.1014065 ,  0.10131676,  0.10124253,  0.1011683 ,
        0.10109406,  0.10102743,  0.10096711,  0.10090679,  0.10084668,
        0.10079787,  0.10074905,  0.10070024,  0.10065606,  0.10061669,
        0.10057733,  0.10053796,  0.10050604,  0.1004744 ,  0.10044277,
        0.10041387,  0.10038852,  0.10036317,  0.10033782,  0.10031711,
        0.10029686,  0.10027661,  0.10025792,  0.10024179,  0.10022565,
        0.10020952,  0.10019621,  0.10018339,  0.10017057,  0.10015862,
        0.10014845,  0.10013829,  0.10012813,  0.10011966,  0.10011163,
        0.10010359,  0.10009601,  0.10008967,  0.10008333,  0.10007699,
        0.10007165,  0.10006666,  0.10006167,  0.10005692,  0.100053  ,
        0.10004908,  0.10004516,  0.10004183,  0.10003876,  0.10003569,
        0.10003273,  0.10003034,  0.10002794,  0.10002554,  0.10002348,
        0.10002161,  0.10001974,  0.10001792,  0.10001647,  0.10001501,
        0.10001356,  0.1000123 ,  0.10001117,  0.10001004,  0.10000893,
        0.10000806,  0.10000719,  0.10000631,  0.10000554,  0.10000487,
        0.10000419,  0.10000352,  0.10000299,  0.10000247,  0.10000195,
        0.10000147,  0.10000106,  0.10000065,  0.10000024,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ,
        0.1       ,  0.1       ,  0.1       ,  0.1       ,  0.1       ])


def SCFprofile_test():

    # basically checking that numpy interp hasn't changed
    data = long_profile.copy()
    result = SCFprofile(np.linspace(0,100,350), chi=.5, chi_s=.3, h_dry=15,
                        l_lat=1, mn=200, m_lat=1, phi_b=0.1, pdi=1.5, disp=False)
    assert np.allclose(result, data, atol=1e-14)


def main():
    from time import time
    start=time()
    calc_g_zs_test()
    SZdist_test()
    SCFeqns_test()
    SCFeqns_u_test()
    SCFsolve_test()
    SCFcache_test()
    SCFprofile_test()
    stop=time()
    print('All tests passed in {:.3g} seconds!'.format(stop-start))


def benchmark():
    from time import time
    from collections import OrderedDict
    start=time()
    cache=OrderedDict()
    SCFcache(0,0,1,.1,0,160.52,0,cache)
    SCFcache(0,0,1.75,.1,0,360.52,0,cache)
    print('Benchmark time:', time()-start, 'seconds.')


if __name__ == '__main__':
    main()
#    benchmark()
