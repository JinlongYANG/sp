#!/usr/bin/env python
# encoding: utf-8
"""
bp_setup.py V1.0

Created by Silvia Zuffi.
Copyright (c) 2015 MPI. All rights reserved.
Copyright (c) 2015 ITC-CNR. 
Copyright (c) 2015 Silvia Zuffi. 

Max-Planck grants you a non-exclusive, non-transferable, free of charge right to use the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited.

The Software may not be reproduced, modified and/or made available in any form to any third party without Max-Planck’s prior written permission.  By downloading the Software, you agree not to reverse engineer it.

* Disclaimer of Representations and Warranties

You expressly acknowledge and agree that the Software results from basic research, is provided “AS IS”, may contain errors, and that any use of the Software is at your sole risk. MAX-PLANCK MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE SOFTWARE, NEITHER EXPRESS NOR IMPLIED, AND THE ABSENCE OF ANY LEGAL OR ACTUAL DEFECTS, WHETHER DISCOVERABLE OR NOT. Specifically, and not to limit the foregoing, Max-Planck makes no representations or warranties (i) regarding the merchantability or fitness for a particular purpose of the Software, (ii) that the use of the Software will not infringe any patents, copyrights or other intellectual property rights of a third party, and  (iii) that the use of the Software will not cause any damage of any kind to you or a third party.

See LICENCE.txt for licensing and contact information.
"""



import numpy as np

def init_schedule(this, A, nEdges):
    schedule = mpp(A>=0)
    # Add 1 as we need 0 to be absence of edge, but the part indexes start at 0
    B = A+1
    Q = (B + (np.tril(B)>0)*nEdges)-2
    return schedule, Q

def mpp(A):
    m = np.sum(np.triu(A))
    S = {'i':np.zeros((m), dtype=np.int), 'j':np.zeros((m), dtype=np.int)}
    n =-1
    Q = [None]
    print 'Setting schedule with node 3 as root (torso is node 3 in the body model)'
    Q[0] = 3

    while len(Q)>0:
        i = Q[0]
        Q = Q[1::]
        for j in np.where(A[i,:])[0]:
            n = n+1
            S['i'][n] = i
            S['j'][n] = j
            Q.append(j)
            A[i,j] = 0
            A[j,i] = 0

    R = revschedule(S, m)
    S['i'] = np.vstack((R['i'], S['i'])).flatten()
    S['j'] = np.vstack((R['j'], S['j'])).flatten()
    return S

def revschedule(S, m):

    R = {'i':np.zeros((m), dtype=np.int), 'j':np.zeros((m), dtype=np.int)}
    for u in range(m):
        i = S['i'][u]
        j = S['j'][u]
        v = m-u-1
        R['i'][v] = j
        R['j'][v] = i

    return R

