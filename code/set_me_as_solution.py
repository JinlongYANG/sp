#!/usr/bin/env python
# encoding: utf-8
"""
set_me_as_solution.py  V1.0

Created by Silvia Zuffi.
Code for the dpmp algorithm is translated from Matlab code written with Jason Pacheco, Brown University.
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
import sbm_basic as ba
import particles

def fun_2(this, rankId=0):
    """
    @params: rankInd = 0 for the MAP solution
    Computes the indexes of the particles for the ranked solution, and set the body model accordingly
    Returns the log posterior (or negative energy)
    """

    best = np.zeros(this.nNodes)

    for partId in this.nodeIdx:
        I_sort = np.argsort(-1*this.b[partId]['value'], axis=None).copy()
        this.mapParticleInd[partId] = I_sort[rankId]
        best[partId] = this.b[partId]['value'][I_sort[rankId]]

    if this.verbose > 1:
        print 'MAP particles'
        print this.mapParticleInd

    # Set the object
    for i in this.nodeIdx:
        x = this.b[i]['x'][:, this.mapParticleInd[i]]
        ra, ta, za, zsa = particles.get_part_params(this.particleIdx[i], x)
        this.body.Zp[i] = za
        this.body.Zs[i] = zsa
        this.body.r_abs[i,:] = ra
        this.body.t[i,:] = ta

    i = this.torso
    p = this.mapParticleInd[i]
    for c in range(this.nCameras):
        this.camera[c][0:3] = this.b[i]['x'][this.camIdx[i][c][0:3],p]
        this.camera[c][3:6] = this.b[i]['x'][this.camIdx[i][c][3:6],p]
        this.camera[c][6] = this.b[i]['x'][this.camIdx[i][c][6],p]
        this.camera[c][7:9] = this.b[i]['x'][this.camIdx[i][c][7:9],p]
        this.camera[c][9:14]  = np.zeros(5)

    logB = best[this.torso]
    return logB


def fun(this, rankInd):
    """
    @params: rankInd = 0 for the MAP solution
    Computes the indexes of the particles for the ranked solution, and set the body model accordingly
    Returns the log posterior (or negative energy)
    Only works for the same number of particles on each node
    """

    logB = None

    # Build a matrix (nParticles x nNodes) with the max-marginal values of all nodes
    B = -np.inf + np.zeros((len(this.b[this.torso]['value']), this.nNodes))
    for i in this.nodeIdx:
        B[:,i] = this.b[i]['value']

    # Sort the matrix to find the global max
    I = np.argsort(-B, axis=None)
    B_sort = B.flatten()[I]

    # Find the unique indexes of the max-marginal (which have the same value on each node)
    _, I_unique = np.unique(B_sort, return_index=True)
    I_unique = I_unique[::-1]

    # The ranked value identifies a root node and a particle. We will traverse the tree from the selected root to find the 
    # particles on each node
    if rankInd < len(I_unique):
        v = np.where(B == B.flatten()[I[I_unique[rankInd]]])
    else:
        v = np.where(B == B.flatten()[I[I_unique[-1]]])
    p = v[0][0]
    rootNode = v[1][0]

    # Depth-first traversal
    schedule = [None]
    schedule[0] = rootNode
    visited = np.zeros((this.nNodes), dtype=np.int)
    parent = np.zeros((this.nNodes), dtype=np.int)

    while len(schedule) > 0:
        # Pop root, add children
        i = schedule[0]
        schedule = schedule[1::]
        if i in this.nodeIdx:
            visited[i] = 1
            nbrs = np.zeros((this.nNodes), dtype=np.int)
            nbrs[this.A[:,i]>=0] = 1
            children = nbrs & ~visited
            N = np.where(children>0)[0]
            schedule.extend(N)
            parent[N] = i
        else:
            continue

        # Do backtracking
        if i == rootNode:
            logB = this.b[i]['value'][p]
            this.mapParticleInd[i] = p
        else:
            # Get the message i->parent that contains the argmax information
            t = this.Q[i, parent[i]]
            ind = np.where(this.m[t]['pDst'] == this.mapParticleInd[parent[i]])[0]
            p = this.m[t]['pSrc'][ind]
            this.mapParticleInd[i] = p
    if this.verbose > 0:
        print 'mapParticleInd:'
        print this.mapParticleInd

    # Set the object 
    for i in this.nodeIdx:
        x = this.b[i]['x'][:, this.mapParticleInd[i]]
        ra, ta, za, zsa = particles.get_part_params(this.particleIdx[i], x)
        this.body.Zp[i] = za
        this.body.Zs[i] = zsa
        this.body.r_abs[i,:] = ra
        this.body.t[i,:] = ta

    return logB
    

