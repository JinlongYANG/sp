#!/usr/bin/env python
# encoding: utf-8
"""
update_all_messages.py  V1.0

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
import sbm_basic as ba
import cv2
import particles
import time

def update(this):
    """
    Run BP: computes all the messages in the messages schedule
    """
    T = 0
  
    for m in xrange(len(this.schedule['i'])):
        i = this.schedule['i'][m]
        j = this.schedule['j'][m]
        if i in this.body.partSet and j in this.body.partSet:
            t = this.Q[i,j]
            preMessage = compute_pre_message(this, i, j, this.b, this.m)
            tic = time.time()
            this.m[t], _ = compute_message(this, i, j, this.b[j]['x'], preMessage)
            toc = time.time()
            T = T + toc - tic

    # Compute the max-marginal on each node
    for v in this.nodeIdx:
        compute_node_belief(this, v)
    if this.verbose > 0:
        print 'time for messages: ' + str(T)

def compute_pre_message(this, i, j, b, M):
    """
    Computes the pre-message for the message from i to j, which is the
    product of the incoming messages to i, excluding the message from j
    @params this: a Dpmp class object
    @params i: source node
    @params j: destination node
    @params b: the local data for BP (particles, likelihood and max-marginals)
    @params M: list of all current messages
    """

    # Temporarily remove the link i,j from the adjiacency matrix
    u = this.A[j,i]
    this.A[j,i] = -1

    # Find the neighbors and put back the link
    ks = np.where(this.A[:,i] >=0)[0]
    this.A[j,i] = u

    # Define the messages product (in log space)
    m = {'value': np.zeros((this.nParticles[i])), 'x':b[i]['x']}
    messageProduct = np.zeros((this.nParticles[i]))
    for k in ks:
        if k in this.nodeIdx:
            t = this.Q[k,i]
            messageProduct = messageProduct + M[t]['value']
            #DEBUG check the messages are over the same set of particles
            #D = np.abs(M[t]['x'] - b[i]['x'])
            #assert D.max() == 0

    m['value'] = messageProduct + b[i]['L'].copy()
    return m

def compute_message(this, src, dst, xDst, preMessage):
    """
    Computes the message in log space as the sum of the pre-message and the pairwise potential
    @params src: source node
    @params dst: destination node
    @params xDst: destination particles
    @params preMessage: pre-message, contains the value and the source particles
    """

    xSrc = preMessage['x']
    Nsrc = xSrc.shape[1]
    Ndst = xDst.shape[1]

    # The pairwise potential is a negative cost for bringing parts together and a camera similarity potential
    stitch_potential = compute_pairwise_stitch_potential(this, src, dst, xSrc, xDst)

    # Sum the pre-message to every columns of the pairwise potential
    MM = np.vstack([preMessage['value']]*Ndst)
    logPsi = MM.T + stitch_potential 

    value = np.max(logPsi, axis=0) 
    pSrc = np.argmax(logPsi, axis=0)
    m = {'value': value, 'logPsi': logPsi, 'pSrc':pSrc, 'x':xDst, 'pDst':xrange(Ndst)}

    return m, logPsi

def compute_pairwise_stitch_potential(this, a, b, xSrc, xDst):
    """
    Computes the pairwise stitch cost for two connected parts
    @params a: parent part
    @params b: child part
    @params xSrc: set of Na particles of the parent
    @params xDst: set of Nb particle of the child
    Return negative cost(Na x Nb) matrix
    """

    Na = xSrc.shape[1]
    Nb = xDst.shape[1]

    # Computes the vertexes of the part a in global coordinates
    [ra, ta, za, zsa ] = particles.get_part_params(this.particleIdx[a], xSrc)

    Pa = ba.get_part_meshes(this.body, a, za, zsa)
    Ra = np.zeros((3,3,Na))
    for i in xrange(Na):
        Ra[:,:,i], J  = cv2.Rodrigues(ra[:,i])
    Paw = ba.object_to_world(Pa, Ra, ta)

    # Computes the vertexes of the part b in global coordinates
    [rb, tb, zb, zsb ] = particles.get_part_params(this.particleIdx[b], xDst)

    Pb = ba.get_part_meshes(this.body, b, zb, zsb)
    Rb = np.zeros((3,3,Nb))
    for i in xrange(Nb):
        Rb[:,:,i], J  = cv2.Rodrigues(rb[:,i])
    Pbw = ba.object_to_world(Pb, Rb, tb)

    cl = this.body.interfacePointsFromTo[b][a]
    clp = this.body.interfacePointsFromTo[a][b]
    clf = this.body.interfacePointsFlex[a][b]
    if len(Pbw.shape) == 3:
        p2 = Pbw[cl,:,:]
        p1 = Paw[clp,:,:]
        n1 = p1.shape[2]
        n2 = p2.shape[2]
        cost = np.zeros((n1,n2))

        P1 = np.dstack([p1[:,0,:]] * n2)
        P2 = np.dstack([p2[:,0,:]] * n1)
        dx = np.transpose(P1, [0, 2, 1]) - P2

        P1 = np.dstack([p1[:,1,:]] * n2)
        P2 = np.dstack([p2[:,1,:]] * n1)
        dy = np.transpose(P1, [0, 2, 1]) - P2

        P1 = np.dstack([p1[:,2,:]] * n2)
        P2 = np.dstack([p2[:,2,:]] * n1)
        dz = np.transpose(P1, [0, 2, 1]) - P2

        Ta = np.array([ta,]*n2).T
        Tb = np.array([tb,]*n1)
        dT = (Ta - Tb)**2
        St = np.sqrt(np.sum(dT, axis=1))

    else:
        p2 = Pbw[cl,:]
        p1 = Paw[clp,:]
        dx = p1[:,0] - p2[:,0]
        dy = p1[:,1] - p2[:,1]
        dz = p1[:,2] - p2[:,2]
        dT = (ta - tb)**2
        St = np.sqrt(np.sum(dT, axis=0))

    # Setting a penalty for part centers to be close penalizes interpenetration but only for parts that have similar length (arms, legs)
    S = np.array(St < 0.05, dtype=int)
    dsq = (dx**2) + (dy**2) + (dz**2)
    weights = np.ones((dsq.shape))
    N = np.sum(weights)
    weights[clf] = 0.8
    Nw = np.sum(weights)
    weights = weights * N / Nw 
    dsq = dsq * weights
    cost = np.mean(dsq, axis=0)

    # Add a penalty for a total reflection of the parts, that would have a low cost, but
    # we should penalize for inter-penetration
    
    return -this.stitchAlpha[a,b] *( cost.T + S ) 


def compute_model_log_posterior(this):
    """"
    Computes the energy of the current sbm body model in the dpmp class (this.body)
    """
    body = this.body

    # Compute particles and likelihood
    X = [None]*this.nNodes
    logL = np.zeros([this.nNodes])
    for part in body.partSet:
        x = particles.get_from_sbm(this, body, part)
        X[part] = np.asarray([x]).T
        logL[part], D = particles.compute_likelihood(this, part, X[part], return_normals_cost=True)
        if this.verbose > 1:
            print 'part ' + this.body.names[part] + ' likelihood: ' + str(logL[part]) + ' normals cost: ' + str(D)

    logPair = np.zeros([body.nPairs])
    for k, pair in enumerate(body.pairs):
        logPair[k] = compute_pairwise_stitch_potential(this, pair[0], pair[1], X[pair[0]], X[pair[1]])
        if this.verbose > 1:
            print 'pair ' + this.body.names[pair[0]] + ' ' + this.body.names[pair[1]] + ' stitch :' +  str(logPair[k]) 

    logLs = np.sum(logL)
    # The pairwise stitch cost and the camera potentials are counted 2 times
    logP = np.sum(logPair)/2
    logPos = logLs + logP 
    return logPos, logLs, logP


def compute_node_belief(this, i):

    # Find the neighbors of the node i
    ks = np.where(this.A[:,i] >=0)[0]
    assert len(ks)>0

    messageProduct = np.zeros((this.nParticles[i]))
    for k in ks:
        if k in this.nodeIdx:
            t = this.Q[k,i]
            #DEBUG check that the messages are defined over the same set of particles
            err = this.b[i]['x'] - this.m[t]['x']
            err = err.T * np.matrix(err)
            assert err.max() == 0
            messageProduct = messageProduct.copy() + this.m[t]['value']

    this.b[i]['value'] = messageProduct + this.b[i]['L']
    
