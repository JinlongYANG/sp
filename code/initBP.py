#!/usr/bin/env python
# encoding: utf-8
"""
initBP.py  V1.0

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
import numpy.random as npr
import get_sample_from_model
import particles as particles
import sbm_basic as ba
import scipy
from scipy import io
import os
import particles
from bp_setup import init_schedule
import cv2
try:
    from body.mesh.meshviewer import MeshViewer
    from body.mesh.mesh import Mesh
except ImportError:
    print 'visualization not supported'


def init(this):

    this.schedule, this.Q = init_schedule(this, this.A, this.nEdges)
    # Initialize the node data (particles, likelihood and max marginal value)
    this.b = [None]*this.nNodes
    this.new_b = [None]*this.nNodes

    for i in this.nodeIdx:
       this.b[i] = {'x':np.zeros((this.nodeDim[i],this.nParticles[i])), 'L':np.zeros((this.nParticles[i])), 'value':np.zeros((this.nParticles[i]))}
       this.new_b[i] = {'x':np.zeros((this.nodeDim[i],this.nParticles[i])), 'L':np.zeros((this.nParticles[i])), 'value':np.zeros((this.nParticles[i]))}

    # Particles initialization
    init_particles(this)

    # Beliefs initialization
    for i in this.nodeIdx:
       this.b[i]['L'] = particles.compute_likelihood(this, i, this.b[i]['x'])
       this.b[i]['value'] = this.b[i]['L']

    # Messages initialization
    init_messages(this)

    return

def init_particles(this):
# Initialize the particles on each node

    maxNp = np.max(this.nParticles)
    if this.display > 3:
         ms = [Mesh(v=this.scanMesh.v, f=this.scanMesh.f).set_vertex_colors('SeaGreen')]

    # Number of particles on each node
    n = np.zeros((this.nNodes))

    k = 0
    for p in range(int(maxNp)):
        Pw, r_abs, t, Zp, Zs = get_sample_from_model.get_sample(this.body, this.nB[this.torso], this.nB, this.nBshape[this.torso], add_global_rotation=this.init_with_global_rotation, init_torso_rotation=this.init_torso_rotation)

        this.body.r_abs = r_abs
        t = t + this.init_torso_location 
        this.body.t = t
        for i in this.nodeIdx:
            this.body.Zs[i] = Zs
        this.body.Zp = Zp

        for i in this.nodeIdx:
            if n[i] < this.nParticles[i]:
                this.b[i]['x'][:,p] = particles.get_from_sbm_with_init_location_noise(this, this.body, i)
                n[i] = n[i]+1

        if this.display > 4:
            partSet = this.body.partSet
            P = ba.get_sbm_points_per_part(this.body)
            for part in partSet:
                ms.append(Mesh(v=P[part], f=this.body.partFaces[part]).set_vertex_colors(this.body.colors[part]))

    if this.display > 4:
         mv = MeshViewer()
         mv.set_static_meshes(ms)



def init_messages(this):
    maxMsgIdx = this.Q.max()
    this.m = [None]*(maxMsgIdx+1)
    for m in range(len(this.schedule['i'])):
        i = this.schedule['i'][m]
        j = this.schedule['j'][m]
        if i in this.body.partSet and j in this.body.partSet:
            t = this.Q[i,j]
            this.m[t] = {'value':np.ones((this.nParticles[j])), 'x': this.b[j]['x']}


def init_based_on_last_frame(this, lastResult):
    this.schedule, this.Q = init_schedule(this, this.A, this.nEdges)
    # Initialize the node data (particles, likelihood and max marginal value)
    this.b = [None]*this.nNodes
    this.new_b = [None]*this.nNodes

    for i in this.nodeIdx:
       #print ' ###*** nParticles ' +str(i) + ': ' + str(this.nParticles[i])
       this.b[i] = {'x':lastResult.new_b[i]['x'][:,:this.nParticles[i]], 'L':np.zeros((this.nParticles[i])), 'value':np.zeros((this.nParticles[i]))}
       #print ' *** x: ' + str(this.b[i]['x'].shape)
       #print ' *** L: ' + str(this.b[i]['L'].shape)
       #print ' *** v: ' + str(this.b[i]['value'].shape)
       this.new_b[i] = {'x':lastResult.new_b[i]['x'][:,:this.nParticles[i]], 'L':np.zeros((this.nParticles[i])), 'value':np.zeros((this.nParticles[i]))}

    # Particles initialization
    init_particles_based_on_last_frame(this, lastResult)

    # Beliefs initialization
    for i in this.nodeIdx:
       this.b[i]['L'] = particles.compute_likelihood(this, i, this.b[i]['x'])
       this.b[i]['value'] = this.b[i]['L']

    # Messages initialization
    init_messages(this)

    return

def init_particles_based_on_last_frame(this, lastResult):
# Initialize the particles on each node

    maxNp = np.max(this.nParticles)
    if this.display > 3:
         ms = [Mesh(v=this.scanMesh.v, f=this.scanMesh.f).set_vertex_colors('SeaGreen')]

    # Number of particles on each node
    n = np.zeros((this.nNodes))

    k = 0
    for p in range(int(maxNp)):
        Pw, r_abs, t, Zp, Zs = get_sample_from_model.get_sample(lastResult.body, lastResult.nB[lastResult.torso], lastResult.nB, lastResult.nBshape[lastResult.torso], add_global_rotation=lastResult.init_with_global_rotation, init_torso_rotation=lastResult.init_torso_rotation)

        this.body.r_abs = r_abs
        t = t + lastResult.init_torso_location 
        this.body.t = t
        for i in this.nodeIdx:
            this.body.Zs[i] = Zs
        this.body.Zp = Zp

        for i in this.nodeIdx:
            if n[i] < this.nParticles[i]:
                this.b[i]['x'][:,p] = particles.get_from_sbm_with_init_location_noise(lastResult, lastResult.body, i)
                n[i] = n[i]+1

        if this.display > 4:
            partSet = lastResult.body.partSet
            P = ba.get_sbm_points_per_part(lastResult.body)
            for part in partSet:
                ms.append(Mesh(v=P[part], f=lastResult.body.partFaces[part]).set_vertex_colors(lastResult.body.colors[part]))

    if this.display > 4:
         mv = MeshViewer()
         mv.set_static_meshes(ms)



