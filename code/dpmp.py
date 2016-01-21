#!/usr/bin/env python
# encoding: utf-8
"""
dpmp.py  V1.0

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
import particles
import pickle as pkl

class Dpmp(object):

    def __init__(self, body, nBtorso, nB, nBshape, nParticles, nCameras, modelName):
        """
        @params: body Is an instance of the class Sbm
        @params: nBtorso Is the number of basis components for the torso for the pose model
        @params: nB Pose components for the other parts
        @params: nBshape Number of components for the shape pca model for all the parts
        @params: nParticles Is the number of particles on each node
        """
        self.body = body
        self.body.load_model(modelName)

        self.nNodes = self.body.nParts

        # Defines if the type of proposal for resampling the particles is randomly selected per particle or per node
        self.proposal_per_particle = True
        
        # When doing the message based selection, do we want to force keeping the MAP particle?
        self.keep_MAP_in_selection = False

        # Resolution to speed-up 3D likelihood computation and LM
        self.resolStep = 10
        self.LMsteps = 5 

        # If to sample initial particles with a torso rotation around the vertical axis
        self.init_with_global_rotation = False

        # Level of verbosity
        self.verbose = 1
        self.display = -1
        self.display_interval = 1

        # Temperature scaling for message-based selection
        self.Tmsg = 1.0

        # Exponent for the weak prior and likelihood (multiplicative factors as we work in log space)
        self.priorAlpha = 1.0
        self.stitchAlpha = 0.25*np.ones((self.nNodes, self.nNodes))
        self.likelihoodAlpha = np.ones(self.nNodes)

        # For the part p, self.neighbors[p] is the set of neighbors 
        self.neighbors = [ [2,13],[15],[0],[18,16,9,8,7],[7,12],[10],[],[3,4],[3,10],[3,11],[5,8],[9,15],[4],[0,16],[],[11,1],[3,13],[],[3]]
        self.torso = 3

        # Number of basis components
        self.nB = nB*np.ones(self.nNodes, dtype=np.int)
        self.nB[self.torso] = nBtorso
        self.nBshape = nBshape*np.ones(self.nNodes, dtype=np.int)

        # Proposals
        self.probRandomWalk = 0.5
        self.probUniform = 0

        # Parameters for the springs between vertexes of neighboring interfaces, used to add noise when resampling from neighbors
        self.springSigma = 0 

        # To assign noise to the samples used to initialize the particles
        self.initSpringSigma = 0.08 

        self.nParticles = nParticles*np.ones((self.nNodes), dtype=np.int)

        # Parameters for particles resampling
        self.particle_genericSigma = 0.001
        self.particle_rSigma = 0.2
        self.particle_tSigma = 0.1
        self.particle_posePCAsigmaScale = 0.1
        self.particle_shapePCAsigmaScale = 0.1

        self.mapParticleInd = -1*np.ones((self.nNodes), dtype=np.int)

        self.nCameras = 0
        self.nCamParams = 0

        self.scanMesh = None
        self.scanMeshNormals = None

        self.nSteps = 0
        self.step = -1

        # Set the indexes of the particles variables
        self.particleIdx = [None]*self.nNodes
        self.nodeDim = [None]*self.nNodes
        for i in range(self.nNodes):
            self.particleIdx[i], self.nodeDim[i] = particles.get_indexes(self.nB[i], self.nBshape[i], self.nCameras, self.nCamParams)

        self.define_graph()

        return

    def define_graph(self):
        # Define the adjacency matrix. The value of the matrix is the part index
        self.A = -1*np.ones((self.nNodes,self.nNodes), dtype=np.int)
        for (p, val) in enumerate(self.body.pairs):
             self.A[val[0]][val[1]] = p+1
        self.nodeIdx = self.body.partSet

        # Define the set of edges
        I = np.where(self.A>0)
        self.nEdges = len(I[0])
        self.E = np.zeros((self.nEdges, 2), dtype=np.int)
        for p in range(self.nEdges):
            self.E[p] = [I[0][p], I[1][p]]


def save_dpmp(d, logB, params, mesh_data, filename=None):
    """
    Saves a file with the solution and optimization parameters
    @params: d is an instance of the class Sbm
    @params: logB is the final value of the energy
    @params: params are the model and optimization parametery
    @params: mesh_data contains two fields: v, the vertices, and f, the faces of the mesh
    @params: filename is the name of the file to write
    """

    if filename is None:
        filename = str(time.time())+'.pkl'
    out_file = open(filename, 'w+')
    dic = {'r_abs': d.body.r_abs, 't':d.body.t, 'Zp': d.body.Zp, 'Zs': d.body.Zs, 'logB':logB, 'params':params, 'mesh_data':mesh_data}
    pkl.dump(dic, out_file)
    out_file.close()
    print 'Wrote to disk %s ' % filename

def read_dpmp(filename):
    """
    Reads a pkl file 
    """
    print 'loading ' + filename
    with open(filename) as f:
        data = pkl.load(f)
        f.close()
    return data

def show_result(filename):
    """
    Reads a file containing a solution and shows the mesh. 
    """
    try:
        from body.mesh.mesh import Mesh
        data = read_dpmp(filename)
        ms = Mesh(v=data['mesh_data']['v'], f=data['mesh_data']['f'])
        ms.show()
    except ImportError:
        print 'visualization not supported'

