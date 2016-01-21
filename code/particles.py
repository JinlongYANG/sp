#!/usr/bin/env python
# encoding: utf-8
"""
particles.py  V1.0

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
import copy
import cv2
import sbm_basic as ba
from scipy.spatial import KDTree
import time
import scipy 
import scipy.ndimage
import numpy.random as npr
from my_mesh.mesh import myMesh
from multivariate import multivariate_normal
import set_me_as_solution

def get_indexes(nB, nBshape, nCameras, nCamParams):
    """
    Returns a dictionary of indexes for accessing the variables stored
    in a particle.
    """

    indexes = {}
    indexes['camIdx'] = [None]*nCameras
    indexes['rIdx'] = [0, 1, 2]
    indexes['OIdx'] = [3, 4, 5]
    indexes['zPoseIdx'] = range(6, 6+nB)
    indexes['zShapeIdx'] = range(indexes['zPoseIdx'][-1]+1, indexes['zPoseIdx'][-1]+1+nBshape)
    idx = indexes['zShapeIdx'][-1]+1
    if nCameras > 0:
        for c in range (nCameras):
            indexes['camIdx'][c] = range(idx, idx+nCamParams)
            idx = idx + nCamParams
    nodeDim = idx
    return indexes, nodeDim

def get_part_params(idx, x):
    """
    Given a particle and the index dictionary, returns the part parameters.
    """
    try:
        ra =  x[idx['rIdx'],:]
        ta =  x[idx['OIdx'],:]
        za =  x[idx['zPoseIdx'],:]
        zsa =  x[idx['zShapeIdx'],:]
    except:
        ra =  x[idx['rIdx']]
        ta =  x[idx['OIdx']]
        za =  x[idx['zPoseIdx']]
        zsa =  x[idx['zShapeIdx']]
    return ra, ta, za, zsa

def get_pose_params(idx, x):
    """
    Given a particle and the index dictionary, returns the pose (rotation and translation) parameters.
    """
    try:
        ra =  x[idx['rIdx'],:]
        ta =  x[idx['OIdx'],:]
    except:
        ra =  x[idx['rIdx']]
        ta =  x[idx['OIdx']]
    return ra, ta

def get_shape_params(idx, x):
    try:
        zsa =  x[idx['zShapeIdx'],:]
    except:
        zsa =  x[idx['zShapeIdx']]
    return zsa

def get_pose_def_params(idx, x):
    try:
        za =  x[idx['zPoseIdx'],:]
    except:
        za =  x[idx['zPoseIdx']]
    return za

"""
def get_camera_params(idx, x, nCameras):
    cam = [None]*nCameras
    for c in xrange(nCameras):
        cam[c] = x[idx['camIdx'][c]].copy()
    return cam
"""

def get_noise_std(dpmp, i, x):
    """
    Define the noise standard deviation to sample particles with random noise.
    """
    sigma = dpmp.particle_genericSigma*np.ones((dpmp.nodeDim[i]))
    idx = dpmp.particleIdx[i]
    sigma[idx['rIdx']] = dpmp.particle_rSigma
    sigma[idx['OIdx']] = dpmp.particle_tSigma

    sigma[idx['zPoseIdx']] = dpmp.particle_posePCAsigmaScale*dpmp.body.posePCA[i]['sigma'][0:len(idx['zPoseIdx'])]  
    sigma[idx['zShapeIdx']] = dpmp.particle_shapePCAsigmaScale*dpmp.body.shapePCA[i]['sigma'][0:len(idx['zShapeIdx'])]  
    return sigma

def set_pose_def_params(idx, x, zp):
    x[idx['zPoseIdx']] = zp[0:len(idx['zPoseIdx'])];
    return x
    
def set_shape_params(idx, x, zp):
    x[idx['zShapeIdx']] = zp[0:len(idx['zShapeIdx'])];
    return x

def set_pose_params(idx, x, r, t):
    x[idx['rIdx']] = r;
    x[idx['OIdx']] = t;
    return x

def get_from_sbm(dpmp, sbm, i):
    """
    Returns a particle that has the parameters of the sbm model
    """
    idx = dpmp.particleIdx[i]
    x = np.zeros([dpmp.nodeDim[i]])
    x[idx['rIdx']] = sbm.r_abs[i,:]
    x[idx['OIdx']] = sbm.t[i,:]
    x[[idx['zPoseIdx']]] = sbm.Zp[i][0:len(idx['zPoseIdx'])]
    x[[idx['zShapeIdx']]] = sbm.Zs[i][0:len(idx['zShapeIdx'])]
    for c in xrange(dpmp.nCameras):
        x[idx['camIdx'][c]] = dpmp.camera[c]
    return x

def get_from_params(dpmp, r_abs, t, Zp, Zs, i):
    idx = dpmp.particleIdx[i]
    x = np.zeros([dpmp.nodeDim[i]])

    if r_abs.shape[0] == dpmp.body.nParts:
        assert(t.shape[0] == dpmp.body.nParts)
        assert(len(Zp) == dpmp.body.nParts)
        assert(len(Zs) == dpmp.body.nParts)
        x[idx['rIdx']] = r_abs[i,:]
        x[idx['OIdx']] = t[i,:]
        x[[idx['zPoseIdx']]] = Zp[i][0:len(idx['zPoseIdx'])]
        x[[idx['zShapeIdx']]] = Zs[i][0:len(idx['zShapeIdx'])]
    else:
        x[idx['rIdx']] = r_abs
        x[idx['OIdx']] = t
        x[[idx['zPoseIdx']]] = Zp[0:len(idx['zPoseIdx'])]
        x[[idx['zShapeIdx']]] = Zs[0:len(idx['zShapeIdx'])]

    for c in xrange(dpmp.nCameras):
        x[idx['camIdx'][c]] = dpmp.camera[c]
    return x

def get_from_sbm_with_init_location_noise(dpmp, sbm, i):
    """
    As above, but adds noise to the location of each part. Used
    for initialization.
    """
    idx = dpmp.particleIdx[i]
    x = np.zeros([dpmp.nodeDim[i]])
    x[idx['rIdx']] = sbm.r_abs[i,:]
    x[idx['OIdx']] = npr.normal(sbm.t[i,:], dpmp.initSpringSigma*np.ones(3))
    x[[idx['zPoseIdx']]] = sbm.Zp[i][0:len(idx['zPoseIdx'])]
    x[[idx['zShapeIdx']]] = sbm.Zs[i][0:len(idx['zShapeIdx'])]
    return x

def get_from_sbm_with_init_noise(dpmp, sbm, i):
    idx = dpmp.particleIdx[i]
    x = np.zeros([dpmp.nodeDim[i]])
    x[idx['rIdx']] = npr.normal(sbm.r_abs[i,:], dpmp.initSpringSigma*np.ones(3))
    x[idx['OIdx']] = npr.normal(sbm.t[i,:], dpmp.initSpringSigma*np.ones(3))
    no = len(idx['zPoseIdx']);
    ns = len(idx['zShapeIdx']);
    shapeSigma = dpmp.body.shapePCA[i]['sigma'][0:ns]
    poseSigma = dpmp.body.posePCA[i]['sigma'][0:no]
    x[[idx['zPoseIdx']]] = npr.normal(np.zeros((no)), poseSigma)
    x[[idx['zShapeIdx']]] = npr.normal(np.zeros((ns)), shapeSigma)
    return x

def compute_likelihood(dpmp, part, x, return_normals_cost=False, returnRegionScores=False):
    return compute_3D_likelihood(dpmp, part, x, return_normals_cost)

def compute_3D_likelihood(dpmp, part, x, return_normals_cost=False):
    """
    Computes the likelihood for the particles x (dim * nParticles) of the part part
    """

    nParticles = x.shape[1]
    logL = np.zeros((nParticles))
    zScore = np.zeros((nParticles))
    mean=np.zeros(dpmp.nB[part])
    sigma=dpmp.body.posePCA[part]['sigma'][0:dpmp.nB[part]]
    cov = sigma**2
    nScore = np.zeros((nParticles))
    if dpmp.likelihoodAlpha[part] == 0:
        return dpmp.likelihoodAlpha[part]*logL

    for p in xrange(nParticles):
        Pw = particle_to_points(dpmp, x[:,p], part)
        Pws = Pw[0::dpmp.resolStep,:]
        out = dpmp.kdtree.query(Pw[0::dpmp.resolStep,:])

        # Also have a cost for matching the normals
        if dpmp.compute_normals_cost:
            normals = dpmp.scanMeshNormals[out[1],:]
            mesh = myMesh(v=Pw, f=dpmp.body.partFaces[part])
            K = mesh.estimate_vertex_normals()
            L = K[0::dpmp.resolStep,:]
            angle = np.arccos(np.sum(normals*L, axis=1))
            # Penalty for normals with opposite direction. Arccos returns the angle in [0,pi]
            opp = np.where( angle > 3*np.pi/4 )
            nScore[p] = - 0.005*len(opp[0]) 

        # Robust function 
        logL[p] = -np.mean((out[0]**2+dpmp.robustOffset)**dpmp.robustGamma)

    logL = logL + zScore + nScore
    if return_normals_cost:
        return dpmp.likelihoodAlpha[part]*(logL), nScore 
    else:
        return dpmp.likelihoodAlpha[part]*logL
            
def particle_to_points(dpmp, x, i):
    idx = dpmp.particleIdx[i]
    Zp = x[[idx['zPoseIdx']]]
    Zs = x[[idx['zShapeIdx']]]
    T = x[[idx['OIdx']]]
    P = dpmp.body.get_part_mesh(i, Zp, Zs)
    r_abs = x[[idx['rIdx']]]
    R, J = cv2.Rodrigues(r_abs)
    Pw = ba.object_to_world(P, R, T)
    return Pw

def particle_to_points_local(dpmp, x, i):
    idx = dpmp.particleIdx[i]
    Zp = x[[idx['zPoseIdx']]]
    Zs = x[[idx['zShapeIdx']]]
    T = x[[idx['OIdx']]]
    P = dpmp.body.get_part_mesh(i, Zp, Zs)
    r_abs = x[[idx['rIdx']]]
    R, J = cv2.Rodrigues(r_abs)
    return P, R, T, Zs, Zp

