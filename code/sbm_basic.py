#!/usr/bin/env python
# encoding: utf-8
"""
sbm_basic.py  V1.0

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
import numpy.linalg as npl
import numpy.random as npr
import pickle as pkl
try:
    from body.mesh.meshviewer import MeshViewer
    from body.mesh.mesh import Mesh
except ImportError:
    print 'visualization not supported'

import cv2
from scipy.spatial.distance import cdist
import time

def compute_conditioned_gaussian(this, aInd, bInd, mu, C, x):
    """
    Computes the mean and Covariance of the conditioned Gaussian of the one
    defined by mu and C, given the conditioning values x. bInd are the indexes
    of the conditioning variables, aInd are the indexes of the conditioned values
    (we are assuming also a marginalization can be specified through the aInd)
    """

    Caa = C[aInd][:,aInd]
    Cbb = C[bInd][:,bInd]
    Cab = C[aInd][:,bInd]
    Cba = C[bInd][:,aInd]
    try:
        C_ab = Caa - Cab * (np.matrix(Cbb).I * Cba)
    except:
        import pdb
        pdb.set_trace()
    mu_ab = mu[:,aInd].T + Cab * np.matrix(Cbb).I * (x - mu[:,bInd]).T
    mu_ab = np.array(mu_ab)[:,0]
    return mu_ab, C_ab


def get_part_mesh(this, part, Zp=None, Zs=None):
    # R.I*j = j0
    # Returns the points P(nx3) of the part in local frame for PCA pose and
    # shape coefficients Zp and Zs respectively

    if this.fixedShape:
        A = this.shapePCA[part]['T']+ this.shapePCA[part]['M']+this.posePCA[part]['M']
        P = np.zeros((len(A)/3, 3))
        P[:,0] = A[0::3]
        P[:,1] = A[1::3]
        P[:,2] = A[2::3]
        return P

    if Zs is None:
        Tp = this.shapePCA[part]['T']
    else:
        nB_s = len(Zs)
        B = this.shapePCA[part]['B']
        # The template for the pose model is built with the PCA for shape
        Tp = this.shapePCA[part]['T'] + this.shapePCA[part]['M'] + np.array(B[:,0:nB_s]*np.matrix(Zs).T)[:,0]

    M = this.posePCA[part]['M']+Tp
    B = this.posePCA[part]['B']

    if Zp is None:
        A = M
    else:
        nB_p = len(Zp)
        A = M + np.array(B[:,0:nB_p]*np.matrix(Zp).T)[:,0]

    P = np.zeros((len(A)/3, 3))
    P[:,0] = A[0::3]
    P[:,1] = A[1::3]
    P[:,2] = A[2::3]

    return P

def get_part_meshes(this, part, Zp, Zs):
    # Can compute more than a part (PCA coefficients are matrices)
    # Zp is a matrix of pca pose coefficients, nB x nParts
    # Zs is a matrix of pca shape coefficients, nB x nParts
    # Returns P(nPoints,3,nParts)

    if this.fixedShape:
        nM = Zp.shape[1]
        M = this.shapePCA[part]['T']+ this.shapePCA[part]['M']+this.posePCA[part]['M']
        A = np.matrix(M).T
        P = np.zeros((A.shape[0]/3, 3, nM))
        P[:,0,:] = A[0::3,:]
        P[:,1,:] = A[1::3,:]
        P[:,2,:] = A[2::3,:]
        return P

    nB_s = Zs.shape[0]   
    nB_p = Zp.shape[0]   
    nM = Zp.shape[1]

    M = this.shapePCA[part]['M']+this.shapePCA[part]['T']
    B = this.shapePCA[part]['B']
    T = np.matrix(M).T + np.array(B[:,0:nB_s]*np.matrix(Zs))

    M = np.matrix(this.posePCA[part]['M']).T+T
    B = this.posePCA[part]['B']

    A = np.matrix(M) + np.array(B[:,0:nB_p]*np.matrix(Zp))
    P = np.zeros((len(A)/3, 3, nM))
    P[:,0,:] = A[0::3,:]
    P[:,1,:] = A[1::3,:]
    P[:,2,:] = A[2::3,:]

    return P


def object_to_world(P, R, T):
    # Computes the points in global frame with the transformation R,T
    # Input points are P(nx3), R(3x3) and T is a 3-dim vector

    if np.prod(T.shape)==3: 
        Pw = np.matrix(R)*P.T + np.matrix(T.flatten()).T
        Pw = np.array(Pw.T)

    else:
        # Case where the vector is higher dimensional
        Pw = P.copy()
        n = P.shape[0]
        m = P.shape[2]
        R11 = np.tile(R[0,0,:], [n,1])
        R12 = np.tile(R[1,0,:], [n,1])
        R13 = np.tile(R[2,0,:], [n,1])
        R21 = np.tile(R[0,1,:], [n,1])
        R22 = np.tile(R[1,1,:], [n,1])
        R23 = np.tile(R[2,1,:], [n,1])
        R31 = np.tile(R[0,2,:], [n,1])
        R32 = np.tile(R[1,2,:], [n,1])
        R33 = np.tile(R[2,2,:], [n,1])

        tx = np.reshape(np.tile(T[0,:], [n,1]), [n,m])
        ty = np.reshape(np.tile(T[1,:], [n,1]), [n,m])
        tz = np.reshape(np.tile(T[2,:], [n,1]), [n,m])

        Pw[:,0,:] = np.array(P[:,0,:])*R11 + np.array(P[:,1,:])*R21 + np.array(P[:,2,:])*R31 + tx
        Pw[:,1,:] = np.array(P[:,0,:])*R12 + np.array(P[:,1,:])*R22 + np.array(P[:,2,:])*R32 + ty
        Pw[:,2,:] = np.array(P[:,0,:])*R13 + np.array(P[:,1,:])*R23 + np.array(P[:,2,:])*R33 + tz
    return Pw

def world_to_object(Pw, R, T):
    # Converts points in global frame in the frame specified by R,T

    P = np.matrix(R).I*np.matrix((Pw - T)).T
    #P = P.T
    return P.T

def get_rotation_from_3D_points(P, Pw):
    R, _ = rigid_transform_3D(P, Pw)
    return R


def align_to_parent(this, part, parent, P, Pparent, R0):
    # Computes the transformation to align a part expressed in local frame
    # to its parent expressed in global frame. Also computes an alignment cost

    cl = this.interfacePointsFromTo[part][parent]
    clp = this.interfacePointsFromTo[parent][part]

    if R0 is None:
        R, T = rigid_transform_3D(P[cl,:], Pparent[clp,:])
    else:
        R = R0
        T = Pparent[clp,:] - P[cl,:]*np.matrix(R).T
        T = np.mean(T,0)

    P1 = P[cl,:]
    P2 = Pparent[clp,:]
    dsq = np.sum((P1-P2)**2,1)
    cost = np.sqrt(np.mean(dsq))

    return R, T, cost

def rigid_transform_3D(A, B):
    # Computes the rigid transformation to align point sets A and B
    # Note: code is for matching the Matlab implementation, but we are going to 
    # use a different rotation matrix (transpose) instead

    c_A = np.matrix(np.mean(A,0))
    c_B = np.matrix(np.mean(B,0))
    A = np.matrix(A)
    B = np.matrix(B)
    H = (A - c_A).T * (B - c_B)

    U,S,V = npl.svd(H)
   
    # To have the same computation in Matlab 
    # The numpy svd is H=U*diag(S)*V and Matlab is H=U*S*V'
    V = np.matrix(V).T
    R = V*np.matrix(U).T

    if npl.det(R) < 0:
        V[:,2] = -1.0*V[:,2]
        R = V*np.matrix(U).T

    T = np.array(-R*c_A.T+c_B.T)
    return R, T

def show_mesh(this, P, dbstop=False, mesh=None, scan=None, filename=None, Pwo=None):
    # P is a list of parts

    mv = MeshViewer()
    #mv.set_background_color(np.array([1.0, 1.0, 1.0]))
    ms = [Mesh(v=P[part], f=this.partFaces[part]).set_vertex_colors('light blue')  for part in this.partSet]

    if mesh is not None:
        ms2 = [Mesh(v=mesh[part], f=this.partFaces[part]).set_vertex_colors('SeaGreen') for part in this.partSet]
        ms = ms + ms2
    if scan is not None:
        s = [Mesh(v=scan.v, f=scan.f).set_vertex_colors('firebrick')]
        ms = ms + s
    if Pwo is not None:
        ms2 = [Mesh(v=Pwo[part], f=this.partFaces[part]).set_vertex_colors('turquoise3') for part in this.partSet]
        ms = ms + ms2

    mv.set_static_meshes(ms)
    if filename is not None:
        time.sleep(1)
        mv.save_snapshot(filename, blocking=True)

    if dbstop:
        import pdb
        pdb.set_trace()
    else:
        time.sleep(4)

def get_part_mesh_from_parameters(this, part, t, r_abs, Zs=None, Zp=None):

    P = get_part_mesh(this, part, Zp, Zs)
    R, _ = cv2.Rodrigues(r_abs)
    Pw = object_to_world(P, R, t)
    return Pw

def get_body_mesh_from_parameters(this, r_abs, Zs, Zp, t=None):

    Pw  = [None]*this.nParts
    if t is None:
        t = np.zeros((this.nParts, 3))
        for part in this.partSet:
            parent = this.parent[part]
            P = get_part_mesh(this, part, Zp[part], Zs[part])
            R, J = cv2.Rodrigues(r_abs[part, :])
            if parent >= 0:
                Runused, T, cost = align_to_parent(this, part, parent, P, Pw[parent], R)
            else:
                T = np.zeros((3))

            Pw[part] = object_to_world(P, R, T)
            t[part,:] = T.flatten()
    else:
        for part in this.partSet:
            parent = this.parent[part]
            P = get_part_mesh(this, part, Zp[part], Zs[part])
            R, J = cv2.Rodrigues(r_abs[part, :])
            T = t[part, :]
            Pw[part] = object_to_world(P, R, T)

    return Pw, t


def get_sbm_points_per_part(this):
    Pw  = [None]*this.nParts
    for part in this.partSet:
        parent = this.parent[part]
        P = get_part_mesh(this, part, this.Zp[part], this.Zs[part])
        R, J = cv2.Rodrigues(this.r_abs[part, :])
        T = this.t[part, :]
        Pw[part] = object_to_world(P, R, T)
    return Pw


def show_me(this, dbstop=False, mesh=None, scan=None, filename=None, Pwo=None, camera=None):

    # Generate the mesh
    Pw = get_sbm_points_per_part(this)
    if camera is not None:
        R,_ = cv2.Rodrigues(camera)
        for i in this.partSet:
            Pw[i] = np.matrix(Pw[i])*np.matrix(R)
            Pw[i] = np.array(Pw[i])
    show_mesh(this, Pw, dbstop=dbstop, mesh=mesh, scan=scan, filename=filename, Pwo=Pwo)


def sbm_to_scape_mesh(this):

    # Load the template, so we have the faces
    data = this.load_sample(this.template_filename + '_' + this.gender + '.pkl')
    v = data['points']
    f = data['tri']

    Pw  = [None]*this.nParts
    for part in this.partSet:
        P = get_part_mesh(this, part, this.Zp[part], this.Zs[part])
        R, J = cv2.Rodrigues(this.r_abs[part, :])
        T = this.t[part, :]
        Pw[part] = object_to_world(P, R, T)
        pidx = this.part2bodyPoints[part]
        v[pidx,:] = Pw[part]

    return v, f

