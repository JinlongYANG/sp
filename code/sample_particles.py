#!/usr/bin/env python
# encoding: utf-8
"""
sample_particles.py  V1.0

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
import sbm_basic as  ba
import particles
import cv2
import time
from particles import particle_to_points

MT_RND_ALL = 1
MT_RND_Z_POSE = 2
MT_NBR_Z_PARENT_COND = 3
MT_NBR_Z_PARENT_AND_ANGLE_COND = 4
MT_FROM_NBRS = 5
MT_FROM_SCAN_DATA = 6

def sample_particles(this, proposal_type):
    """
    Sample particles. Generates a new set of particles for each node and stores them in this.new_b[]['x']
    The proposal type can be specified, and in this case it is the same for all nodes and all particles,
    otherwise it is resampled for each particle.
    """

    funDict = {'RANDWALK':sample_from_randwalk, 'NBR':sample_from_nbr, 'UNIFORM':sample_from_uniform}
    movedTypes = [None]*this.nNodes 

    # Resample all the particles, so we double their number 
    toc = 0
    for v in this.nodeIdx:
        new_p = this.b[v]['x'].copy()
        movedTypes[v] = np.zeros((this.nParticles[v]), dtype=np.int)
     
        for p_i in xrange(this.nParticles[v]):
            if this.proposal_per_particle:
                randVal = npr.rand()
                if randVal <= (1.0 - this.probRandomWalk - this.probUniform):
                    augment_type = 'NBR'
                else:
                    if randVal <= (1.0 - this.probUniform):
                        augment_type = 'RANDWALK'
                    else:
                        augment_type = 'UNIFORM'
                proposal_type = augment_type

            proposedParticle, movedType = funDict[proposal_type](this, v, this.b[v]['x'], p_i)

            # If the likelihood is 3D (for alignment), run few steps of LM optimization after the resampling.
            # Note that the cost that LM optimizes is not the likelihood (quadratic cost vs robust function)
            # so the likelihood can decrease in practice. 
            if this.likelihoodType == '3Dlikelihood':
                tic = time.time()
                if this.LMsteps > 0:
                    if this.body.fixedShape:
                        proposedParticle = move_with_LM_only_pose(this, v, proposedParticle)
                    else:
                        proposedParticle = move_with_LM(this, v, proposedParticle)
                toc = toc + time.time() - tic

            new_p[:, p_i] = proposedParticle.copy()
            movedTypes[v][p_i] = movedType
        
        this.new_b[v]['x'] = new_p.copy()

    if this.likelihoodType == '3Dlikelihood' and this.verbose > 0:
        print 'time for LM: ' + str(toc)
      
    return movedTypes

def sample_from_randwalk(this, i, x, p_i):
    """
    @params: this, a Dpmp object
    @params: i, integer, the index of the node or part
    @params: x, (D,N) array of particles, where D is the particle dimension and N the number of particles
    @params: p_i, integer, the index of the particle we take the random walk from
    """

    if this.use_map_particle_for_rnd_walk:
        p = this.mapParticleInd[i]
    else:
        p = p_i
    p_value = 0.5
    
    if npr.rand() > p_value:
        movedType = MT_RND_ALL
        sigma = particles.get_noise_std(this, i, x[:,p])
        proposedParticle = npr.normal(x[:,p], sigma)

        # Generate a random Rodrigues vector
        r = npr.normal(np.zeros(3), np.array([this.particle_rSigma, this.particle_rSigma, this.particle_rSigma]))
        R, _ = cv2.Rodrigues(r)
        Rx, _ = cv2.Rodrigues(x[0:3,p])
        Rt = np.matrix(R)*np.matrix(Rx)
        rt, _ = cv2.Rodrigues(Rt)
        proposedParticle[0:3] = rt[:,0]

    else:
        movedType = MT_RND_Z_POSE
        # Instead of a random walk from the current value for all parameters, resample only the pose-dependent deformations coefficients
        # from the whole distribution
        proposedParticle = x[:, p].copy()
        nB = this.nB[i]
        zp = npr.normal(np.zeros(nB), this.body.posePCA[i]['sigma'][0:nB])
        proposedParticle = particles.set_pose_def_params(this.particleIdx[i], proposedParticle, zp)


    return proposedParticle, movedType
            
def sample_from_nbr(this, i, x, p_i):
    """
    Sample a new particle by looking at the neighbors. We first pick a neighbor, and then generate a particle from the model.
    """
     
    #pa = this.body.parent[i]
    #ch = this.body.child[i]
 
    r_min = this.body.rmin
    r_max = this.body.rmax

    ks = np.where(this.A[:,i] >=0)[0]

    nNbrs = len(ks)
    assert nNbrs > 0

    num_x = x.shape[1] 
    x_per_nbr = np.max([1, int(num_x / nNbrs)])
    
    A = xrange(x_per_nbr,num_x+1,x_per_nbr)
    try:
        I_nbr = np.min(np.where(p_i<=np.array(A))[0])
    except:
        I_nbr = 0
    k = ks[I_nbr]

    # Select the neighbor particle at random
    num_x = this.b[k]['x'].shape[1]
    I_k = np.random.randint(0,num_x,1)[0]
    x_k = this.b[k]['x'][:,I_k]

    a = k
    b = i
    proposedParticle = np.zeros(this.nodeDim[b])

    za = particles.get_pose_def_params(this.particleIdx[a], x_k)
    na = len(za)
    mu = this.body.poseDefModelA2B[a][b]['mu']
    C = this.body.poseDefModelA2B[a][b]['C']

    # Indexes of the conditioning variables
    if npr.rand()>0.5 or k != this.body.parent[b]: 
        cInd = xrange(0,na)
        X = za
        movedType = MT_NBR_Z_PARENT_COND
    else:
        l = np.prod(mu.shape)
        cInd = np.concatenate((xrange(0,na), xrange(l-3,l)))
        if k == this.body.parent[b]:
            alpha = npr.rand(3)
            r_rel = r_min[b,:] + alpha * (r_max[b,:] - r_min[b,:])
            X = np.concatenate((za, r_rel))
            movedType = MT_NBR_Z_PARENT_AND_ANGLE_COND

    nb = this.nB[b] 
    # Indexes of the resulting variables
    rInd = xrange(this.body.nPoseBasis[a], this.body.nPoseBasis[a]+nb)

    mu_ab, C_ab = ba.compute_conditioned_gaussian(this.body, rInd, cInd, mu, C, np.expand_dims(X, axis=1))
    proposedParticle = particles.set_pose_def_params(this.particleIdx[b], proposedParticle, mu_ab)

    # For the shape parameters, we propagate the same shape
    zs = particles.get_shape_params(this.particleIdx[a], x_k)
    proposedParticle = particles.set_shape_params(this.particleIdx[b], proposedParticle, zs)

    # Get the neighbor points in world frame
    Paw = particles.particle_to_points(this, x_k, a)

    # Get the points of the proposed particle
    Pb = ba.get_part_mesh(this.body, b, mu_ab, zs)

    # Compute the alignment
    R, T, cost = ba.align_to_parent(this.body, b, a, Pb, Paw, None)	

    # Add some noise to the spring
    if this.springSigma != 0:
        T = npr.normal(T, this.springSigma)

    r, _ = cv2.Rodrigues(R)
    proposedParticle = particles.set_pose_params(this.particleIdx[b], proposedParticle, r, T)

    return proposedParticle, movedType


def sample_from_nbrs(this, i, x, p_i):
    """
    Sample a new particle by looking at the neighbors. 
    Differently from above, we look at the neighbors and sample the pose and shape deformations.
    We have to consider the model poseDefModelNeighbors, that has variables [z_parent, z_part, r_rel_part, r_rel_child],
    and we want to sample the part pose deformations z_part given the relative pose w.r.t parent and child
    @params this: a dpmp object
    @params i   : the index of the node
    @params x   : the particle array 
    @params p_i : the index of the particle x in the node array b[i]['x']
    """

    pa = this.body.parent[i]
    ch = this.body.child[i]

    # Select the best neighbors particles or at random
    """
    I_pa = np.argmax(this.b[pa]['value'])
    num_x = this.b[pa]['x'].shape[1]
    I_pa = np.random.randint(0,num_x,1)[0]
    x_pa = this.b[pa]['x'][:,I_pa]
    I_ch = np.argmax(this.b[ch]['value'])
    num_x = this.b[ch]['x'].shape[1]
    I_ch = np.random.randint(0,num_x,1)[0]
    x_ch = this.b[ch]['x'][:,I_ch]
    """

    # Select the current best
    I_pa = np.argmax(this.b[pa]['value'])
    I_ch = np.argmax(this.b[ch]['value'])

    proposedParticle = x[:,p_i].copy()
    mu = this.body.poseDefModelNeighbors[i]['mu']
    C = this.body.poseDefModelNeighbors[i]['C']

    # Conditioning varibles
    # Compute the relative Rodrigues vector between the particle and the neightbors. Note that the relative Rodrigues vectors are
    # always defined between the part and its parent, so we need r_rel of i w.r.t pa and of ch w.r.t. i

    x_pa = this.b[pa]['x'][:,I_pa]
    rpa, tpa, zpa, zspa = particles.get_part_params(this.particleIdx[pa], x_pa)

    x_ch = this.b[ch]['x'][:,I_ch]
    rch, tch, zch, zsch = particles.get_part_params(this.particleIdx[ch], x_ch)

    r, t, z, zs = particles.get_part_params(this.particleIdx[i], proposedParticle)

    R0_parent, J = cv2.Rodrigues(rpa)
    R0_part, J = cv2.Rodrigues(r)
    A = np.matrix(R0_part)
    B = np.matrix(R0_parent)

    R = A*B.T*(B*B.T).I

    r_rel_pa, J = cv2.Rodrigues(R)
    R0_child, J = cv2.Rodrigues(rch)
    A = np.matrix(R0_child)
    B = np.matrix(R0_part)

    R = A*B.T*(B*B.T).I

    r_rel_ch, J = cv2.Rodrigues(R)

    X = np.concatenate((r_rel_pa, r_rel_ch))

    # Indexes of the conditioning variables
    na = len(X)
    nmu = np.prod(mu.shape)
    cInd = xrange(nmu-na,nmu)

    # Indexes of the resulting variables
    nb = this.nB[i]
    rInd = xrange(this.body.nPoseBasis[pa], this.body.nPoseBasis[pa]+nb)
     
    mu_ab, C_ab = ba.compute_conditioned_gaussian(this.body, rInd, cInd, mu, C, X)
    proposedParticle[this.particleIdx[i]['zPoseIdx']] = mu_ab

    # For the shape parameters, we propagate the same shape from the parent
    proposedParticle[this.particleIdx[i]['zPoseIdx']] = x_pa[this.particleIdx[pa]['zPoseIdx']]

    return proposedParticle


def sample_from_uniform(this, i, x, p_i):

    # TBI
    proposedParticle = x[:,p_i]
    return proposedParticle

def move_with_LM_only_pose(this, i, x):
    """
    Modify the particle x by running few iterations of the Levenberg-Marquardt algorithm
    (see Robust Registration of 2D and 3D Point Sets, A. Fitzgibbon)
    """

    proposedParticle = x
    xk = x.copy()
    r, t = particles.get_pose_params(this.particleIdx[i], xk)
    q = np.concatenate([r, t])
    qprev = q
    k = 1
    eprev = np.inf
    eprev2 = np.inf
    finito = False
    I = np.eye(len(q))
    Lambda = 0.1 

    while (k<this.LMsteps) and not finito:
        xk[this.particleIdx[i]['rIdx']] = q[0:3]
        xk[this.particleIdx[i]['OIdx']] = q[3:6]
        P = particles.particle_to_points(this, xk, i)
        P = P[0::this.resolStep,:]
        r = q[0:3]
        R, Jr = cv2.Rodrigues(r)

        out = this.kdtree.query(P)
        M = this.kdpoints[out[1],:]

        # Residuals
        e = P - M
        ex = e[:,0]
        ey = e[:,1]
        ez = e[:,2]
        Px = P[:,0]
        Py = P[:,1]
        Pz = P[:,2]

        # Jacobian
        J1 = 2*ex*(Jr[0,0]*Px + Jr[0,1]*Py + Jr[0,2]*Pz) + 2*ey*(Jr[0,3]*Px + Jr[0,4]*Py + Jr[0,5]*Pz) + 2*ez*(Jr[0,6]*Px + Jr[0,7]*Py + Jr[0,8]*Pz)
        J2 = 2*ex*(Jr[1,0]*Px + Jr[1,1]*Py + Jr[1,2]*Pz) + 2*ey*(Jr[1,3]*Px + Jr[1,4]*Py + Jr[1,5]*Pz) + 2*ez*(Jr[1,6]*Px + Jr[1,7]*Py + Jr[1,8]*Pz)
        J3 = 2*ex*(Jr[2,0]*Px + Jr[2,1]*Py + Jr[2,2]*Pz) + 2*ey*(Jr[2,3]*Px + Jr[2,4]*Py + Jr[2,5]*Pz) + 2*ez*(Jr[2,6]*Px + Jr[2,7]*Py + Jr[2,8]*Pz)

        J4 = 2*ex
        J5 = 2*ey
        J6 = 2*ez
        J = np.vstack([J1, J2, J3, J4, J5, J6])
        e = np.sum(e**2, axis=1)
        J = np.matrix(J)
        dq = - (J*J.T+Lambda*I).I*J*np.matrix(e).T
        q = np.array((q+dq.flatten()))[0]
        e = np.mean(e)
        if e > eprev:
            Lambda = Lambda*10
            q = qprev.copy()
        else:
            Lambda = Lambda/10
            eprev2 = eprev
            eprev = e
            qprev = q.copy()
        k = k+1
        finito = abs(eprev - eprev2) < 1e-6
    proposedParticle = particles.set_pose_params(this.particleIdx[i], proposedParticle, q[0:3], q[3:6])

    return proposedParticle


def move_with_LM(this, i, x):
    """
    Modify the particle x by running few iterations of the Levenberg-Marquardt algorithm
    (see Robust Registration of 2D and 3D Point Sets, A. Fitzgibbon)
    """

    proposedParticle = x
    xk = x.copy()
    r, t = particles.get_pose_params(this.particleIdx[i], xk)
    z = particles.get_pose_def_params(this.particleIdx[i], xk)
    q = np.concatenate([r, t, z])
    qprev = q
    k = 1
    eprev = np.inf
    eprev2 = np.inf
    finito = False
    I = np.eye(len(q))
    Lambda = 0.1 
    B  = this.body.posePCA[i]['B']
    nB_p = len(z)
    B = B[:,0:nB_p]
    nP = B.shape[0]/3
    Jz = [None]*nB_p

    D0 = np.zeros((nP, 3, nB_p))
    while (k<this.LMsteps) and not finito:
        xk[this.particleIdx[i]['rIdx']] = q[0:3]
        xk[this.particleIdx[i]['OIdx']] = q[3:6]
        xk[this.particleIdx[i]['zPoseIdx']] = q[6:]
        P = particles.particle_to_points(this, xk, i)
        P = P[0::this.resolStep,:]
        r = q[0:3]
        R, Jr = cv2.Rodrigues(r)

        out = this.kdtree.query(P)
        M = this.kdpoints[out[1],:]

        # Residuals
        e = P - M
        ex = e[:,0]
        ey = e[:,1]
        ez = e[:,2]
        Px = P[:,0]
        Py = P[:,1]
        Pz = P[:,2]

        # Jacobian
        J4 = 2*ex
        J5 = 2*ey
        J6 = 2*ez
        J1 = J4*(Jr[0,0]*Px + Jr[0,1]*Py + Jr[0,2]*Pz) + J5*(Jr[0,3]*Px + Jr[0,4]*Py + Jr[0,5]*Pz) + J6*(Jr[0,6]*Px + Jr[0,7]*Py + Jr[0,8]*Pz)
        J2 = J4*(Jr[1,0]*Px + Jr[1,1]*Py + Jr[1,2]*Pz) + J5*(Jr[1,3]*Px + Jr[1,4]*Py + Jr[1,5]*Pz) + J6*(Jr[1,6]*Px + Jr[1,7]*Py + Jr[1,8]*Pz)
        J3 = J4*(Jr[2,0]*Px + Jr[2,1]*Py + Jr[2,2]*Pz) + J5*(Jr[2,3]*Px + Jr[2,4]*Py + Jr[2,5]*Pz) + J6*(Jr[2,6]*Px + Jr[2,7]*Py + Jr[2,8]*Pz)

        # Partial derivatives of the residuals with respect to the PCA coefficients
        A = B*z
        D0[:,0,:] = A[0::3,:]
        D0[:,1,:] = A[1::3,:]
        D0[:,2,:] = A[2::3,:]

        D = D0[0::this.resolStep,:,:]
    
        for l in xrange(nB_p): 
            D[:,:,l] = np.squeeze(D[:,:,l])*np.matrix(R)
            Jz[l] = J4*D[:,0,l]+J5*D[:,1,l]+J6*D[:,2,l];

        J = np.vstack([J1, J2, J3, J4, J5, J6, Jz])
        e = np.sum(e**2, axis=1)
        J = np.matrix(J)
        dq = - (J*J.T+Lambda*I).I*J*np.matrix(e).T
        q = np.array((q+dq.flatten()))[0]
        e = np.mean(e)
        if e > eprev:
            Lambda = Lambda*10
            q = qprev.copy()
        else:
            Lambda = Lambda/10
            eprev2 = eprev
            eprev = e
            qprev = q.copy()
        k = k+1
        finito = abs(eprev - eprev2) < 1e-6

    proposedParticle = particles.set_pose_params(this.particleIdx[i], proposedParticle, q[0:3], q[3:6])

    return proposedParticle
