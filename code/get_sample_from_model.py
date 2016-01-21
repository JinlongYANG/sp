#!/usr/bin/env python
# encoding: utf-8
"""
get_sample_from_model.py  V1.0

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
import cv2  
import sbm_basic as  ba
import time

def get_sample(this, nB_torso, nB_pose, nB_shape, fixedShape=True, add_global_rotation=True, init_torso_rotation=np.zeros((3))):
    """
    Returns the set of points of a model sample obtained with the specified number of basis components.
    The number of basis components can be a vector of dimension number of parts, or a single value for each part.
    Also returns the model parameters.
    @params: nB_torso, integer, the number of PCA components for the pose-dependent deformation model for the torso 
    @params: nB, integer or array, the number of PCA components for the pose-dependent deformation model for the parts 
    @params: fixedShape, boolean, if to use the mean intrinsic shape
    @params: add_global_rotation, boolean, if to sample a random rotation around the vertical axis for the torso
    @params: init_torso_rotation, 3-dim array, is the orientation of the sample
    """

    torso = this.parts['torso']
    Zp = [None] * this.nParts
    # If the number of PCA components for the pose-dependent model is not specified, we consider a fixed pose
    # (this is only used to visualize the model without deformations)
    if nB_torso is not None and nB_pose is not None:
        if np.size(nB_pose) == 1:
            nB_p = nB_pose*np.ones((this.nParts), dtype=np.int)
        else:
            nB_p = nB_pose
        nB_p[torso] = nB_torso
        fixedPose = False
    else:
        fixedPose = True

    if fixedPose:
        for b in this.partSet:
            Zp[b] = np.zeros((this.nPoseBasis[b]))
    else:
        for b in this.partSet:
            if b == this.parts['torso']:
                idx = np.arange(0, nB_p[torso])
                Zp[b] = npr.normal(np.zeros((nB_p[b])), this.posePCA[torso]['sigma'][idx])
            else:
                a = this.parent[b]
                x = Zp[a]
                cInd = np.arange(0,nB_p[a])
                rInd = np.arange(this.nPoseBasis[a],this.nPoseBasis[a]+nB_p[b])
                mu = this.poseDefModelA2B[a][b]['mu']
                C = this.poseDefModelA2B[a][b]['C']
                mu_ab, C_ab = ba.compute_conditioned_gaussian(this, rInd, cInd, mu, C, x)
                Zp[b] = mu_ab
    if fixedShape:
        zs = np.zeros((this.nShapeBasis[b]))
    else:
        zs = npr.normal(this.shapePCA[torso]['M'][0:nB_shape], this.shapePCA[torso]['sigma'][0:nB_shape])

    # Generate the mesh
    Pw  = [None]*this.nParts
    r_abs = np.zeros((this.nParts, 3))

    t = np.zeros((this.nParts, 3))
    for part in this.partSet:

        parent = this.parent[part]
        P = ba.get_part_mesh(this, part, Zp[part], zs)
        if parent >= 0:
            R = None
            R, T, cost = ba.align_to_parent(this, part, parent, P, Pw[parent], R)
        else:
            # Add a global rotation to the torso
            b = this.parts['torso']

            if add_global_rotation:
                alpha = npr.rand(1)
                r_abs[b,1] = -np.pi + alpha * 2.0*np.pi
            else:
                r_abs[b,:] = 0.0

            R1, _ = cv2.Rodrigues(r_abs[b,:])
            R2, _ = cv2.Rodrigues(init_torso_rotation)
            R = np.matrix(R2)*np.matrix(R1)
            r, _ = cv2.Rodrigues(R)
            r_abs[b,:] = r[:,0]
            T = np.zeros((3))
        Pw[part] = ba.object_to_world(P, R, T)
        r, _ = cv2.Rodrigues(R)
        r_abs[part, :] = r.flatten()
        t[part,:] = T.flatten()

    return Pw, r_abs, t, Zp, zs


