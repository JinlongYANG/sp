#!/usr/bin/env python
# encoding: utf-8
"""
load_mesh.py  V1.0

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
try:
    from body.mesh.mesh import Mesh
except ImportError:
    print 'visualization not supported'

from my_mesh.mesh import myMesh

import cPickle

def load_FAUST_scan(dpmp, id, isTest):
    """
    Load a scan from the Faust dataset
    """

    if isTest:
        FaustDir = '../MPI-FAUST/test/scans/'
        filename = FaustDir + 'test_scan_' + id + '.ply'
    else:
        FaustDir = '../MPI-FAUST/training/scans/'
        filename = FaustDir + 'tr_scan_' + id + '.ply'

    print 'loading ' + filename    
    mym = myMesh(filename=filename)

    points = mym.v

    # Center data
    dpmp.scanCenter =  np.mean(points, axis=0)
    points = points - dpmp.scanCenter

    # Build kdtree for likelihood computation
    from scipy.spatial import cKDTree
    dpmp.kdtree = cKDTree(points)
    dpmp.kdpoints = points

    # Store mesh and normals
    mym.v = points
    dpmp.scanMesh = mym 
    dpmp.scanMeshNormals = mym.estimate_vertex_normals()



