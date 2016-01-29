#!/usr/bin/env python
# encoding: utf-8
"""
sbm.py  V1.0

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
import pickle as pkl
import scipy
from scipy import io
import cv2
import sbm_basic as ba
import get_sample_from_model
import jonas_maxmin as jo
import pdb
import os

class Sbm(object):
    """Stitched Puppet or Stitched Body Model"""

    partPoints = []
    part2bodyPoints = []
    body2partPoints = []
    partFaces = []
    interfaceFacesFromTo = []
    interfacePointsFromTo = []
    interfacePointsFlex = []

    posePCA = []
    poseDefModel = []
    poseDefModelNeighbors = []
    poseDefModelA2B = []

    def __init__(self, baseDir):
        # The parts should be orderd from root to leaves

        self.partSet = [3,18,7,8,9,16,11,13,4,10,15,0,1,2,5,12]
        self.pairs = [[3,18],[3,7],[3,8],[3,9],[3,16],[9,11],[16,13],[11,15],[13,0],[15,1],[0,2],[7,4],[8,10],[10,5],[4,12],[18,3],[7,3],[8,3],[9,3],[16,3],[11,9],[13,16],[15,11],[0,13],[1,15],[2,0],[4,7],[10,8],[5,10],[12,4]]
        self.parent = [13,15,0,-1,7,10,-1,3,3,3,8,9,4,16,-1,11,3,-1,3]
        self.child = [2,-1,-1,-1,12,-1,-1,4,10,11,5,15,-1,0,-1,1,13,-1,-1]

        self.nParts = np.max(self.partSet)+1
        self.parts = {'la_r':0, 'hand_l':1, 'hand_r':2, 'torso':3, 'll_l':4, 'foot_r':5, 'ul_l':7, 'ul_r':8, 'should_l':9, 'll_r':10, 'ua_l':11, 'foot_l':12, 'ua_r':13, 'la_l':15, 'should_r':16, 'head':18}
        self.names = ['la_r', 'hand_l', 'hand_r', 'torso', 'll_l', 'foot_r', '', 'ul_l', 'ul_r', 'should_l', 'll_r', 'ua_l', 'foot_l', 'ua_r', '', 'la_l', 'should_r', '', 'head']

        self.gender = 'female'
        self.template_filename = 'mptem_ho'
        self.pose_filename = 'scmp_ho'

        self.data_dir = baseDir + '/training/'
        self.model_dir = baseDir + '/models/'

        self.nCMU = 600
        self.nPairs = len(self.pairs)
        self.nMaxPoseBasis = 30
        self.nMaxShapeBasis = 30
        self.nPoseBasis = np.zeros((self.nParts), dtype=np.int)
        self.nShapeBasis = np.zeros((self.nParts), dtype=np.int)
        self.Zp = [None]*self.nParts
        self.Zs = [None]*self.nParts
        self.fixedShape = False

        a = jo.jonas_maxmin(0)
        self.rmin = np.reshape(a[0][3:], (19,3))
        self.rmax = np.reshape(a[1][3:], (19,3))

        self.template_r_abs = 0
        self.template_t = 0
        self.template_skel = 0

        self.colors = ['turquoise3' , 'LightSalmon', 'seashell3', 'LavenderBlush3', 'wheat3',\
        'DarkOliveGreen3', 'OrangeRed3', 'maroon3', 'chartreuse3', 'coral3' ,'thistle3', 'gold3', 'LightPink3',\
        'orchid3', 'PaleTurquoise3', 'aquamarine3','RoyalBlue3', 'gold3', 'LightSkyBlue3', \
        'SpringGreen3', 'sienna3', 'GhostWhite', 'thistle3', 'RoyalBlue3',  \
        'LavenderBlush3', 'gold3' ]


    def abs_to_rel(self, r_abs):
        r_rel = r_abs.copy()
        for part in self.partSet:
            parent = self.parent[part]
            if parent >= 0:
                R0_parent, J = cv2.Rodrigues(r_abs[parent])
                R0_part, J = cv2.Rodrigues(r_abs[part])
                #R = R0part/R0parent
                A = np.matrix(R0_part)
                B = np.matrix(R0_parent)
                R = A*B.T*(B*B.T).I
                r, J = cv2.Rodrigues(R)
                r_rel[part] = r.T
        return r_rel


    def rel_to_abs(self, r_rel):
        r_abs = r_rel.copy()
        for part in self.partSet:
            parent = self.parent[part]
            if parent >= 0:
                R0_parent, J = cv2.Rodrigues(r_abs[parent])
                R0_part, J = cv2.Rodrigues(r_rel[part])
                A = np.matrix(R0_part)
                B = np.matrix(R0_parent)
                R = A*B
                r, J = cv2.Rodrigues(R)
                r_abs[part] = r.T
        return r_abs
    
    def load_sample(self, filename):
        print 'loading ' + filename
	with open(self.data_dir + filename) as f:
            data = pkl.load(f)
            f.close()
	return data

    def load_model(self, name):
        filename = name + ".pkl"
        load_file = os.path.join(self.model_dir, filename)
        print 'Loading %s ' % load_file
        with open(load_file) as f:
            data = pkl.load(f)

        self.partFaces = data['partFaces']
        self.part2bodyPoints = data['part2bodyPoints']
        self.body2partPoints = data['body2partPoints']
        self.partPoints = data['partPoints']
        self.posePCA = data['posePCA']
        self.poseDefModelNeighbors = data['poseDefModelNeighbors']
        self.poseDefModelA2B = data['poseDefModelA2B']
        self.shapePCA = data['shapePCA']
        self.interfacePointsFromTo = data['interfacePointsFromTo']
        self.interfacePointsFlex = data['interfacePointsFlex']
        self.interfaceBones = data['interfaceBones']
        self.poseDefModelTorsoNoShoulders = data['poseDefModelTorsoNoShoulders'] 
        for i in self.partSet: 
            self.nPoseBasis[i] = len(self.posePCA[i]['sigma'])
            self.nShapeBasis[i] = len(self.shapePCA[i]['sigma'])
            self.Zp[i] = np.zeros((self.nPoseBasis[i]))
            self.Zs[i] = np.zeros((self.nShapeBasis[i]))
        self.t = np.zeros((self.nParts,3))
        self.r_abs = np.zeros((self.nParts,3))

        return

    def show_mesh(self, P):
        ba.show_mesh(self, P)
        return

    def get_part_mesh(self, i, Zp, Zs):
        P = ba.get_part_mesh(self, i, Zp, Zs)
        return P


