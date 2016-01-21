#!/usr/bin/env python
# encoding: utf-8
"""
dpmp_3D.py  V1.0

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
from run_dpmp_step import run_DPMP_step
import sbm
import load_mesh
import sbm_basic as ba
import cv2
import scipy
import scipy.ndimage
from update_all_messages import compute_model_log_posterior
import time
import pickle as pkl
import particles
import initBP
try:
    from body.mesh.meshviewer import MeshViewer
    from body.mesh.mesh import Mesh
except ImportError:
    print 'visualization not supported'

ADAPTIVE_WEIGHTS = True

import dpmp 

def show_all_particles(this, faustId, nParticles, s):
    """
    @params: this, a dpmp object
    @params: faustId, ID of the faust scan, only used for the filename of the picture
    @params: nParticles, how many particles to show
    @params: step, only used for the filename of the picture
    """
    ms = [Mesh(v=this.scanMesh.v, f=this.scanMesh.f).set_vertex_colors('firebrick')]
    for i in range(0, nParticles):
        for part in this.body.partSet:
            P = particles.particle_to_points(this, this.b[part]['x'][:,i], part)
            ms.append(Mesh(v=P, f=this.body.partFaces[part]).set_vertex_colors(this.body.colors[part]))

    mv = MeshViewer()
    #mv.set_background_color(np.array([1.0, 1.0, 1.0]))
    mv.set_static_meshes(ms)
    time.sleep(4)
    mv.save_snapshot('particles_'+faustId+'_'+str(s)+'.png', blocking=True)

def run(nParticles, nSteps, faustId, isTest, params, code, seed):

    np.random.seed(seed)
    nBtorso = 12 
    nB =  5 
    nBshape = 4 

    gender = 'male'
    sbmModel = "model_ho_male_5_reduced"
    femaleIdxs = range(20,40)+ range(80,140)+ range(160,180)
    if isTest and (int(faustId) in femaleIdxs):
        gender = 'female'
        sbmModel = "model_ho_female_5_reduced"

    b = sbm.Sbm()
    b.gender = gender
    d = dpmp.Dpmp(b, nBtorso, nB, nBshape, nParticles, 0, sbmModel)

    d.body.fixedShape = False
    d.compute_normals_cost = True
    d.select_msg = True
    d.probRandomWalk = 0.5
    d.use_map_particle_for_rnd_walk = False
    d.LMsteps = 5 # Actually is 4
    d.init_torso_location = np.zeros((3))
    d.init_torso_rotation = np.zeros((3))
    d.init_with_global_rotation = True
    d.springSigma = 0 
    d.display = -1
    d.verbose = 1
    d.likelihoodType = '3Dlikelihood' 
    if d.verbose > 0:
        print 'MODEL ' + sbmModel
        print 'GENDER ' + d.body.gender

    # Inference parameters
    d.particle_genericSigma = params['genericSigma']; d.particle_rSigma = params['rSigma']; d.particle_tSigma = params['tSigma'] 
    d.particle_posePCAsigmaScale = params['posePCAsigmaScale']; d.particle_shapePCAsigmaScale = params['shapePCAsigmaScale'] 
    d.robustOffset = params['robustOffset']; d.robustGamma = params['robustGamma']; l_alphaNormal = params['l_alphaNormal']
    l_alphaLoose = params['l_alphaLoose']; l_alphaVeryLoose = params['l_alphaVeryLoose']; s_alphaNormal = params['s_alphaNormal']
    s_alphaLoose = params['s_alphaLoose']; s_alphaTight = params['s_alphaTight']; alphaRef = params['alphaRef']

    # When to change parameters during inference
    if ADAPTIVE_WEIGHTS:
        fullModelStart = nSteps/4
        refinementStart = 2*nSteps/4
        greedyStart = 3*nSteps/4
    else:
        fullModelStart = 1
        refinementStart = nSteps+1
        greedyStart = nSteps+1

    # Load one example to use as test data
    load_mesh.load_FAUST_scan(d, faustId, isTest)

    # Inference
    lower_parts = np.array([2, 0, 5, 10, 12, 4, 1, 15])
    logB = np.zeros((nSteps))

    if ADAPTIVE_WEIGHTS:
        d.likelihoodAlpha[:] =  l_alphaNormal 
        d.likelihoodAlpha[lower_parts] =  l_alphaVeryLoose 
        d.stitchAlpha = s_alphaNormal*np.ones((d.nNodes, d.nNodes))
        d.stitchAlpha[d.body.parts['ll_r'], d.body.parts['foot_r']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['foot_r'], d.body.parts['ll_r']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['ll_l'], d.body.parts['foot_l']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['foot_l'], d.body.parts['ll_l']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['la_r'], d.body.parts['hand_r']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['hand_r'], d.body.parts['la_r']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['la_l'], d.body.parts['hand_l']] = s_alphaLoose
        d.stitchAlpha[d.body.parts['hand_l'], d.body.parts['la_l']] = s_alphaLoose

        d.stitchAlpha[d.body.parts['ul_r'], d.body.parts['torso']] = s_alphaTight
        d.stitchAlpha[d.body.parts['torso'], d.body.parts['ul_r']] = s_alphaTight
        d.stitchAlpha[d.body.parts['ul_l'], d.body.parts['torso']] = s_alphaTight
        d.stitchAlpha[d.body.parts['torso'], d.body.parts['ul_l']] = s_alphaTight
    else:
        d.likelihoodAlpha[:] =  l_alphaNormal 
        d.stitchAlpha = s_alphaNormal*np.ones((d.nNodes, d.nNodes))


    d.nSteps = nSteps 
    for s in range(nSteps):
        d.step = s
        if ADAPTIVE_WEIGHTS:
            if s == fullModelStart:
                d.likelihoodAlpha[lower_parts] = l_alphaNormal
                d.likelihoodAlpha[d.body.parts['hand_r']] = l_alphaLoose
                d.likelihoodAlpha[d.body.parts['hand_l']] = l_alphaLoose
                d.likelihoodAlpha[d.body.parts['foot_r']] = l_alphaLoose
                d.likelihoodAlpha[d.body.parts['foot_l']] = l_alphaLoose
                d.stitchAlpha = s_alphaNormal*np.ones((d.nNodes, d.nNodes))
                d.stitchAlpha[d.body.parts['ul_r'], d.body.parts['torso']] = s_alphaTight
                d.stitchAlpha[d.body.parts['torso'], d.body.parts['ul_r']] = s_alphaTight
                d.stitchAlpha[d.body.parts['ul_l'], d.body.parts['torso']] = s_alphaTight
                d.stitchAlpha[d.body.parts['torso'], d.body.parts['ul_l']] = s_alphaTight
                # Recompute the likelihood of the particles as I have changed the weight
                d.compute_normals_cost = True
                for v in d.nodeIdx:
                    new_L = particles.compute_likelihood(d, v, d.b[v]['x'])
                    d.b[v]['L'] = new_L.copy()

            # Refinement
            if s == refinementStart:
                d.particle_genericSigma = alphaRef*d.particle_genericSigma
                d.particle_rSigma = alphaRef*d.particle_rSigma
                d.particle_tSigma = alphaRef*d.particle_tSigma
                d.particle_posePCAsigmaScale = alphaRef*d.particle_posePCAsigmaScale
                d.particle_shapePCAsigmaScale = alphaRef*d.particle_shapePCAsigmaScale
                d.stitchAlpha = s_alphaTight*np.ones((d.nNodes, d.nNodes))

            # Greedy resampling around the best solution
            if s == greedyStart:
                d.select_msg = False # Use m-best instead
                d.probRandomWalk = 1.0
                d.use_map_particle_for_rnd_walk = True

            tic = time.time()
            logB[s] = run_DPMP_step(d, s)
            toc = time.time() - tic

            if d.display ==4: 
                show_all_particles(d, faustId, nParticles, s)

            if d.verbose > 0:
                print str(s) + ' time to run DPMP step: ' + str(toc)
                logPos, logL, logP = compute_model_log_posterior(d)
                print 'iter ' + str(s) + ' logPos= ' + str(logPos) + ' logL= ' + str(logL) + ' logP= ' + str(logP) 
            print str(s) + ': ' + str(logB[s])
            filename = 'faustID_' + faustId + '_' + str(seed) + '_' + str(s) + '.png'
            if d.display > 0:
                ba.show_me(d.body, scan=d.scanMesh, filename='dpmp_step_'+faustId + '_' +str(s)+'.png')


    # Show the solution
    if d.display > 0:
        filename = code + 'faustID_' + faustId + '_' + str(seed) + '.png'
        ba.show_me(d.body, dbstop=False, scan=d.scanMesh, filename=filename)

    if d.verbose > 0:
        print 'negative energy at each iteration:'
        print logB

    # Save the result as a single mesh
    v, f = ba.sbm_to_scape_mesh(d.body)
    mesh_data = {'v':v, 'f':f} 

    filename = '../results/' + code + 'faustID_' + faustId + '_' + str(seed) + '.pkl'
    dpmp.save_dpmp(d, logB, params, mesh_data, filename)
    dpmp.show_result(filename)

    # Save in ply
    from my_mesh.mesh import myMesh
    m = myMesh(v=v, f=f)
    filename = '../results/' + code + 'faustID_' + faustId + '_' + str(seed) + '.ply'
    m.save_ply(filename)



if __name__ == '__main__':

    nParticles = 30
    nSteps = 60
    isTest = True

    params = {'genericSigma': 0.001, 'rSigma':0.3, 'tSigma':0.1, 'posePCAsigmaScale':0.1, 'shapePCAsigmaScale':0.5, 'robustOffset':0.001, 'robustGamma':0.45, 
            'l_alphaNormal': 1.0, 'l_alphaLoose': 0.1, 'l_alphaVeryLoose': 0.001, 's_alphaNormal': 0.25, 's_alphaLoose':0.001, 's_alphaTight':0.5, 'alphaRef':0.1}

    code = '0_'
    faustId = '033' 

    tic = time.time()
    run(nParticles, nSteps, faustId, isTest, params, code, 0)
    toc = time.time()
    print 'Execution time: ' + str(toc-tic)

