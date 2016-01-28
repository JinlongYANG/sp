#!/usr/bin/env python
# encoding: utf-8
"""
run_dpmp_step.py  V1.0

Created by Silvia Zuffi.
Code for the dpmp algorithm is translated from Matlab code written with Jason Pacheco, Brown University.
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
import random
import update_all_messages
import initBP
import sample_particles
import particles
import set_me_as_solution
import time
from update_all_messages import compute_model_log_posterior

def run_DPMP_step(dpmp, s, frameId, lastResult = None):
    """
    Run a step of diverse particle max product. 
    @params: dpmp Is an object of class Dpmp
    @params: s Is the step index
    """

    if dpmp.verbose >= 1:
        print 'PBP Iter ' + str(s)

    if s == 0 and frameId == 0:
        # Only do the initialization and run BP
        initBP.init(dpmp)
        update_all_messages.update(dpmp)

    elif s == 0:
	initBP.init_based_on_last_frame(dpmp, lastResult)
	update_all_messages.update(dpmp)

    else:
        # In each step we perform augment + BP + select + BP

        # Save the current number of particles
        nParticles_old = dpmp.nParticles.copy()

        # Define the augmentation strategy. Can be chosen per step, or a different strategy per particle (in this case
        # we do the selection later)
        if dpmp.proposal_per_particle:
            augment_type = None
        else:
            randVal = random.random()
            if randVal <= (1.0 - dpmp.probRandomWalk - dpmp.probUniform):
                augment_type = 'NBR'
            else:
                if randVal <= (1.0 - dpmp.probUniform):
                    augment_type = 'RANDWALK'
                else:
                    augment_type = 'UNIFORM'

        # Sample new particles (the are saved in dpmp.new_b)
        tic = time.time()
        augmentTypes = sample_particles.sample_particles(dpmp, augment_type)
        toc = time.time() 
        if dpmp.verbose > 0:
            print 'time to sample new particles: ' + str(toc-tic)

        # Augment the partice set on each node (dpmp.b)           
        tic = time.time()
        for v in dpmp.nodeIdx:
            dpmp.b[v]['x'] = np.hstack((dpmp.b[v]['x'].copy(), dpmp.new_b[v]['x'].copy()))
            dpmp.nParticles[v] = dpmp.b[v]['x'].shape[1]
            new_L = particles.compute_likelihood(dpmp, v, dpmp.new_b[v]['x'])
            dpmp.b[v]['L'] = np.vstack((dpmp.b[v]['L'].copy(), new_L.copy())).flatten()
            dpmp.b[v]['value'] = np.zeros((dpmp.nParticles[v]))
        toc = time.time()
        if dpmp.verbose > 0:
            print 'time for likelihood of augmented particles: ' + str(toc - tic)
        
        # Update the messages on the extended sets of particles
        update_all_messages.update(dpmp)

        # Select: we select into new_b the chosen particles from the extended set in b
        dpmp.nParticles = nParticles_old.copy()

        if dpmp.select_msg:
            I_accept = select_msg_faster(dpmp)
        else:
            if dpmp.verbose > 0:
                print 'select m-best!'
            I_accept = select_mbest(dpmp)

        if dpmp.verbose > 2:
            print 'Acceptance rates:'
            for v in dpmp.nodeIdx:
                print dpmp.body.names[v] + ': ' + str(np.sum(I_accept[v] >= dpmp.nParticles[v])/(2.0*dpmp.nParticles[v]))

        for v in dpmp.nodeIdx:
            dpmp.b[v]['value'] = dpmp.new_b[v]['value'].copy()
            dpmp.b[v]['L'] = dpmp.new_b[v]['L'].copy()
            dpmp.b[v]['x'] = dpmp.new_b[v]['x'].copy()

        update_all_messages.update(dpmp)


    logB = set_me_as_solution.fun(dpmp, 0)
    if dpmp.verbose > 1:
        print 'negative energy ' + str(logB)
    return logB


def select_msg_faster(dpmp):
    """
    Selection strategy described in the ICML paper J.Pacheco, S.Zuffi, M.Black, E.Sudderth, "Preserving Modes and Messages via Diverse Particle Selection", ICML 2014.
    """
    I_accept = [None]*dpmp.nNodes
    # Add always the MAP particle
    if dpmp.keep_MAP_in_selection:
        I_accept_best = get_best(dpmp)

    for partId in dpmp.nodeIdx:
        M = dpmp.nParticles[partId]
        logM = []
        logPsi = []
        for k in dpmp.neighbors[partId]:
            if k in dpmp.nodeIdx:
                thisM = dpmp.m[dpmp.Q[partId,k]]
                logM.append(thisM['value'])
                logPsi.append(thisM['logPsi'])
            else:
                continue
        logPsi = np.column_stack(logPsi)
        logM = np.concatenate(logM)
        
        logZ = logPsi.max()
        M_out = np.exp(1.0/dpmp.Tmsg*logM - 1.0/dpmp.Tmsg*logZ)
        Psi = np.exp(1.0/dpmp.Tmsg*logPsi - 1.0/dpmp.Tmsg*logZ)
        N_i = Psi.shape[0]

        # Select particles
        M_hat = []
        I_accept[partId] = np.zeros((M), dtype=np.int)
        start = 0
        if dpmp.keep_MAP_in_selection:
	    # The first particle is the best particle
            I_accept[partId][0] = I_accept_best[partId]
            start = 1

        for m in range(start,M):
            if m == start:
                delta = M_out - Psi
                objval = np.max(delta, axis=1)
                b = np.argmin(objval)
            else:
                delta = M_out - M_hat
                a_star = np.argmax(delta)
                # Get unused particles
                b_used = I_accept[partId][0:m]
                b_unused = range(N_i)
                b_unused = np.setdiff1d(b_unused, b_used) 

                # Select next particle
                idx_max = np.argmax(Psi[b_unused,a_star], axis=0)
                b = b_unused[idx_max]

            I_accept[partId][m] = b

            # Update message approx
            if M_hat == []:
                M_hat = Psi[b,:]
            else:
                B = np.vstack((M_hat, Psi[b,:]))
                M_hat = np.max(B, axis=0)

        # Save accepted particles
        new_x = dpmp.b[partId]['x'][:,I_accept[partId]].copy()

        # New belief struct
        dpmp.new_b[partId]['x'] = new_x.copy()
        dpmp.new_b[partId]['value'] = np.zeros((M))
        dpmp.new_b[partId]['L'] = dpmp.b[partId]['L'][I_accept[partId]].copy()

        assert len(dpmp.new_b[partId]['L']) == M
        assert new_x.shape[1] == M

    return I_accept


def get_best(dpmp):
    I_accept = [None]*dpmp.nNodes
    Idx = [None]*dpmp.nNodes
    map = -1*np.ones((dpmp.nNodes, dpmp.nNodes), dtype=np.int)

    for partId in dpmp.nodeIdx:
        M = dpmp.nParticles[partId]
        I_sort = np.argsort(-1*dpmp.b[partId]['value'], axis=None)
        idx = I_sort[0].copy()
        I_accept[partId] = idx

    return I_accept

def select_mbest(dpmp):
    """
    This is the simplest selection strategy, it just selects the top particles, there is no diversity enforcement.
    """

    I_accept = [None]*dpmp.nNodes
    Idx = [None]*dpmp.nNodes
    map = -1*np.ones((dpmp.nNodes, dpmp.nNodes), dtype=np.int)

    for partId in dpmp.nodeIdx:
        M = dpmp.nParticles[partId]
        I_sort = np.argsort(-1*dpmp.b[partId]['value'], axis=None)
        idx = I_sort[0:M].copy()
        dpmp.new_b[partId]['x'] = dpmp.b[partId]['x'][:,idx].copy()
        dpmp.new_b[partId]['L'] = dpmp.b[partId]['L'][idx].copy()
        dpmp.new_b[partId]['value'] = np.zeros((M))
        I_accept[partId] = idx

    return I_accept


