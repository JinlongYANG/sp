#!/usr/bin/env python
# encoding: utf-8
"""
jonas_maxmin.py  V1.0

Copyright (c) 2015 MPI. All rights reserved.

Max-Planck grants you a non-exclusive, non-transferable, free of charge right to use the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited.

The Software may not be reproduced, modified and/or made available in any form to any third party without Max-Planck’s prior written permission.  By downloading the Software, you agree not to reverse engineer it.

* Disclaimer of Representations and Warranties

You expressly acknowledge and agree that the Software results from basic research, is provided “AS IS”, may contain errors, and that any use of the Software is at your sole risk. MAX-PLANCK MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE SOFTWARE, NEITHER EXPRESS NOR IMPLIED, AND THE ABSENCE OF ANY LEGAL OR ACTUAL DEFECTS, WHETHER DISCOVERABLE OR NOT. Specifically, and not to limit the foregoing, Max-Planck makes no representations or warranties (i) regarding the merchantability or fitness for a particular purpose of the Software, (ii) that the use of the Software will not infringe any patents, copyrights or other intellectual property rights of a third party, and  (iii) that the use of the Software will not cause any damage of any kind to you or a third party.

See LICENCE.txt for licensing and contact information.
"""


import numpy as np

def jonas_maxmin(n_betas):
   # minimium and maximum
   pi = np.pi
   minarray = np.array([-1,-1,-1,                  #  0 translate
                        -pi/4.0, -pi/4.0, -pi/4.0, #  3 pelvis          bts
                        -1.5, -1.0, -1.0,          #  6 midsection      bts
                        -2.0, -pi/2.0, -0.25,      #  9 left thigh      bts
                        -2.0, -pi/2.0, -1.0,       # 12 right thigh     bts
                        -0.25, -0.25, -0.25,       # 15 left torso      bts?
                        -0.25, -0.25, -0.25,       # 18 right torso     bts?
                        -1.0, -pi/2.0, -1.0,       # 21 head            bts
                        0, -0.5, -0.25,            # 24 left calf       bts
                        -1.0, -pi/2.0, -0.5,       # 27 left foot       bts
                        0, -0.5, -0.25,            # 30 right calf      bts
                        -1.0, -pi/2.0, -0.5,       # 33 right foot      bts
                        -0.1, -0.1, -pi/4.0,       # 36 left shoulder   tsb
                        -pi/2.0, -0.75*pi, -1.0,   # 39 left upper arm  tsb
                        -0.1, -2.2, -0.1,          # 42 left lower arm  tsb
                        -pi/2.0, -1.0, -pi/2.0,    # 45 left hand       tsb
                        -0.1, -0.1, -pi/4.0,       # 48 right shoulder  tsb
                        -pi/2.0, -pi/2.0, -1.5,    # 51 right upper arm tsb
                        -0.1, 0, -0.1,             # 54 right lower arm tsb
                        -pi/2.0, -1.0, -pi/2.0])   # 57 right hand      tsb

   minarray = np.hstack((minarray,-2.5*np.ones((n_betas))))
   maxarray = -minarray
   # Adjust only those values that are asymetric.
   maxarray[6] = 2.0
   maxarray[9] = pi/2.0
   maxarray[11] = 1.0
   maxarray[12] = pi/2.0
   maxarray[14] = 0.25
   maxarray[24] = 2.5
   maxarray[30] = 2.5
   maxarray[40] = pi/2.0
   maxarray[41] = 1.5
   maxarray[43] = 0
   maxarray[52] = 0.75*pi
   maxarray[53] = 1.0
   maxarray[55] = 2.2

   # forearms are very limited
   #minarray[range(42,45) + range(54,57)] -= 1.0
   #maxarray[range(42,45) + range(54,57)] += 1.0
   ## calfs sometimes are limited because of wrong thigh orientation
   #minarray[range(24,36)] -= 1.0
   #maxarray[range(24,36)] += 1.0
   return np.vstack((minarray,maxarray))
