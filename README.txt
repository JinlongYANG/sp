
This Software implements the model and methods described in the paper:
Silvia Zuffi, Michael J. Black, The Stitched Puppet: A Graphical Model of 3D Human Shape and Pose, CVPR 2015.

REQUIREMENTS:

The Software is written in Python and has been developed with version 2.7.
It requires the packages: numpy, scipy and opencv.

It also requires methods to read and write ply files and for mesh visualization.

In the original implementation of the Software, we have used a proprietary library for read/write, process and visualize 3D meshes. 
In order to distribute our work, we have therefore considered alternative solutions.  
In particular to read/write 3D meshes the Software uses the python-plyfile library (https://github.com/dranjan/python-plyfile) that you should download and install.
The Software does not support interactive visualization, but it saves the solution in ply format that can be visualized with meshlab (http://meshlab.sourceforge.net/)

The Software performs alignment of a Stitched Puppet (SP) model to a scan from the FAUST dataset (http://faust.is.tue.mpg.de/).
We provide the Software with one of the FAUST scans, in order to run a demo.

NOTE:
1) Unfortunately the figures in the paper have been generated with code that we cannot distribute, but you can visualize the solution with meshlab. 
2) We do not provide the code to compute the mesh correspondences for the FAUST challenges. This is because we used code that we cannot distribute. The code addresses the difference in topology between the male and female models.
3) In the 'models' directory we provide two sets of models. The models "model_ho_female_reduced_cvpr.pkl" and "model_ho_male_reduced_cvpr.pkl" are the ones used for the CVPR paper. The models "model_ho_female_5_reduced.pkl" and "model_ho_male_5_reduced.pkl" are new models that we learned after the submission. We realized that the models used for the CVPR paper have residual translations in the intrinsic shape deformation basis components, that we removed in the more recent models. 

USE:

Download and unzip the Software. 
From the root directory type:
cd code
python dpmp_3D.py

Ths runs the example of Figure 7 in the CVPR paper.
At the end you should obtain solutions files in ../results. You can open the file 0_faustID_033_0.ply with meshlab.
The scan is aligned to the global frame before inference, so the SP solution and the original scan have a relative translation.

Please send any questions to 
Silvia Zuffi: silvia.zuffi@tue.mpg.de






