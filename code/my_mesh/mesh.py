
__all__ = ["myMesh"]
from plyfile import PlyData, PlyElement
import numpy as np
import pdb

class myMesh(object):
    """
    Attributes:
        v: Vx3 array of vertices
        f: Fx3 array of faces
    """
    def __init__(self, v=None, f=None, e=None, filename=None, rotationFlag = None):
        self.v = None
        self.f = None

        if v is not None:
            self.v = np.array(v, dtype=np.float64)
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

	if e is not None:
	    self.e = np.require(e, dtype=np.uint32)

        if filename is not None:
            self.load_from_ply(filename, rotationFlag)

        self.vn = None
        self.fn = None
        self.vf = None
        self.v_indexed_by_faces = None
        self.vc = np.array([1.0, 0.0, 0.0])

    def load_from_ply(self, filename, rotationFlag):

        plydata = PlyData.read(filename)
        self.plydata = plydata

        self.f = np.vstack(plydata['face'].data['vertex_indices'])	
	if rotationFlag is not None:
        	x = -plydata['vertex'].data['z']
        	y = plydata['vertex'].data['x']
        	z = -plydata['vertex'].data['y']
	else:
        	x = plydata['vertex'].data['x']
        	y = plydata['vertex'].data['y']
        	z = plydata['vertex'].data['z']
        self.v = np.zeros([x.size, 3])
        self.v[:,0] = x
        self.v[:,1] = y
        self.v[:,2] = z
        #self.vn = self.estimate_vertex_normals()


    def save_ply(self, filename):
        vertex = np.array([tuple(i) for i in self.v], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        face = np.array([(tuple(i), 0, 100, 255) for i in self.f] , 
            dtype=[('vertex_indices', 'i4', (3,)),
            ('red', 'u1'), ('green', 'u1'),
            ('blue', 'u1')])
	edge = np.array([(tuple(i)[0], tuple(i)[1], 255, 255, 255) for i in self.e] , 
            dtype=[('vertex1', 'i4'), ('vertex2', 'i4'),
            ('red', 'u1'), ('green', 'u1'),
            ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face, 'face')
	el3 = PlyElement.describe(edge, 'edge')
        plydata = PlyData([el, el2, el3])
        plydata.write(filename)


    def set_vertex_colors(self, color_name):
        self.vc = np.array([1.0, 0.0, 0.0])

    def get_vertices_per_face(self):
        """
        Returns an array with the vertices for each face
        """
        if (self.v_indexed_by_faces is None and self.v is not None):
            self.v_indexed_by_faces = self.v[self.f]
        return self.v_indexed_by_faces


    def get_vertex_faces(self):
        """
        List mapping each vertex index to a list of face indices that use it.
        """
        if self.vf is None:
            self.vf = [[] for i in xrange(len(self.v))]
            for i in xrange(self.f.shape[0]):
                face = self.f[i]
                for ind in face:
                    self.vf[ind].append(i)
        return self.vf

    def faces_by_vertex(self):
        import scipy.sparse as sp
        row = self.f.flatten()
        col = np.array([range(self.f.shape[0])]*3).T.flatten()
        data = np.ones(len(col))
        faces_by_vertex = sp.csr_matrix((data, (row, col)), shape=(self.v.shape[0], self.f.shape[0]))
        return faces_by_vertex


    def get_face_normals(self):
        """
        Return a (Nf, 3) array of normal vectors for face.
        """
        if self.fn is None:
            v = self.get_vertices_per_face()
            self.fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            return self.fn

 
    def estimate_vertex_normals(self):
        """
        Return a (N,3) array of normal vectors.
        """
        if self.vn is None:
            faceNorms = self.get_face_normals()
            vertFaces = self.get_vertex_faces()
            faces_by_vert = self.faces_by_vertex()
            norm_not_scaled = faces_by_vert * faceNorms
            norms = (np.sum(norm_not_scaled**2.0, axis=1)**0.5).T
            norms[norms == 0] = 1.0
            return (norm_not_scaled.T/norms).T

        return self.vn

    def save_lnd(self, filename):
	f = open(filename, 'w')
	f.write('SUBJECT_ID = vstemplate.ply\n')
	f.write('SCAN_TYPE  = NO TYPE\n')
	f.write('STUDY_NAME = * * *  NO STUDY  * * *\n')
	f.write('LAND_STUDY = New Study\n')
	f.write('STD_LAND = 0\n')
	f.write('AUX_LAND = 14\n')
	f.write('STANDARD =\n')
	f.write('AUX =\n')
	jointList = ['SRellion','Rt. Acromion', 'Rt. Olecranon', 'Rt. Ulnar Styloid', 'Lt. Acromion', 'Lt. Olecranon', 'Lt. Ulnar Styloid', 'Rt. Knee Crease', 'Rt. Calcaneous, Post.', 'Rt. Digit II', 'Lt. Knee Crease', 'Lt. Calcaneous, Post.', 'Lt. Digit II', 'Crotch']
	
	#for i in range(0,14):
		#f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[i][0], 1000*self.v[i][1], 1000*self.v[i][2], jointList[i]))

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[15][0], 1000*self.v[15][1], 1000*self.v[15][2], jointList[0])) #forehead

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[8][0], 1000*self.v[8][1], 1000*self.v[8][2], jointList[1])) #right shoulder
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[10][0], 1000*self.v[10][1], 1000*self.v[10][2], jointList[2])) #right elbow
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[13][0], 1000*self.v[13][1], 1000*self.v[13][2], jointList[3])) #right wrist

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[14][0], 1000*self.v[14][1], 1000*self.v[14][2], jointList[4])) #left shoulder
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[12][0], 1000*self.v[12][1], 1000*self.v[12][2], jointList[5])) #left elbow
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[0][0], 1000*self.v[0][1], 1000*self.v[0][2], jointList[6])) #left wrist

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[6][0], 1000*self.v[6][1], 1000*self.v[6][2], jointList[7])) #right knee
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[4][0], 1000*self.v[4][1], 1000*self.v[4][2], jointList[8])) #right heel
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[11][0], 1000*self.v[11][1], 1000*self.v[11][2], jointList[9])) #right foot

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[7][0], 1000*self.v[7][1], 1000*self.v[7][2], jointList[10])) #left knee
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[9][0], 1000*self.v[9][1], 1000*self.v[9][2], jointList[11])) #left heel
	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[5][0], 1000*self.v[5][1], 1000*self.v[5][2], jointList[12])) #left foot

	f.write('\t0 \t0 \t0 \t0 \t%.3f \t%.3f \t%.3f %s\n'%(1000*self.v[3][0], 1000*self.v[3][1], 1000*self.v[3][2], jointList[13])) #belly
	f.write('END =')
	f.close()









