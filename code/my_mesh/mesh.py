
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
    def __init__(self, v=None, f=None, filename=None):
        self.v = None
        self.f = None

        if v is not None:
            self.v = np.array(v, dtype=np.float64)
        if f is not None:
            self.f = np.require(f, dtype=np.uint32)

        if filename is not None:
            self.load_from_ply(filename)

        self.vn = None
        self.fn = None
        self.vf = None
        self.v_indexed_by_faces = None
        self.vc = np.array([1.0, 0.0, 0.0])

    def load_from_ply(self, filename):

        plydata = PlyData.read(filename)
        self.plydata = plydata

        self.f = np.vstack(plydata['face'].data['vertex_indices'])
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
        face = np.array([(tuple(i), 255, 255, 255) for i in self.f] , 
            dtype=[('vertex_indices', 'i4', (3,)),
            ('red', 'u1'), ('green', 'u1'),
            ('blue', 'u1')])
        el = PlyElement.describe(vertex, 'vertex')
        el2 = PlyElement.describe(face, 'face')
        plydata = PlyData([el, el2])
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

