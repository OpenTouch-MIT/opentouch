import torch
from smplx import MANO
import numpy as np
import trimesh
import pyrender
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
from scipy import sparse
from scipy.sparse import csr_matrix

def smooth_vertex_colors(mesh, iterations=1):
    # Assume mesh.visual.vertex_colors exists and is (n_vertices, 4)
    colors = mesh.visual.vertex_colors.astype(np.float64)
    
    # Get adjacency matrix
    G = mesh.vertex_adjacency_graph  # scipy sparse CSR

    n = len(mesh.vertices)
    edges = list(G.edges())

    row, col = zip(*edges)
    data = np.ones(len(row), dtype=np.float32)
    adjacency = csr_matrix((data, (row, col)), shape=(n, n))
    # Ensure symmetric (undirected)
    adjacency = adjacency + adjacency.T
    # Add self-connections
    adjacency.setdiag(1)

    adjacency = adjacency + sparse.eye(n)
    
    # Normalize adjacency so each row sums to 1 (average)
    row_sums = np.array(adjacency.sum(axis=1)).flatten()
    adjacency_normalized = adjacency.multiply(1.0 / row_sums[:, np.newaxis])
    
    for _ in range(iterations):
        colors = adjacency_normalized @ colors  # smooth
    
    # Assign back (cast to uint8)
    mesh.visual.vertex_colors = colors.astype(np.uint8)
    return mesh

class ManoRenderer:
    def __init__(self, image_size=(512, 512), focal_length=8000, mano_vertices=None, mano_faces=None):
        self.image_size = image_size
        self.focal_length = focal_length
        self.renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
        self.vertices = mano_vertices
        self.faces = mano_faces

    def add_dir_light(self, theta_deg, phi_deg, intensity):
        T = np.eye(4)
        # spherical to Cartesian
        th = np.radians(theta_deg); ph = np.radians(phi_deg); r = 4.0
        pos = np.array([r*np.sin(th)*np.cos(ph), r*np.sin(th)*np.sin(ph), r*np.cos(th)])
        z = -pos/np.linalg.norm(pos); x = np.array([-z[1], z[0], 0.]); 
        if np.linalg.norm(x) == 0: x = np.array([1.,0.,0.])
        x /= np.linalg.norm(x); y = np.cross(z, x)
        T[:3,:3] = np.stack([x,y,z], axis=1); T[:3,3] = pos
        node = pyrender.Node(light=pyrender.DirectionalLight(color=np.ones(3), intensity=intensity), matrix=T)
        return node
    
    def render(self, vertex_colors = None, colormap_fn = lambda x: (np.array(x)),
               camera_t: np.ndarray = np.array([0, 0, 2]), smooth=True) -> np.ndarray:
        """
        Render a MANO mesh with per-face colors.

        Args:
            vertices (np.ndarray): (V, 3) array of vertex positions.
            faces (np.ndarray): (F, 3) array of face indices.
            face_colors (np.ndarray): (F, 4) array of per-face RGBA colors.
            camera_t (np.ndarray): (3,) camera translation vector.

        Returns:
            np.ndarray: (H, W, 3) RGB image of the rendered mesh.
        """
        # Build per-face-colored mesh by duplicating vertices for each face
        

        # Create pyrender mesh
        tri_mesh = trimesh.Trimesh(vertices=self.vertices,
                                faces=self.faces,
                                vertex_colors=vertex_colors,
                                process=False)
        # tri_mesh = smooth_vertex_colors(tri_mesh, iterations=5)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0])
        tri_mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0])
        tri_mesh.apply_transform(rot)
        _ = tri_mesh.vertex_normals

        # mat = pyrender.MetallicRoughnessMaterial(
        #     baseColorFactor=[1, 1, 1, 0],  # don't tint vertex colors
        #     metallicFactor=0.005,                    # shiny
        #     roughnessFactor=0.08,                  # lower = glossier
        #     alphaMode='MASK',                     # <<< keep alpha (transparent background)
        #     doubleSided=True,
        #     alphaCutoff = 0.5
        # )

        mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=True)

        # Build scenergb()
        # scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[0.4, 0.4, 0.4])
        scene = pyrender.Scene(bg_color=[140, 228, 255, 0], ambient_light=[0.4, 0.4, 0.4])
        scene.add(mesh)

        # Add camera
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length, fy=self.focal_length,
            cx=self.image_size[0] / 2.0, cy=self.image_size[1] / 2.0, zfar=1e12
        )
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_t
        scene.add(camera, pose=camera_pose)

        node = self.add_dir_light(40,   0, 3.0)   # key
        scene.add_node(node)
        node = self.add_dir_light(60, 120, 2.0)   # fill
        scene.add_node(node)
        node = self.add_dir_light(75,-120, 2.0)   # rim
        scene.add_node(node)
    
        # # Add lighting
        # for node in self.create_raymond_lights():
        #     scene.add_node(node)

        # Render
        color, _ = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        return (color[:, :, :3].astype(np.uint8))  # Return RGB

    
    def create_raymond_lights(self, n_lights=3, elevation=np.pi / 3, dist=4.0):
        thetas = elevation * np.ones(n_lights)
        phis = 2 * np.pi * np.arange(n_lights) / n_lights
        nodes = []

        for phi, theta in zip(phis, thetas):
            x = dist * np.sin(theta) * np.cos(phi)
            y = dist * np.sin(theta) * np.sin(phi)
            z = dist * np.cos(theta)
            pos = np.array([x, y, z])
            z_axis = -pos / np.linalg.norm(pos)
            x_axis = np.array([-z_axis[1], z_axis[0], 0])
            if np.linalg.norm(x_axis) == 0:
                x_axis = np.array([1, 0, 0])
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            mat = np.eye(4)
            mat[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
            mat[:3, 3] = pos
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=2.0),
                matrix=mat
            ))
        return nodes


def _resolve_mano_pkl(filename='MANO_RIGHT.pkl'):
    for candidate in [
        os.path.join("data", filename),
        os.path.join(os.path.dirname(__file__), "data", filename),
        os.path.join(os.path.dirname(__file__), "scratch", filename),
    ]:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Missing {filename}")


def load_mano(mano_model_path=None, is_rhand=True):
    if mano_model_path is None:
        mano_model_path = _resolve_mano_pkl('MANO_RIGHT.pkl' if is_rhand else 'MANO_LEFT.pkl')
    mano_model = MANO(
        model_path=mano_model_path,
        is_rhand=is_rhand,
        use_pca=False,
        flat_hand_mean=True
    ).to('cpu')
    pose = torch.zeros(1, 48)
    output = mano_model(
        global_orient=pose[:, :3],
        hand_pose=pose[:, 3:],
        betas=torch.zeros(1, 10)
    )  

    vertices = output.vertices[0].detach().cpu().numpy()
    faces = mano_model.faces
    return vertices, faces
