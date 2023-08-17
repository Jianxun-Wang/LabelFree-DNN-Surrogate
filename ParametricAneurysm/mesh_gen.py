import numpy as np

def create_3d_mesh(x_values, mesh_2d, z_values):
    """
    Create a 3D mesh based on given x_axis values, 2D mesh, and z_axis values.

    Args:
    - x_values: A 1D numpy array containing x values.
    - mesh_2d: A 2D numpy array containing flatted x and y values.
    - z_values: A 1D numpy array containing z values.

    Returns:
    - y_mesh: A 2D numpy array containing y values for the 3D mesh.
    - z_mesh: A 2D numpy array containing z values for the 3D mesh.
    """
    tmp_3d = np.expand_dims(np.tile(z_values, len(mesh_2d)), 1).astype('float')
    mesh_3d = np.expand_dims(np.repeat(mesh_2d, len(z_values)), axis=1)

    return mesh_3d, tmp_3d