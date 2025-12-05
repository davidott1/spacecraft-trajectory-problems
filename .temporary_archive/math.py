import numpy as np

def rotation_matrix(
  axis        : str,
  angle       : float,
  rotate_type : str = 'vector', # 'vector' or 'frame'
) -> np.ndarray:
  """
  Generate a rotation matrix for a given axis and angle.

  Input:
  ------
  axis : str
    Rotation axis ('x', 'y', or 'z')
  angle : float
    Rotation angle [rad]
  rotate_type : str
    'vector' for rotating a vector, 'frame' for rotating a frame (default: 'vector')
  
  Output:
  -------
  rot_mat : np.ndarray
    3x3 numpy array representing the rotation matrix
  """
  if rotate_type.lower() == 'frame':
    angle = -angle

  if axis.lower() == 'x':
    rot_mat = np.array([
      [ 1,             0,              0 ],
      [ 0, np.cos(angle), -np.sin(angle) ],
      [ 0, np.sin(angle),  np.cos(angle) ]
    ])
  elif axis.lower() == 'y':
    rot_mat = np.array([
      [  np.cos(angle), 0, np.sin(angle) ],
      [              0, 1,             0 ],
      [ -np.sin(angle), 0, np.cos(angle) ]
    ])
  elif axis.lower() == 'z':
    rot_mat = np.array([
      [ np.cos(angle), -np.sin(angle), 0 ],
      [ np.sin(angle),  np.cos(angle), 0 ],
      [             0,              0, 1 ]
    ])
  else:
    raise ValueError("Axis must be 'x', 'y', or 'z'")
  return rot_mat
