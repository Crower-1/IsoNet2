'''
rotation_list = {

0: [(((0,1),1),((1,2),1)), (((0,2),1),((1,2),1))],
1: [(((1,0),1),((1,2),1)), (((2,0),1),((1,2),1))],
2: [(((0,1),1),((1,2),3)), (((0,2),1),((1,2),3))],
3: [(((1,0),1),((1,2),3)), (((2,0),1),((1,2),3))],
4: [(((0,1),1),((1,2),0)), (((0,2),1),((1,2),0))],
5: [(((1,0),1),((1,2),0)), (((2,0),1),((1,2),0))],
6: [(((0,1),1),((1,2),2)), (((0,2),1),((1,2),2))],
7: [(((1,0),1),((1,2),2)), (((2,0),1),((1,2),2))],
}



rotation_list = {

0: [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1))],
1: [(((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3))],
2: [(((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1))],
3: [(((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3))],
}

'''
# this is used in mwr_cli 26 Dec
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((1,2),1),((0,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((1,2),1),((0,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((1,2),3),((0,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((1,2),3),((0,2),2)), (((0,2),1),((1,2),3))]
# this is from old dgx(workhorse branch) version
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
                # (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
                # (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)), 
                # (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3))]

#All 20 rotation
# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3)),
#                 (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2))]

# rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
#                 (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
#                 (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),3),((1,2),1)), 
#                 (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),3),((1,2),3)),
#                 (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2))]

rotation_list_24 = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 
                    (((0,1),0),((0,2),0)), (((0,1),0),((0,2),1)), (((0,1),0),((0,2),2)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),0)), (((0,1),2),((0,2),1)), (((0,1),2),((0,2),2)), (((0,1),2),((0,2),3))]

rotation_list_aug2125 = [(((0,1),1),((0,2),0)), (((0,1),1),((0,2),1)), (((0,1),1),((0,2),2)), (((0,1),1),((0,2),3)), 
                    (((0,1),3),((0,2),0)), (((0,1),3),((0,2),1)), (((0,1),3),((0,2),2)), (((0,1),3),((0,2),3)), 
                    (((1,2),1),((0,2),0)), (((1,2),1),((0,2),1)), (((1,2),1),((0,2),2)), (((1,2),1),((0,2),3)), 
                    (((1,2),3),((0,2),0)), (((1,2),3),((0,2),1)), (((1,2),3),((0,2),2)), (((1,2),3),((0,2),3)), 
                    (((0,1),0),((0,2),1)), (((0,1),0),((0,2),3)), 
                    (((0,1),2),((0,2),1)), (((0,1),2),((0,2),3))]

#All 20 rotation
rotation_list = [(((0,1),1),((1,2),0)), (((0,1),1),((1,2),1)), (((0,2),1),((1,2),0)), (((0,2),1),((1,2),1)), 
                (((0,1),1),((1,2),2)), (((0,1),1),((1,2),3)), (((0,2),1),((1,2),2)), (((0,2),1),((1,2),3)), 
                (((0,1),3),((1,2),0)), (((0,1),3),((1,2),1)), (((0,2),3),((1,2),0)), (((0,2),1),((1,2),1)), 
                (((0,1),3),((1,2),2)), (((0,1),3),((1,2),3)), (((0,2),3),((1,2),2)), (((0,2),1),((1,2),3)),
                (((1,2),1),((0,2),0)), (((1,2),1),((0,2),2)), (((1,2),3),((0,2),0)), (((1,2),3),((0,2),2))]

#rotation_list = [(((0,1),0),((0,2),0)), (((0,1),1),((0,1),0)), (((0,1),1),((0,1),1)),
#                 (((0,2),0),((0,2),0)), (((0,2),1),((0,1),0)), (((0,2),1),((0,1),1))]

#((0,1),0),((1,2),0)), (((0,1),0),((1,2),1)),
#((0,1),0),((1,2),2)), (((0,1),0),((1,2),3)),
#((0,1),2),((1,2),0)), (((0,1),2),((1,2),1)),
#((0,1),2),((1,2),2)), (((0,1),2),((1,2),3)),

import itertools

import torch
import torch.nn.functional as F


def _permutation_parity(permutation):
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )
    return -1 if inversions % 2 else 1


# The 24 orientation-preserving symmetries of a cube.  ``permutation`` maps
# output spatial axes to input spatial axes and ``signs`` records axis
# reversals.  Unlike the historical hand-written list above, this list is
# complete and contains no duplicates.
cube_rotations_24 = tuple(
    (permutation, signs)
    for permutation in itertools.permutations(range(3))
    for signs in itertools.product((-1, 1), repeat=3)
    if _permutation_parity(permutation) * signs[0] * signs[1] * signs[2] == 1
)


def rotate_cube_24(volume, rotation, fourier=False):
    """Apply an exact cube rotation to a BCHWD tensor.

    Fourier masks use a periodic frequency-axis reversal so that the DC voxel
    remains at the fftshift centre for even-sized arrays.
    """
    permutation, signs = rotation
    spatial_dims = tuple(axis + 2 for axis in permutation)
    rotated = volume.permute(0, 1, *spatial_dims)
    for axis, sign in enumerate(signs):
        if sign >= 0:
            continue
        dim = axis + 2
        if fourier:
            size = rotated.shape[dim]
            centre = size // 2
            indices = (2 * centre - torch.arange(size, device=rotated.device)) % size
            rotated = rotated.index_select(dim, indices)
        else:
            rotated = torch.flip(rotated, dims=(dim,))
    return rotated

def sample_rot_axis_and_angle(device=None):
    """Sample a Haar-uniform SO(3) rotation as axis and angle."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    quaternion = torch.randn(4, device=device)
    quaternion = quaternion / quaternion.norm().clamp_min(1e-8)
    scalar = quaternion[0].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(scalar)
    sin_half = torch.sqrt((1.0 - scalar.square()).clamp_min(0.0))
    if sin_half < 1e-6:
        axis = torch.tensor([1.0, 0.0, 0.0], device=device)
    else:
        axis = quaternion[1:] / sin_half
    return [axis, angle]


def rotation_matrix(axis, angle):
    axis = axis / axis.norm()  # Normalize the axis
    cos_theta = torch.cos(angle)
    sin_theta = torch.sin(angle)
    ux, uy, uz = axis

    # Rotation matrix using Rodrigues' rotation formula
    R = torch.stack([
        torch.stack([cos_theta + ux**2 * (1 - cos_theta), ux * uy * (1 - cos_theta) - uz * sin_theta, ux * uz * (1 - cos_theta) + uy * sin_theta]),
        torch.stack([uy * ux * (1 - cos_theta) + uz * sin_theta, cos_theta + uy**2 * (1 - cos_theta), uy * uz * (1 - cos_theta) - ux * sin_theta]),
        torch.stack([uz * ux * (1 - cos_theta) - uy * sin_theta, uz * uy * (1 - cos_theta) + ux * sin_theta, cos_theta + uz**2 * (1 - cos_theta)]),
    ]).to(dtype=torch.float32)

    return R

# Function to rotate the volume using affine transformation
def rotate_vol_around_axis_torch(volume, rot, fourier=False):
    axis = rot[0]
    angle = rot[1]
    # Ensure volume is on the correct device (either 'cpu' or 'cuda')
    device = volume.device
    
    batch_size, _, Z, Y, X = volume.shape

    # Get the 3x3 rotation matrix
    R = rotation_matrix(axis, angle).to(device)

    # grid_sample coordinates are ordered X, Y, Z.  Density is rotated around
    # the geometric voxel centre; fftshift masks are rotated around their DC
    # voxel instead.  Building the grid explicitly avoids the half-voxel DC
    # displacement produced by affine_grid on even Fourier arrays.
    centres = (
        torch.tensor([X // 2, Y // 2, Z // 2], device=device, dtype=torch.float32)
        if fourier
        else torch.tensor([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2], device=device)
    )
    z = torch.arange(Z, device=device, dtype=torch.float32) - centres[2]
    y = torch.arange(Y, device=device, dtype=torch.float32) - centres[1]
    x = torch.arange(X, device=device, dtype=torch.float32) - centres[0]
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
    coordinates = torch.stack((xx, yy, zz), dim=-1)
    source = torch.matmul(coordinates, R.transpose(0, 1)) + centres
    grid = torch.empty_like(source)
    grid[..., 0] = 2.0 * source[..., 0] / max(X - 1, 1) - 1.0
    grid[..., 1] = 2.0 * source[..., 1] / max(Y - 1, 1) - 1.0
    grid[..., 2] = 2.0 * source[..., 2] / max(Z - 1, 1) - 1.0
    grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    return F.grid_sample(
        volume,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True,
    )
