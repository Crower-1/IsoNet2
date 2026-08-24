
import numpy as np

def mw2D_odd(dim,missingAngle=[30,30]):
    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=np.pi/180*(90-missingAngle)
    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:# and y!=0:
                theta=np.pi/2
            #elif x==0 and y==0:
            #    theta=0
            #elif x!=0 and y==0:
            #    theta=np.pi/2
            else:
                theta=abs(np.arctan(y/x))

            if x**2+y**2<=(dim/2)**2:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1#np.cos(theta)
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1#np.cos(theta)

            if int(y) == 0:
                mw[i,j]=1

    return mw

def mw2D(dim, missingAngle=[30,30], tilt_step=None, start_dim = 48):

    if tilt_step not in [None,"None"]:
        import os
        from IsoNet.utils.fileio import read_mrc
        path = os.path.abspath(__file__).split("IsoNet/utils")[0]
        file = f"{path}missing_wedges/simulated_F_{dim}_{tilt_step}.mrc"
        precalculated_mw,_ = read_mrc(file)
    else:
        precalculated_mw = np.ones((dim, dim))
        start_dim = 0


    mw=np.zeros((dim,dim),dtype=np.double)
    missingAngle = np.array(missingAngle)
    missing=90-missingAngle

    for i in range(dim):
        for j in range(dim):
            y=(i-dim/2)
            x=(j-dim/2)
            if x==0:
                theta=90
            else:
                theta=np.rad2deg(abs(np.arctan(y/x)))

            R2 = x**2+y**2
            if R2>((start_dim)/2)**2:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=precalculated_mw[i][j]
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=precalculated_mw[i][j]
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=precalculated_mw[i][j]
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=precalculated_mw[i][j]
            else:
                if x > 0 and y > 0 and theta < missing[1]:
                    mw[i,j]=1
                if x < 0 and y < 0 and theta < missing[1]:
                    mw[i,j]=1
                if x > 0 and y < 0 and theta < missing[0]:
                    mw[i,j]=1
                if x < 0 and y > 0 and theta < missing[0]:
                    mw[i,j]=1       

            if int(y) == 0:
                mw[i,j]=1


    return mw.astype(np.float32)

def mw3D(
    dim,
    missingAngle=[30, 30],
    tilt_step=None,
    start_dim=48,
    spherical=True,
    dose_weight=1,
    tilt_axis_angle=0.0,
    taper_deg=0.0,
):
    """Return an fftshift-ordered measured Fourier-support mask.

    ``missingAngle`` retains IsoNet's historical convention: values are the
    missing angular extents derived from ``[90 + tilt_min, 90 - tilt_max]``.
    ``tilt_axis_angle`` is measured in the reconstruction XY plane, with zero
    preserving the historical Y-aligned tilt axis.  A positive ``taper_deg``
    replaces the hard wedge boundary with a cosine confidence transition.

    ``dim`` may be an integer (a cubic patch) or a full ``(Z, Y, X)`` shape.
    The latter is used by full-volume Fourier data consistency at inference.
    """
    if tilt_step not in [None, "None"]:
        # Between-tilt weighting is a legacy optional mode.  Keep its exact
        # precomputed behaviour until a soft, axis-aware equivalent exists.
        if not isinstance(dim, (int, np.integer)):
            raise ValueError("between-tilt masks require a cubic integer size")
        mw = mw2D(int(dim), missingAngle, tilt_step, start_dim)
        mw = np.repeat(mw[:, np.newaxis, :], int(dim), axis=1).astype(np.float32)
        if spherical:
            centre = int(dim) // 2
            z, y, x = np.ogrid[:int(dim), :int(dim), :int(dim)]
            radial_squared = (
                ((z - centre) / dose_weight) ** 2
                + ((y - centre) / dose_weight) ** 2
                + ((x - centre) / dose_weight) ** 2
            )
            mw[radial_squared > centre**2] = 0.0
        return mw

    if isinstance(dim, (int, np.integer)):
        shape = (int(dim), int(dim), int(dim))
    else:
        shape = tuple(int(value) for value in dim)
        if len(shape) != 3:
            raise ValueError(f"dim must be an integer or (Z, Y, X), got {dim}")

    z_size, y_size, x_size = shape
    kz_values = (np.arange(z_size, dtype=np.float32) - z_size // 2) / z_size
    ky_values = (np.arange(y_size, dtype=np.float32) - y_size // 2) / y_size
    kx_values = (np.arange(x_size, dtype=np.float32) - x_size // 2) / x_size
    ky, kx = np.meshgrid(ky_values, kx_values, indexing="ij")

    axis_angle = np.deg2rad(float(tilt_axis_angle))
    # Component perpendicular to the tilt axis.  angle=0 means a Y tilt axis
    # and therefore reproduces the original XZ wedge geometry.
    k_perpendicular = kx * np.cos(axis_angle) - ky * np.sin(axis_angle)

    missing_angle = np.asarray(missingAngle, dtype=np.float32)
    negative_tilt_limit = 90.0 - missing_angle[0]
    positive_tilt_limit = 90.0 - missing_angle[1]
    taper_deg = max(float(taper_deg), 0.0)
    mw = np.empty(shape, dtype=np.float32)
    radial_xy_squared = ky**2 + kx**2
    for z_index, kz in enumerate(kz_values):
        theta = np.rad2deg(np.arctan2(abs(kz), np.abs(k_perpendicular)))
        same_sign = kz * k_perpendicular >= 0
        cutoff = np.where(same_sign, positive_tilt_limit, negative_tilt_limit)
        if taper_deg == 0:
            plane = (theta <= cutoff).astype(np.float32)
        else:
            lower = cutoff - taper_deg / 2.0
            transition = np.clip((theta - lower) / taper_deg, 0.0, 1.0)
            plane = (0.5 * (1.0 + np.cos(np.pi * transition))).astype(np.float32)
            plane[theta <= lower] = 1.0
            plane[theta >= cutoff + taper_deg / 2.0] = 0.0
        if abs(kz) < np.finfo(np.float32).eps:
            plane.fill(1.0)
        if spherical:
            radial_frequency_squared = (kz**2 + radial_xy_squared) / max(
                float(dose_weight) ** 2, 1e-8
            )
            plane[radial_frequency_squared > 0.25] = 0.0
        mw[z_index] = plane
    return mw.astype(np.float32)


def get_F_cone(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = ( (i-size/2)**2 + (j-size/2)**2 ) **0.5
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
    data=1-data
    return data

def get_F_wedge(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = abs(i-size/2)
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
    data=1-data
    return data

def get_F_double_wedge(size=160, angle=45):
    data = np.zeros((size,size,size), dtype = np.float32)
    for i in range(size):
        for j in range(size):
            for k in range(size):
                r = abs(i-size/2)
                z = k  - size/2
                threshold = r*np.tan(np.radians(angle))
                if abs(z) > threshold:
                    data[k,j,i] = 1
                if data[k,j,i] ==0:
                    r = abs(j-size/2)
                    z = k  - size/2
                    threshold = r*np.tan(np.radians(angle))
                    if abs(z) > threshold:
                        data[k,j,i] = 1
    return data


    
