import tomopy
import dxchange
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level=logging.INFO)
proj, flat, dark, theta = dxchange.read_aps_32id(
    fname='tooth.h5',
    sino=(0, 2),  # Select the sinogram range to reconstruct.
)
print(proj.shape, flat.shape, dark.shape, theta.shape)
plt.imshow(proj[:, 0, :])
plt.show()
if theta is None:
    theta = tomopy.angles(proj.shape[0])
proj = tomopy.normalize(proj, flat, dark)
proj = tomopy.minus_log(proj)
rot_center = tomopy.find_center(proj, theta, init=290, ind=0, tol=0.5)
print(rot_center)
recon = tomopy.recon(proj, theta, center=rot_center, algorithm='gridrec', sinogram_order=False)
recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)
plt.imshow(recon[0, :, :])
plt.show()