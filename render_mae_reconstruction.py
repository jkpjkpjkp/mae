import os
import pickle
import numpy as np
import torch
import trimesh
import healpy as hp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from a import read_stl_file, center_of_mass
from c import mae_1d_tiny_patch64_short

# --- Config ---
STL_PATH = 'data/A39/44.stl'
NPY_PATH = 'processed_data_gpu/A39_44.npy'
CHECKPOINT_PATH = 'checkpoints/mae_tiny_final.pth'
NSIDE = 16  # adjust if needed

# --- Load original mesh ---
vertices, triangles = read_stl_file(STL_PATH)
com = center_of_mass(vertices, triangles)

# --- Load preprocessed distances ---
true_distances = np.load(NPY_PATH)

# --- Get HEALPix directions ---
npix = hp.nside2npix(NSIDE)
pixel_indices = np.arange(npix)
pixel_directions = np.array(hp.pix2vec(NSIDE, pixel_indices, nest=True)).T  # (npix, 3)

# --- Load MAE model and checkpoint ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mae_1d_tiny_patch64_short().to(device)
ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# --- Prepare input for MAE (normalize as in training) ---
input_seq = torch.from_numpy(true_distances).float().unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, seq_len)
input_seq = (input_seq - input_seq.mean()) / (input_seq.std() + 1e-8)

# --- Run MAE reconstruction ---
with torch.no_grad():
    loss, pred, mask = model(input_seq, mask_ratio=0.75)
    # Unpatchify to get reconstructed sequence
    recon_seq = model.unpatchify(pred).cpu().numpy()[0, 0]  # (seq_len,)

# --- Convert distances to 3D points ---
true_points = com + pixel_directions * true_distances[:, None]
recon_points = com + pixel_directions * recon_seq[:, None]

# --- Render using matplotlib 3D ---
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Original mesh
mesh_obj = trimesh.Trimesh(vertices, triangles)
try:
    # Use plot_trisurf directly on ax1
    ax1.plot_trisurf(mesh_obj.vertices[:, 0], mesh_obj.vertices[:, 1], mesh_obj.vertices[:, 2], triangles=mesh_obj.faces, color='lightblue', alpha=0.5)
except Exception as e:
    print(f"plot_trisurf failed: {e}, falling back to scatter.")
    ax1.scatter(mesh_obj.vertices[:, 0], mesh_obj.vertices[:, 1], mesh_obj.vertices[:, 2], c='lightblue', alpha=0.5)
ax1.set_title('Original STL Mesh')

# MAE reconstruction as point cloud
ax2.scatter(recon_points[:, 0], recon_points[:, 1], recon_points[:, 2], c='r', s=1)
ax2.set_title('MAE Reconstruction (point cloud)')

for ax in [ax1, ax2]:
    ax.axis('off')

plt.tight_layout()
# plt.show()
plt.savefig('mae_vs_gt.png')
print('Saved comparison image to mae_vs_gt.png') 