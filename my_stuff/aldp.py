import torch
import bgflow as bg
from bgmol.datasets import Ala2TSF300
from bgmol.zmatrix import ZMatrixFactory
import numpy as np
import bgmol
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float32
ctx = torch.zeros([], device=device, dtype=dtype)

is_data_here = os.path.isfile("Ala2TSF300.npy")
dataset = Ala2TSF300(download=(not is_data_here), read=True)
system = dataset.system
coordinates = dataset.coordinates
dim_cartesian = len(system.rigid_block) * 3 - 6

n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)
all_data = coordinates.reshape(-1, dataset.dim)
training_data = torch.tensor(all_data[permutation]).to(ctx)

#coordinate_transform = bg.MixedCoordinateTransformation(
#    data=training_data, 
#    z_matrix=system.z_matrix,
#    fixed_atoms=system.rigid_block,
#    keepdims=dim_cartesian, 
#    normalize_angles=True,
#).to(ctx)

if True: # generate the z-matrix
    zfactory = ZMatrixFactory(system.mdtraj_topology, cartesian=())
    z_matrix, fixed_atoms = zfactory.build_naive()
    coordinate_trafo = bg.GlobalInternalCoordinateTransformation(z_matrix=z_matrix, enforce_boundaries=True)
else: # use the default z-matrix
    coordinate_trafo = bg.GlobalInternalCoordinateTransformation(z_matrix=bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX, enforce_boundaries=True)

shape_info = bg.ShapeDictionary.from_coordinate_transform(
    coordinate_trafo, #n_constraints=
)
print(shape_info)