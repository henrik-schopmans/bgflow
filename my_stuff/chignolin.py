import torch
import numpy as np
import os
from bgmol.datasets import ChignolinOBC2PT
from matplotlib import pyplot as plt
import matplotlib as mpl
from bgmol.zmatrix import ZMatrixFactory

temperatures = [250., 274.64013583, 301.70881683, 331.44540173, 364.11284061, 400.]

for current_temperature in temperatures:
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    dtype = torch.float32
    ctx = {"device": device, "dtype": dtype}

    is_data_here = os.path.isdir("ChignolinOBC2PT")
    dataset = ChignolinOBC2PT(download=not is_data_here, read=True, temperature=current_temperature)

    system = dataset.system
    coordinates = dataset.coordinates
    temperature = dataset.temperature
    dim = dataset.dim

    print(coordinates.shape)

    if False:
        temperature_str = str(int(current_temperature))
        np.save("/home/henrik/Dokumente/Archive/Big_Files/single/chignolin_implicit_noe/" + temperature_str + ".npy", coordinates)
    else:

        n_train = len(dataset)//2
        n_test = len(dataset) - n_train
        permutation = np.random.permutation(n_train)

        all_data = coordinates.reshape(-1, dataset.dim)
        training_data = torch.tensor(all_data[permutation], **ctx)
        test_data = torch.tensor(all_data[permutation + n_train], **ctx)

        if True:
            tics = system.to_tics(dataset.xyz, eigs_kept=2)

            def plot_tics(ax, tics, bins=100):
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 4)
                ax.set_xlabel("TIC1")
                ax.set_ylabel("TIC2")
                ax.hist2d(tics[...,0], tics[...,1], bins=bins, range=[[-2,2], [-2,4]], density=True, norm=mpl.colors.LogNorm())
                
            fig, ax = plt.subplots(figsize=(3,3))
            plot_tics(ax, tics)

            plt.show()

        else:

            #c_alpha = system.mdtraj_topology.select("name == CA")
            zfactory = ZMatrixFactory(system.mdtraj_topology, cartesian=())
            z_matrix, fixed_atoms = zfactory.build_naive()

            # "warnings.warn("Z-matrix torsions are not fully independent because of a constraint on HA.")"
            # TODO: What exactly does this mean?