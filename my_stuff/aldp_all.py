from bgmol.systems.ala2 import AlanineDipeptideImplicit
from bgflow import TORSIONS, BONDS, ANGLES
import bgflow as bg
import torch
import bgmol
import time

system = AlanineDipeptideImplicit(constraints=None, hydrogenMass=None)
system.reinitialize_energy_model(n_workers=16)

coordinate_trafo = bg.GlobalInternalCoordinateTransformation(
    z_matrix=bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX,
    enforce_boundaries=True,
    normalize_angles=True,
)

shape_info = bg.ShapeDictionary.from_coordinate_transform(coordinate_trafo)

builder = bg.BoltzmannGeneratorBuilder(
    prior_dims=shape_info,
    target=system.energy_model,
    device="cuda",
    dtype=torch.float32,
)

transformer_kwargs = dict()
transformer_kwargs["spline_disable_identity_transform"] = True

builder.add_condition(
    ANGLES,
    on=TORSIONS,
    add_reverse=False,
    conditioner_type="residual",
    transformer_kwargs=transformer_kwargs,
    context_dims=1,
)

builder.add_map_to_ic_domains()
builder.add_map_to_cartesian(coordinate_transform=coordinate_trafo)

generator = builder.build_generator()

torch.manual_seed(123)
torch.cuda.manual_seed(123)

print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))

chiral_torsions = bgmol.is_chiral_torsion(
    coordinate_trafo.torsion_indices, system.mdtraj_topology
)

generator.flow.forward(
    torch.randn(1000, 66, device="cuda"),
    context=torch.randn(1000, 1, device="cuda"),
    inverse=True,
)

system.energy_model.energy(torch.randn(10000, 66, device="cuda"))
system.energy_model.energy(torch.randn(10000, 66, device="cuda"))

test_samples = torch.randn(10000, 66, device="cuda")

start = time.time()
system.energy_model.energy(test_samples)
print("Time for 10k samples:", time.time() - start, "s")

system.energy_model._bridge.context_wrapper.terminate()
