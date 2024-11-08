from bgmol.systems.ala2 import AlanineDipeptideImplicit
from bgflow import TORSIONS, BONDS, ANGLES
import bgflow as bg
import torch
import bgmol

system = AlanineDipeptideImplicit(constraints=None, hydrogenMass=None)

coordinate_trafo = bg.GlobalInternalCoordinateTransformation(
    z_matrix=bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX,
    enforce_boundaries=True,
    normalize_angles=True,
)

shape_info = bg.ShapeDictionary.from_coordinate_transform(coordinate_trafo)
print(shape_info)

builder = bg.BoltzmannGeneratorBuilder(
    shape_info, target=system.energy_model, device="cuda", dtype=torch.float32
)

n_couplings = 4
for _ in range(n_couplings):
    builder.add_condition(TORSIONS, on=(ANGLES, BONDS))
    builder.add_condition(ANGLES, on=(TORSIONS, BONDS))
    builder.add_condition(BONDS, on=(TORSIONS, ANGLES))

generator = builder.build_generator()

torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Print total number of parameters:
print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))

samples = generator.sample(1000)
print([item.shape for item in samples])

a = torch.rand(1000, 21, device="cuda")
b = torch.rand(1000, 20, device="cuda")
c = torch.rand(1000, 19, device="cuda")
xs = (a, b, c)

zs = generator._flow.forward(
    *xs, inverse=True
)  # This additionally also outputs the log determinant

print([item.shape for item in zs])

xs = generator._flow.forward(*zs[:-1])
print([item.shape for item in xs])

print(xs[0][0:5, 0:5])
