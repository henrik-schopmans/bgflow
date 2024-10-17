import notebooks.alanine_dipeptide_basics as basic
from bgflow import TORSIONS, BONDS, ANGLES
import bgflow as bg
import torch
import bgmol

coordinate_trafo = bg.GlobalInternalCoordinateTransformation(z_matrix=bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX, enforce_boundaries=True, normalize_angles=True)

shape_info = bg.ShapeDictionary.from_coordinate_transform(
    coordinate_trafo
)
print(shape_info)

builder = bg.BoltzmannGeneratorBuilder(
    shape_info, 
    target=basic.target_energy, 
    device=basic.device, 
    dtype=basic.dtype
)

for i in range(4):
    builder.add_condition(TORSIONS, on=ANGLES)
    builder.add_condition(ANGLES, on=TORSIONS)
for i in range(2): # TODO: Does this really make sense?
    builder.add_condition(BONDS, on=ANGLES)
    builder.add_condition(ANGLES, on=BONDS)

generator = builder.build_generator()

torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Print total number of parameters:
print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))

samples = generator.sample(1000)
print([item.shape for item in samples])

a = torch.randn(1000, 21, device="cuda")
b = torch.randn(1000, 20, device="cuda")
c = torch.randn(1000, 19, device="cuda")
xs = (a,b,c)

zs = generator._flow.forward(*xs, inverse=True) # This additionally also outputs the log determinant

print([item.shape for item in zs])

xs = generator._flow.forward(*zs[:-1])
print([item.shape for item in xs])

print(xs[0][0:5,0:5])