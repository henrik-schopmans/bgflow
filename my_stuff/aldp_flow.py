import notebooks.alanine_dipeptide_basics as basic
from bgflow import TORSIONS, FIXED, BONDS, ANGLES
import bgflow as bg
import torch
import sys
sys.setrecursionlimit(2000)  # Increase limit

shape_info = bg.ShapeDictionary.from_coordinate_transform(
    basic.coordinate_transform, #n_constraints=
)
print(shape_info)

builder = bg.BoltzmannGeneratorBuilder(
    shape_info, 
    target=basic.target_energy, 
    device=basic.device, 
    dtype=basic.dtype
)

for i in range(4):
    builder.add_condition(TORSIONS, on=FIXED)
    builder.add_condition(FIXED, on=TORSIONS)
for i in range(2):
    builder.add_condition(BONDS, on=ANGLES)
    builder.add_condition(ANGLES, on=BONDS)
#builder.add_map_to_ic_domains() # This is problematic when debugging
#builder.add_map_to_cartesian(basic.coordinate_transform)
generator = builder.build_generator()

torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Print total number of parameters:
print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))

samples = generator.sample(1000)
print([item.shape for item in samples])

a = torch.randn(1000, 17, device="cuda")
b = torch.randn(1000, 17, device="cuda")
c = torch.randn(1000, 17, device="cuda")
d = torch.randn(1000, 9, device="cuda")
xs = (a,b,c,d)

zs = generator._flow.forward(*xs, inverse=True) # This additionally also outputs the log determinant

print([item.shape for item in zs])

xs = generator._flow.forward(*zs[:-1])
print([item.shape for item in xs])

print(xs[0][0:5,0:5])

# - Check exactly how periodic representations are handled => Cos/Sin representation, exactly what I am doing!
# - What kind of network are they using, anyways? => Just a dense network!

# TODO:
# - Check exactly the domain size / range of everything
# - What kind of base distributions do they use?
# - Check if the direction matters; any wrapping etc. going on for the dihedrals?
# - Where does this come from? "UserWarning: No target energy for TensorInfo(name='ANGLES', is_circular=False, is_cartesian=False)""
# - Fix the warning "UserWarning: your nflows version does not support 'enable_identity_init'."