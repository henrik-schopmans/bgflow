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

if True:
    prior_type = dict()
    prior_kwargs = dict()

    # Only torsions keep the default, which is a [0, 1] uniform distribution
    prior_type[BONDS] = bg.TruncatedNormalDistribution
    prior_type[ANGLES] = bg.TruncatedNormalDistribution

    prior_kwargs[BONDS] = {
        "mu": 0.5,
        "sigma": 0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    }
    prior_kwargs[ANGLES] = {
        "mu": 0.5,
        "sigma": 0.1,
        "lower_bound": 0.0,
        "upper_bound": 1.0,
    }
else:
    prior_type = None
    prior_kwargs = None

builder = bg.BoltzmannGeneratorBuilder(
    shape_info,
    target=system.energy_model,
    device="cuda",
    dtype=torch.float32,
    prior_type=prior_type,
    prior_kwargs=prior_kwargs,
)

n_couplings = 4
for _ in range(n_couplings):
    builder.add_condition(TORSIONS, on=(ANGLES, BONDS), conditioner_type="residual")
    builder.add_condition(ANGLES, on=(TORSIONS, BONDS), conditioner_type="residual")
    builder.add_condition(BONDS, on=(TORSIONS, ANGLES), conditioner_type="residual")

generator = builder.build_generator(use_sobol=True)

torch.manual_seed(123)
torch.cuda.manual_seed(123)

# Print total number of parameters:
print("Total number of parameters:", sum(p.numel() for p in generator.parameters()))

samples = generator.sample(1024)
print([item.shape for item in samples])
