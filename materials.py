"""Material model container."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    E: float           # Young's modulus [Pa]
    nu: float          # Poisson ratio [-]
    density: float     # [kg/m^3]
    failure_stress: float  # yield / failure stress [Pa]

    @classmethod
    def from_config(cls, mat: dict) -> "Material":
        return cls(
            name=str(mat["name"]),
            E=float(mat["youngs_modulus_pa"]),
            nu=float(mat["poisson_ratio"]),
            density=float(mat["density_kg_m3"]),
            failure_stress=float(mat["failure_stress_pa"]),
        )
