"""Static designs of experiments — one-shot sample generation."""

from polarisopt.design.base import Design, design_registry, make_design
from polarisopt.design.lhs import LHSDesign
from polarisopt.design.manual import ManualDesign
from polarisopt.design.morris import MorrisDesign
from polarisopt.design.sobol import SobolDesign

__all__ = [
    "Design",
    "LHSDesign",
    "ManualDesign",
    "MorrisDesign",
    "SobolDesign",
    "design_registry",
    "make_design",
]
