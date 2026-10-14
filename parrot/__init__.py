"""PARROT: Parallel-branch Hadamard Optimal Transport for Speech Emotion Recognition"""

__version__ = "1.0.0"
__author__ = "Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, and others"

from .fusion import PARROTFusion
from .transport import optimal_transport_distance

__all__ = ["PARROTFusion", "optimal_transport_distance"]
