from dataclasses import dataclass
from BaseActivation import BaseActivation
import numpy as np

@dataclass
class Layer:
    weights: np.ndarray
    activation: BaseActivation

