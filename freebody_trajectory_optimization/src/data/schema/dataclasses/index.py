from dataclasses import dataclass
from typing import List

@dataclass
class RocketState:
    time     : float        # time at this state
    position : List[float]  # position in 3D space
    velocity : List[float]  # velocity in 3D space
    mass     : float        # mass of the rocket