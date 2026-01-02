"""
Propagation Schemas
===================

Dataclasses for propagation inputs, outputs, and configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.schemas.state import ClassicalOrbitalElements, ModifiedEquinoctialElements


@dataclass
class TimeGrid:
    """
    Time grid for propagation.
    
    Attributes:
        epoch_dt: Reference epoch as datetime (UTC)
        epoch_et: Reference epoch as ephemeris time [s past J2000]
        time_s: Time values relative to epoch [s], shape (N,)
        time_et: Absolute ephemeris times [s past J2000], shape (N,)
    """
    epoch_dt: datetime
    epoch_et: float
    time_s: np.ndarray
    time_et: Optional[np.ndarray] = None
    
    def __post_init__(self):
        self.time_s = np.asarray(self.time_s)
        if self.time_et is None:
            self.time_et = self.epoch_et + self.time_s
    
    @property
    def n_points(self) -> int:
        return len(self.time_s)
    
    @property
    def duration_s(self) -> float:
        return self.time_s[-1] - self.time_s[0]


@dataclass
class PropagationResult:
    """
    Result of orbit propagation.
    
    Attributes:
        success: Whether propagation completed successfully
        message: Status or error message
        time_grid: Time grid used for propagation
        state: Cartesian state array, shape (6, N)
        coe: Classical orbital elements at each time
        mee: Modified equinoctial elements at each time
        at_ephem_times: Results interpolated to ephemeris times (optional)
    """
    success: bool
    message: str = ""
    time_grid: Optional[TimeGrid] = None
    state: Optional[np.ndarray] = None
    coe: Optional[dict] = None  # Will be ClassicalOrbitalElements after full refactor
    mee: Optional[dict] = None  # Will be ModifiedEquinoctialElements after full refactor
    at_ephem_times: Optional[dict] = None
    
    # Legacy accessors for backward compatibility
    @property
    def plot_time_s(self) -> Optional[np.ndarray]:
        """Time array relative to epoch [s]."""
        if self.time_grid is not None:
            return self.time_grid.time_s
        return None
    
    @property
    def time(self) -> Optional[np.ndarray]:
        """Absolute ephemeris time array [s past J2000]."""
        if self.time_grid is not None:
            return self.time_grid.time_et
        return None
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility."""
        result = {
            'success': self.success,
            'message': self.message,
        }
        if self.time_grid is not None:
            result['plot_time_s'] = self.time_grid.time_s
            result['time'] = self.time_grid.time_et
        if self.state is not None:
            result['state'] = self.state
        if self.coe is not None:
            result['coe'] = self.coe
        if self.mee is not None:
            result['mee'] = self.mee
        if self.at_ephem_times is not None:
            result['at_ephem_times'] = self.at_ephem_times
        return result
    
    @classmethod
    def from_dict(cls, data: dict, epoch_dt: Optional[datetime] = None, epoch_et: float = 0.0) -> 'PropagationResult':
        """Create from dictionary."""
        time_grid = None
        if 'plot_time_s' in data:
            time_grid = TimeGrid(
                epoch_dt=epoch_dt or datetime(2000, 1, 1, 12, 0, 0),
                epoch_et=epoch_et,
                time_s=data['plot_time_s'],
                time_et=data.get('time'),
            )
        
        return cls(
            success=data.get('success', True),
            message=data.get('message', ''),
            time_grid=time_grid,
            state=data.get('state'),
            coe=data.get('coe'),
            mee=data.get('mee'),
            at_ephem_times=data.get('at_ephem_times'),
        )
    
    def get(self, key: str, default=None):
        """Dict-like access for backward compatibility."""
        return self.to_dict().get(key, default)
    
    def __getitem__(self, key: str):
        """Dict-like access for backward compatibility."""
        return self.to_dict()[key]


@dataclass
class PropagationConfig:
    """
    Configuration for orbit propagation.
    
    Attributes:
        time_o_dt: Start time (UTC datetime)
        time_f_dt: End time (UTC datetime)
        dt_output: Output time step [s]
        rtol: Relative tolerance for integrator
        atol: Absolute tolerance for integrator
        method: Integration method (e.g., 'DOP853', 'RK45')
    """
    time_o_dt: datetime
    time_f_dt: datetime
    dt_output: float = 60.0
    rtol: float = 1e-12
    atol: float = 1e-12
    method: str = 'DOP853'
    
    @property
    def duration_s(self) -> float:
        """Total propagation duration in seconds."""
        return (self.time_f_dt - self.time_o_dt).total_seconds()
