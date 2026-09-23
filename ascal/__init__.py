"""ascal: geometric calibration of all-sky (fisheye) cameras from star positions."""
__version__ = "0.3.0"

from .bootstrap import CalibrationResult, Site, calibrate, evaluate  # noqa: F401
from .catalog import load_catalog, sky_stars  # noqa: F401
from .detect import Frame, load_frame  # noqa: F401
from .model import CameraModel, band_statistics, fit, residuals, robust_fit  # noqa: F401
