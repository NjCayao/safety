"""
Yawn Detection Module Package
============================
Módulo de detección de bostezos con calibración personalizada.
"""

from .yawn_detection import YawnDetector
from .yawn_calibration import YawnCalibration
from .yawn_dashboard import YawnDashboard
from .integrated_yawn_system import IntegratedYawnSystem

__all__ = [
    'YawnDetector',
    'YawnCalibration', 
    'YawnDashboard',
    'IntegratedYawnSystem'
]