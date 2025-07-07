"""
Face Recognition Module Package
==============================
Módulo de reconocimiento facial con calibración personalizada.
"""

from .face_recognition_module import FaceRecognitionModule
from .face_recognition_calibration import FaceRecognitionCalibration
from .face_recognition_dashboard import FaceRecognitionDashboard
from .integrated_face_system import IntegratedFaceSystem

__all__ = [
    'FaceRecognitionModule',
    'FaceRecognitionCalibration',
    'FaceRecognitionDashboard',
    'IntegratedFaceSystem'
]