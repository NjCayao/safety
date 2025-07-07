# -*- coding: utf-8 -*-
"""
Módulo de Análisis Avanzado
===========================
Sistema integrado de análisis facial para detección de fatiga, estrés y anomalías.
"""

# Importar en orden correcto para evitar dependencias circulares
from .calibration_manager import CalibrationManager
from .fatigue_detector import FatigueDetector
from .stress_analyzer import StressAnalyzer
from .pulse_estimator import PulseEstimator
from .emotion_analyzer import EmotionAnalyzer
from .anomaly_detector import AnomalyDetector
from .analysis_dashboard import AnalysisDashboard

# Importar al final porque depende de los anteriores
from .integrated_analysis_system import IntegratedAnalysisSystem

__all__ = [
    'CalibrationManager',
    'FatigueDetector',
    'StressAnalyzer', 
    'PulseEstimator',
    'EmotionAnalyzer',
    'AnomalyDetector',
    'AnalysisDashboard',
    'IntegratedAnalysisSystem'
]

__version__ = '1.0.0'