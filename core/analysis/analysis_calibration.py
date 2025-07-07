"""
Módulo de Calibración de Análisis
=================================
Gestiona la carga de calibraciones pregeneradas para el sistema de análisis integrado.
"""

import json
import os
import numpy as np
import logging
from datetime import datetime

class AnalysisCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de análisis.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('AnalysisCalibration')
        
        # Estado actual
        self.current_baseline = None
        self.current_operator_id = None
        self.is_calibrated = False
    
    def load_baseline(self, operator_id, operator_name=None):
        """
        Carga el baseline pregenerado para un operador.
        
        Args:
            operator_id: ID del operador
            operator_name: Nombre del operador (opcional)
            
        Returns:
            bool: True si se cargó correctamente
        """
        self.current_operator_id = operator_id
        
        # Buscar analysis_baseline.json
        baseline_path = os.path.join(self.baseline_dir, operator_id, "analysis_baseline.json")
        
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    self.current_baseline = json.load(f)
                
                self.is_calibrated = True
                self.logger.info(f"Baseline de análisis cargado para {operator_name or operator_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error cargando baseline: {e}")
                self.is_calibrated = False
                return False
        else:
            self.logger.warning(f"No existe baseline de análisis para {operator_id}")
            self.is_calibrated = False
            return False
    
    def get_current_baseline(self):
        """Retorna el baseline actual"""
        return self.current_baseline
    
    def get_baseline_date(self):
        """Retorna la fecha del baseline actual"""
        if self.current_baseline:
            return self.current_baseline['calibration_info'].get('created_at')
        return None
    
    def is_operator_calibrated(self, operator_id):
        """Verifica si un operador tiene calibración"""
        baseline_path = os.path.join(self.baseline_dir, operator_id, "analysis_baseline.json")
        return os.path.exists(baseline_path)
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos extraídos (llamado por MasterCalibrationManager).
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas extraídas
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de análisis para {operator_id}")
        
        try:
            # Generar estructura de baseline compatible
            baseline_data = {
                "operator_id": operator_id,
                "calibration_info": {
                    "created_at": datetime.now().isoformat(),
                    "photos_processed": photos_count,
                    "version": "2.0",
                    "source": "master_calibration"
                },
                "facial_metrics": self._generate_facial_metrics(data),
                "thresholds": self._generate_thresholds(data),
                "environment_conditions": self._analyze_environment(data),
                "quality_score": self._calculate_quality_score(data, photos_count)
            }
            
            # Guardar baseline
            return self._save_baseline(operator_id, baseline_data)
            
        except Exception as e:
            self.logger.error(f"Error generando calibración: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_facial_metrics(self, data):
        """Genera métricas faciales desde los datos extraídos"""
        metrics = {}
        
        # Métricas de ojos
        if 'ear_values' in data and data['ear_values']:
            ear_values = np.array(data['ear_values'])
            metrics['eye_measurements'] = {
                'eye_openness_avg': float(np.mean(ear_values)),
                'eye_openness_std': float(np.std(ear_values)),
                'eye_openness_min': float(np.min(ear_values)),
                'eye_openness_max': float(np.max(ear_values)),
                'blink_rate_per_minute': 15.0  # Valor estimado estándar
            }
        
        # Métricas de cejas
        if 'eyebrow_distances' in data and data['eyebrow_distances']:
            eyebrow_values = np.array(data['eyebrow_distances'])
            metrics['eyebrow_measurements'] = {
                'eyebrow_distance_avg': float(np.mean(eyebrow_values)),
                'eyebrow_distance_std': float(np.std(eyebrow_values))
            }
        
        # Métricas de boca
        if 'mar_values' in data and data['mar_values']:
            mar_values = np.array(data['mar_values'])
            metrics['mouth_measurements'] = {
                'mouth_ratio_avg': float(np.mean(mar_values)),
                'mouth_ratio_std': float(np.std(mar_values))
            }
        
        # Estructura facial
        if 'face_widths' in data and data['face_widths']:
            face_widths = np.array(data['face_widths'])
            metrics['face_structure'] = {
                'face_width_avg': float(np.mean(face_widths)),
                'face_width_std': float(np.std(face_widths)),
                'jaw_tension_avg': 0.5  # Valor por defecto
            }
        
        return metrics
    
    def _generate_thresholds(self, data):
        """Genera umbrales adaptativos desde los datos"""
        thresholds = {}
        
        # Umbrales de fatiga
        if 'ear_values' in data and data['ear_values']:
            ear_avg = np.mean(data['ear_values'])
            thresholds['fatigue'] = {
                'microsleep_threshold': float(ear_avg * 0.6),
                'severe_fatigue_threshold': float(ear_avg * 0.4),
                'blink_duration_threshold': 0.3
            }
        else:
            # Valores por defecto
            thresholds['fatigue'] = {
                'microsleep_threshold': 0.15,
                'severe_fatigue_threshold': 0.10,
                'blink_duration_threshold': 0.3
            }
        
        # Umbrales de estrés
        if 'eyebrow_distances' in data and data['eyebrow_distances']:
            eyebrow_avg = np.mean(data['eyebrow_distances'])
            thresholds['stress'] = {
                'high_tension_threshold': float(eyebrow_avg * 0.7),
                'movement_threshold': 0.05,
                'jaw_tension_threshold': 0.7
            }
        else:
            thresholds['stress'] = {
                'high_tension_threshold': 0.7,
                'movement_threshold': 0.05,
                'jaw_tension_threshold': 0.7
            }
        
        return thresholds
    
    def _analyze_environment(self, data):
        """Analiza condiciones ambientales desde las fotos"""
        conditions = {}
        
        if 'light_levels' in data and data['light_levels']:
            light_levels = np.array(data['light_levels'])
            conditions['lighting_average'] = float(np.mean(light_levels))
            conditions['lighting_variation'] = float(np.std(light_levels))
            conditions['predominantly_dark'] = float(np.mean(light_levels)) < 80
        
        return conditions
    
    def _calculate_quality_score(self, data, photos_count):
        """Calcula un score de calidad de la calibración"""
        score = 0.0
        
        # Más fotos = mejor calibración
        photo_score = min(1.0, photos_count / 4.0)
        score += photo_score * 0.3
        
        # Consistencia en las métricas
        if 'ear_values' in data and len(data['ear_values']) > 1:
            ear_std = np.std(data['ear_values'])
            ear_mean = np.mean(data['ear_values'])
            consistency = 1.0 - min(1.0, ear_std / ear_mean) if ear_mean > 0 else 0
            score += consistency * 0.4
        
        # Variedad de condiciones de luz
        if 'light_levels' in data and len(data['light_levels']) > 1:
            light_std = np.std(data['light_levels'])
            variety_score = min(1.0, light_std / 50.0)  # Normalizar a 50
            score += variety_score * 0.3
        
        return round(score, 2)
    
    def _save_baseline(self, operator_id, baseline_data):
        """Guarda el baseline en archivo JSON"""
        try:
            # Crear directorio
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Guardar archivo
            baseline_path = os.path.join(operator_dir, "analysis_baseline.json")
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Baseline de análisis guardado: {baseline_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando baseline: {e}")
            return False