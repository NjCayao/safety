"""
Módulo de Calibración de Reconocimiento Facial (SIMPLIFICADO)
=============================================================
Gestiona solo los umbrales de reconocimiento por operador.
"""

import json
import os
import numpy as np
from datetime import datetime
import logging

class FaceRecognitionCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de reconocimiento facial.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('FaceRecognitionCalibration')
        
        # Umbrales por defecto (SOLO RECONOCIMIENTO)
        self.default_thresholds = {
            # Umbrales de reconocimiento
            'face_tolerance': 0.6,
            'min_confidence': 0.4,
            'recognition_threshold': 0.5,
            
            # Modo nocturno
            'night_mode_threshold': 50,
            'night_tolerance_adjustment': 0.1,
            'enable_night_mode': True,
            
            # Confianza de calibración
            'calibration_confidence': 0.0
        }
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos extraídos (SOLO RECONOCIMIENTO).
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas de reconocimiento
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de reconocimiento para {operator_id}")
        
        try:
            # Copiar umbrales por defecto
            thresholds = self.default_thresholds.copy()
            
            # Ajustar tolerancia basada en variabilidad facial
            if 'face_encodings_std' in data and data['face_encodings_std']:
                avg_std = np.mean(data['face_encodings_std'])
                
                # Si hay mucha variabilidad, aumentar tolerancia
                if avg_std > 0.1:
                    thresholds['face_tolerance'] = min(0.8, 0.6 + avg_std)
                    thresholds['recognition_threshold'] = min(0.7, 0.5 + avg_std)
                elif avg_std < 0.05:
                    # Si hay poca variabilidad, podemos ser más estrictos
                    thresholds['face_tolerance'] = 0.5
                    thresholds['recognition_threshold'] = 0.4
            
            # Ajustar según condiciones de iluminación típicas
            if 'light_levels' in data and data['light_levels']:
                light_levels = np.array(data['light_levels'])
                avg_light = np.mean(light_levels)
                
                # Si las fotos son principalmente oscuras
                if avg_light < 80:
                    thresholds['night_mode_threshold'] = 90
                    thresholds['night_tolerance_adjustment'] = 0.15
            
            # Calcular confianza de calibración
            thresholds['calibration_confidence'] = min(1.0, photos_count / 4.0)
            
            # Estadísticas básicas
            statistics = self._calculate_statistics(data)
            
            # Crear estructura de calibración
            calibration_data = {
                "operator_id": operator_id,
                "calibration_info": {
                    "created_at": datetime.now().isoformat(),
                    "photos_processed": photos_count,
                    "version": "2.0",
                    "source": "master_calibration"
                },
                "thresholds": thresholds,
                "statistics": statistics
            }
            
            # Guardar calibración
            return self._save_calibration(operator_id, calibration_data)
            
        except Exception as e:
            self.logger.error(f"Error generando calibración: {e}")
            return False
    
    def _calculate_statistics(self, data):
        """Calcula estadísticas básicas de reconocimiento"""
        stats = {}
        
        # Estadísticas de variabilidad facial
        if 'face_encodings_std' in data:
            stats['encoding_variability'] = {
                'mean_std': float(np.mean(data['face_encodings_std'])),
                'max_std': float(np.max(data['face_encodings_std'])),
                'consistency_score': float(1.0 - np.mean(data['face_encodings_std']))
            }
        
        # Estadísticas de iluminación
        if 'light_levels' in data and data['light_levels']:
            light_levels = np.array(data['light_levels'])
            stats['lighting_stats'] = {
                'mean': float(np.mean(light_levels)),
                'std': float(np.std(light_levels)),
                'predominantly_dark': float(np.mean(light_levels)) < 80
            }
        
        # Estadísticas de tamaño facial
        if 'face_areas' in data and data['face_areas']:
            face_areas = np.array(data['face_areas'])
            stats['face_size_stats'] = {
                'mean': float(np.mean(face_areas)),
                'std': float(np.std(face_areas))
            }
        
        return stats
    
    def _save_calibration(self, operator_id, calibration_data):
        """Guarda la calibración en archivo JSON"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Guardar archivo
            calibration_path = os.path.join(operator_dir, "face_recognition_baseline.json")
            with open(calibration_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Calibración guardada en {calibration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración: {e}")
            return False
    
    def load_calibration(self, operator_id):
        """Carga calibración existente de un operador"""
        try:
            calibration_path = os.path.join(
                self.baseline_dir, 
                operator_id, 
                "face_recognition_baseline.json"
            )
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No existe calibración para {operator_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None
    
    def get_thresholds(self, operator_id):
        """Obtiene umbrales para un operador"""
        calibration = self.load_calibration(operator_id)
        
        if calibration and 'thresholds' in calibration:
            self.logger.info(f"Usando calibración personalizada para {operator_id}")
            return calibration['thresholds']
        else:
            self.logger.info(f"Usando valores por defecto para {operator_id}")
            return self.default_thresholds.copy()