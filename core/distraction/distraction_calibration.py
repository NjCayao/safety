"""
Módulo de Calibración de Distracciones
=====================================
Gestiona la calibración personalizada de detección de distracciones por operador.
"""

import json
import os
import numpy as np
from datetime import datetime
import logging

class DistractionCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de distracciones.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('DistractionCalibration')
        
        # Umbrales por defecto
        self.default_thresholds = {
            # Umbrales de rotación
            'rotation_threshold_day': 2.6,
            'rotation_threshold_night': 2.8,
            'extreme_rotation_threshold': 2.5,
            
            # Temporización de alertas (en segundos)
            'level1_time': 3,
            'level2_time': 7,
            
            # Sensibilidad y detección
            'visibility_threshold': 15,
            'frames_without_face_limit': 5,
            'confidence_threshold': 0.7,
            
            # Modo nocturno
            'night_mode_threshold': 50,
            'enable_night_mode': True,
            
            # Buffer y ventanas de tiempo
            'prediction_buffer_size': 10,
            'distraction_window': 600,
            'min_frames_for_reset': 10,
            
            # Control de audio
            'audio_enabled': True,
            'level1_volume': 0.8,
            'level2_volume': 1.0,
            
            # FPS de la cámara para cálculos
            'camera_fps': 4,
            
            # Confianza de calibración
            'calibration_confidence': 0.0
        }
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos extraídos.
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas de rotación de cabeza
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de distracciones para {operator_id}")
        
        try:
            # Copiar umbrales por defecto
            thresholds = self.default_thresholds.copy()
            
            # Ajustar umbrales basados en datos del operador
            if 'head_rotations' in data and data['head_rotations']:
                rotations = np.array(data['head_rotations'])
                
                # Estadísticas de rotación
                mean_rotation = np.mean(rotations)
                std_rotation = np.std(rotations)
                
                self.logger.info(f"Estadísticas de rotación del operador:")
                self.logger.info(f"  - Media: {mean_rotation:.3f}")
                self.logger.info(f"  - Desv. estándar: {std_rotation:.3f}")
                
                # Ajustar umbral basado en la variabilidad natural del operador
                if std_rotation > 0.3:
                    # Operador con mucho movimiento natural
                    thresholds['rotation_threshold_day'] = 2.8
                    thresholds['rotation_threshold_night'] = 3.0
                    self.logger.info("Ajustes para operador con movimiento natural alto")
                elif std_rotation < 0.1:
                    # Operador muy estático
                    thresholds['rotation_threshold_day'] = 2.4
                    thresholds['rotation_threshold_night'] = 2.6
                    self.logger.info("Ajustes para operador con poco movimiento natural")
            
            # Ajustar para condiciones de luz
            if 'light_levels' in data and data['light_levels']:
                light_levels = np.array(data['light_levels'])
                avg_light = np.mean(light_levels)
                
                if avg_light < 80:
                    thresholds['night_mode_threshold'] = 90
                    self.logger.info("Ajustes para condiciones de poca luz aplicados")
            
            # Ajustar sensibilidad según el tamaño facial
            if 'face_widths' in data and data['face_widths']:
                face_widths = np.array(data['face_widths'])
                avg_width = np.mean(face_widths)
                
                # Si las caras son pequeñas en las fotos, ajustar visibilidad
                if avg_width < 100:
                    thresholds['visibility_threshold'] = 10
                    self.logger.info("Ajuste de visibilidad para rostros pequeños")
            
            # Calcular confianza de calibración
            thresholds['calibration_confidence'] = min(1.0, photos_count / 4.0)
            
            # Estadísticas
            statistics = self._calculate_statistics(data)
            
            # Crear estructura de calibración
            calibration_data = {
                "operator_id": operator_id,
                "calibration_info": {
                    "created_at": datetime.now().isoformat(),
                    "photos_processed": photos_count,
                    "version": "1.0",
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
        """Calcula estadísticas de las métricas"""
        stats = {}
        
        # Estadísticas de rotación de cabeza
        if 'head_rotations' in data and data['head_rotations']:
            rotations = np.array(data['head_rotations'])
            stats['rotation_stats'] = {
                'mean': float(np.mean(rotations)),
                'std': float(np.std(rotations)),
                'min': float(np.min(rotations)),
                'max': float(np.max(rotations)),
                'percentiles': {
                    '25': float(np.percentile(rotations, 25)),
                    '50': float(np.percentile(rotations, 50)),
                    '75': float(np.percentile(rotations, 75))
                }
            }
        
        # Estadísticas de inclinación
        if 'head_tilts' in data and data['head_tilts']:
            tilts = np.array(data['head_tilts'])
            stats['tilt_stats'] = {
                'mean': float(np.mean(tilts)),
                'std': float(np.std(tilts))
            }
        
        # Estadísticas de tamaño facial
        if 'face_widths' in data and data['face_widths']:
            widths = np.array(data['face_widths'])
            stats['face_width_stats'] = {
                'mean': float(np.mean(widths)),
                'std': float(np.std(widths))
            }
        
        return stats
    
    def _save_calibration(self, operator_id, calibration_data):
        """Guarda la calibración en archivo JSON"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Convertir tipos numpy a tipos Python nativos
            def convert_numpy_types(obj):
                if isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            # Convertir toda la estructura
            calibration_data = convert_numpy_types(calibration_data)
            
            # Guardar archivo
            calibration_path = os.path.join(operator_dir, "distraction_baseline.json")
            with open(calibration_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Calibración guardada en {calibration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_calibration(self, operator_id):
        """Carga calibración existente de un operador"""
        try:
            calibration_path = os.path.join(
                self.baseline_dir,
                operator_id,
                "distraction_baseline.json"
            )
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No existe calibración de distracciones para {operator_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None
    
    def get_thresholds(self, operator_id):
        """Obtiene umbrales para un operador"""
        calibration = self.load_calibration(operator_id)
        
        if calibration and 'thresholds' in calibration:
            self.logger.info(f"Usando calibración personalizada de distracciones para {operator_id}")
            thresholds = calibration['thresholds']
            
            # Asegurar valores correctos para level1 y level2
            thresholds['level1_time'] = 3
            thresholds['level2_time'] = 7
            
            return thresholds
        else:
            self.logger.info(f"Usando valores por defecto de distracciones para {operator_id}")
            return self.default_thresholds.copy()