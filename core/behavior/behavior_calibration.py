"""
Módulo de Calibración de Comportamientos
========================================
Gestiona la calibración personalizada de detección de comportamientos por operador.
"""

import json
import os
import numpy as np
from datetime import datetime
import logging

class BehaviorCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de comportamientos.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('BehaviorCalibration')
        
        # Umbrales por defecto
        self.default_thresholds = {
            # Umbrales de confianza
            'confidence_threshold': 0.4,
            'night_confidence_threshold': 0.35,
            
            # Umbrales de tiempo para teléfono
            'phone_alert_threshold_1': 3,  # Primera alerta
            'phone_alert_threshold_2': 7,  # Segunda alerta
            
            # Umbrales para cigarrillo
            'cigarette_pattern_window': 30,
            'cigarette_pattern_threshold': 3,
            'cigarette_continuous_threshold': 7,
            
            # Configuración nocturna
            'night_mode_threshold': 50,
            'enable_night_mode': True,
            
            # Factor de proximidad facial
            'face_proximity_factor': 2,
            
            # Configuración de optimización
            'enable_optimization': True,
            'processing_interval': 2,
            'roi_enabled': True,
            'roi_scale_factor': 0.6,
            
            # Confianza de calibración
            'calibration_confidence': 0.0
        }
        
        # Métricas a calibrar desde fotos
        self.metrics_to_calibrate = {
            'face_sizes': [],
            'hand_positions': [],
            'lighting_conditions': [],
            'pose_variations': [],
            'object_detection_confidence': []
        }
        
        # Parámetros de comportamiento específicos del operador
        self.behavior_patterns = {
            'phone_usage_tendency': 0.5,  # 0-1: tendencia a usar teléfono
            'smoking_tendency': 0.5,       # 0-1: tendencia a fumar
            'movement_patterns': []        # Patrones de movimiento típicos
        }
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos ya extraídos por el MasterCalibrationManager.
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas específicas de comportamiento
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de comportamientos para {operator_id}")
        
        try:
            # Copiar umbrales por defecto
            thresholds = self.default_thresholds.copy()
            
            # Ajustar umbrales basados en datos extraídos
            if 'face_areas' in data and data['face_areas']:
                face_areas = np.array(data['face_areas'])
                
                # Ajustar factor de proximidad basado en tamaño promedio de rostro
                avg_face_area = np.mean(face_areas)
                if avg_face_area > 20000:  # Rostro grande/cerca
                    thresholds['face_proximity_factor'] = 1.5
                elif avg_face_area < 10000:  # Rostro pequeño/lejos
                    thresholds['face_proximity_factor'] = 2.5
                
                # Ajustar confianza según variabilidad
                face_std = np.std(face_areas)
                if face_std / avg_face_area < 0.2:  # Poca variabilidad
                    thresholds['confidence_threshold'] = 0.35
            
            # Ajustar según condiciones de iluminación
            if 'light_levels' in data and data['light_levels']:
                light_levels = np.array(data['light_levels'])
                avg_light = np.mean(light_levels)
                
                # Si las fotos son principalmente oscuras, ajustar umbrales
                if avg_light < 80:
                    thresholds['night_confidence_threshold'] = 0.3
                    thresholds['night_mode_threshold'] = 60
            
            # Calcular confianza de calibración
            thresholds['calibration_confidence'] = min(1.0, photos_count / 4.0)
            
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
                "statistics": self._calculate_statistics(data)
            }
            
            # Guardar calibración
            return self._save_calibration(operator_id, calibration_data)
            
        except Exception as e:
            self.logger.error(f"Error generando calibración de comportamientos: {e}")
            import traceback
            traceback.print_exc()
    def analyze_behavior_tendencies(self, operator_id):
        """
        Analiza las tendencias de comportamiento del operador basándose
        en datos históricos si están disponibles.
        
        Args:
            operator_id: ID del operador
            
        Returns:
            dict: Tendencias detectadas
        """
        try:
            # Buscar reportes históricos del operador
            reports_dir = f"reports/behavior/{operator_id}"
            tendencies = {
                'phone_usage_frequency': 0,
                'average_phone_duration': 0,
                'smoking_frequency': 0,
                'peak_activity_hours': [],
                'risk_level': 'low'  # low, medium, high
            }
            
            # Aquí se podría analizar reportes históricos
            # Por ahora retornamos valores base
            return tendencies
            
        except Exception as e:
            self.logger.error(f"Error analizando tendencias: {e}")
            return None
    
    def get_operator_profile(self, operator_id):
        """
        Obtiene el perfil completo de comportamiento del operador.
        
        Args:
            operator_id: ID del operador
            
        Returns:
            dict: Perfil completo con calibración y tendencias
        """
        calibration = self.load_calibration(operator_id)
        tendencies = self.analyze_behavior_tendencies(operator_id)
        
        profile = {
            'operator_id': operator_id,
            'has_calibration': calibration is not None,
            'calibration_confidence': calibration['thresholds']['calibration_confidence'] if calibration else 0,
            'thresholds': calibration['thresholds'] if calibration else self.default_thresholds,
            'statistics': calibration.get('statistics', {}) if calibration else {},
            'tendencies': tendencies or {},
            'last_updated': calibration['calibration_info']['created_at'] if calibration else None
        }
        
        return profile
    
    def _calculate_statistics(self, data):
        """Calcula estadísticas de las métricas"""
        stats = {}
        
        # Estadísticas de áreas faciales
        if 'face_areas' in data and data['face_areas']:
            face_areas = np.array(data['face_areas'])
            stats['face_area_stats'] = {
                'mean': float(np.mean(face_areas)),
                'std': float(np.std(face_areas)),
                'min': float(np.min(face_areas)),
                'max': float(np.max(face_areas))
            }
        
        # Estadísticas de iluminación
        if 'light_levels' in data and data['light_levels']:
            light_levels = np.array(data['light_levels'])
            stats['lighting_stats'] = {
                'mean': float(np.mean(light_levels)),
                'std': float(np.std(light_levels)),
                'predominantly_dark': float(np.mean(light_levels)) < 80
            }
        
        # Estadísticas de posición nariz-boca (para detectar objetos cerca)
        if 'nose_to_mouth' in data and data['nose_to_mouth']:
            distances = np.array(data['nose_to_mouth'])
            stats['face_geometry'] = {
                'nose_mouth_distance_mean': float(np.mean(distances)),
                'nose_mouth_distance_std': float(np.std(distances))
            }
        
        return stats
    
    def _save_calibration(self, operator_id, calibration_data):
        """Guarda la calibración en archivo JSON"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Guardar archivo
            calibration_path = os.path.join(operator_dir, "behavior_baseline.json")
            with open(calibration_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Calibración de comportamientos guardada en {calibration_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración: {e}")
            return False
    
    def load_calibration(self, operator_id):
        """Carga calibración existente de un operador"""
        try:
            calibration_path = os.path.join(self.baseline_dir, operator_id, "behavior_baseline.json")
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No existe calibración de comportamientos para {operator_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None
    
    def get_thresholds(self, operator_id):
        """Obtiene umbrales para un operador (calibrados o default)"""
        calibration = self.load_calibration(operator_id)
        
        if calibration and 'thresholds' in calibration:
            self.logger.info(f"Usando umbrales calibrados de comportamiento para {operator_id}")
            return calibration['thresholds']
        else:
            self.logger.info(f"Usando umbrales por defecto de comportamiento para {operator_id}")
            return self.default_thresholds.copy()
    
    def update_thresholds(self, operator_id, new_thresholds):
        """
        Actualiza umbrales específicos para un operador.
        
        Args:
            operator_id: ID del operador
            new_thresholds: Dict con nuevos valores de umbrales
            
        Returns:
            bool: True si se actualizó correctamente
        """
        try:
            # Cargar calibración existente o crear nueva
            calibration = self.load_calibration(operator_id)
            
            if not calibration:
                # Crear nueva calibración
                calibration = {
                    "operator_id": operator_id,
                    "calibration_info": {
                        "created_at": datetime.now().isoformat(),
                        "photos_processed": 0,
                        "version": "1.0",
                        "source": "manual_update"
                    },
                    "thresholds": self.default_thresholds.copy(),
                    "statistics": {}
                }
            
            # Actualizar umbrales
            calibration['thresholds'].update(new_thresholds)
            calibration['calibration_info']['last_updated'] = datetime.now().isoformat()
            
            # Guardar
            return self._save_calibration(operator_id, calibration)
            
        except Exception as e:
            self.logger.error(f"Error actualizando umbrales: {e}")
            return False