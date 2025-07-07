"""
Módulo de Calibración de Fatiga
===============================
Gestiona la calibración personalizada de detección de fatiga por operador.
"""

import json
import os
import numpy as np
from datetime import datetime
import logging
import cv2
import dlib
from scipy.spatial import distance

class FatigueCalibration:
    def __init__(self, baseline_dir="operators/baseline-json"):
        """
        Inicializa el gestor de calibración de fatiga.
        
        Args:
            baseline_dir: Directorio donde se guardan los baselines
        """
        self.baseline_dir = baseline_dir
        self.logger = logging.getLogger('FatigueCalibration')
        
        # Configuración de detección
        self.landmark_predictor = None
        self.face_detector = None
        
        # Métricas a calibrar
        self.metrics_to_calibrate = {
            'ear_values': [],           # Eye Aspect Ratio values
            'blink_durations': [],      # Duraciones de parpadeos
            'blink_frequencies': [],    # Frecuencias de parpadeo
            'eye_closure_patterns': [], # Patrones de cierre de ojos
            'lighting_conditions': []   # Condiciones de iluminación
        }
        
        # Umbrales por defecto (si no hay calibración)
        self.default_thresholds = {
            'ear_threshold': 0.20,           # FIJO: conservador para todos
            'ear_night_adjustment': 0.02,    # Ajuste mínimo nocturno
            'microsleep_threshold': 1.5,     # FIJO: 1.5 segundos
            'blink_rate_normal': 15,         # parpadeos por minuto
            'calibration_confidence': 0.0,
            'frames_to_confirm': 4           # FIJO: 4 frames para todos
        }
        
    def initialize_detectors(self, model_path):
        """Inicializa los detectores necesarios para calibración"""
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(model_path)
            self.logger.info("Detectores inicializados para calibración")
            return True
        except Exception as e:
            self.logger.error(f"Error inicializando detectores: {e}")
            return False
    
    def calibrate_from_photos(self, operator_id, photos_path):
        """
        Genera calibración de fatiga desde las fotos del operador.
        
        Args:
            operator_id: DNI del operador
            photos_path: Ruta a las fotos del operador
            
        Returns:
            dict: Calibración generada o None si falla
        """
        self.logger.info(f"Iniciando calibración de fatiga para operador {operator_id}")
        
        # Resetear métricas
        for key in self.metrics_to_calibrate:
            self.metrics_to_calibrate[key] = []
        
        # Procesar cada foto
        photo_count = 0
        for filename in os.listdir(photos_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                photo_path = os.path.join(photos_path, filename)
                
                try:
                    # Cargar imagen
                    image = cv2.imread(photo_path)
                    if image is None:
                        continue
                    
                    # Extraer métricas
                    metrics = self._extract_fatigue_metrics(image)
                    if metrics:
                        self._add_metrics(metrics)
                        photo_count += 1
                        self.logger.debug(f"Procesada foto {filename}")
                    
                except Exception as e:
                    self.logger.warning(f"Error procesando {filename}: {e}")
                    continue
        
        if photo_count == 0:
            self.logger.error("No se pudieron procesar fotos para calibración")
            return None
        
        # Calcular umbrales personalizados
        thresholds = self._calculate_personalized_thresholds()
        
        # Crear estructura de calibración
        calibration_data = {
            "operator_id": operator_id,
            "calibration_info": {
                "created_at": datetime.now().isoformat(),
                "photos_processed": photo_count,
                "version": "1.0"
            },
            "thresholds": thresholds,
            "statistics": self._calculate_statistics(),
            "metadata": {
                "lighting_average": np.mean(self.metrics_to_calibrate['lighting_conditions']) if self.metrics_to_calibrate['lighting_conditions'] else 128
            }
        }
        
        # Guardar calibración
        if self._save_calibration(operator_id, calibration_data):
            self.logger.info(f"Calibración de fatiga completada para {operator_id}")
            return calibration_data
        else:
            return None
    
    def _extract_fatigue_metrics(self, image):
        """Extrae métricas de fatiga de una imagen"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = self.face_detector(gray, 0)
            if not faces:
                return None
            
            # Usar el primer rostro
            face = faces[0]
            landmarks = self.landmark_predictor(gray, face)
            
            # Extraer puntos de los ojos
            left_eye = self._get_eye_points(landmarks, 36, 42)
            right_eye = self._get_eye_points(landmarks, 42, 48)
            
            # Calcular EAR
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Nivel de iluminación
            light_level = np.mean(gray)
            
            # Crear diccionario de métricas
            metrics = {
                'ear': ear,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'light_level': light_level,
                'face_area': (face.right() - face.left()) * (face.bottom() - face.top())
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extrayendo métricas: {e}")
            return None
    
    def _add_metrics(self, metrics):
        """Agrega métricas al buffer de calibración"""
        self.metrics_to_calibrate['ear_values'].append(metrics['ear'])
        self.metrics_to_calibrate['lighting_conditions'].append(metrics['light_level'])
        
        # Simular detección de parpadeos (para calibración inicial)
        if metrics['ear'] < 0.2:  # Posible ojo cerrado en foto
            self.metrics_to_calibrate['blink_durations'].append(0.15)  # Duración típica
    
    # def _calculate_personalized_thresholds(self):
    #     """Calcula umbrales personalizados basados en las métricas"""
    #     thresholds = self.default_thresholds.copy()
        
    #     if self.metrics_to_calibrate['ear_values']:
    #         ear_values = np.array(self.metrics_to_calibrate['ear_values'])
            
            
    #         thresholds['ear_threshold'] = 0.20             
           
    #         self.logger.info(f"EAR promedio en fotos: {np.mean(ear_values):.3f}")
    #         self.logger.info(f"Usando umbral EAR FIJO: 0.20 (conservador)")
            
    #         thresholds['microsleep_threshold'] = 1.5  # FIJO: 1.5 segundos para microsueños
            
            
    #         thresholds['ear_night_adjustment'] = 0.02  # Pequeño ajuste para modo nocturno
            
    #         # Frames para confirmar - más frames = menos falsos positivos
    #         thresholds['frames_to_confirm'] = 5  # FIJO: 5 frames para todos
            
    #         # Confianza de calibración
    #         thresholds['calibration_confidence'] = min(1.0, len(ear_values) / 4.0)
            
    #         # Log final
    #         self.logger.info(f"Calibración estándar aplicada:")
    #         self.logger.info(f"  - Umbral EAR: 0.20 (fijo)")
    #         self.logger.info(f"  - Tiempo microsueño: 1.5s (fijo)")
    #         self.logger.info(f"  - Frames confirmación: 5 (fijo)")
        
    #     return thresholds

    def _calculate_personalized_thresholds(self):
        """Calcula umbrales personalizados basados en las métricas"""
        thresholds = self.default_thresholds.copy()
        
        # VALORES FIJOS PARA TODOS - No importa las fotos
        thresholds['ear_threshold'] = 0.20
        thresholds['microsleep_threshold'] = 1.5
        thresholds['frames_to_confirm'] = 4
        thresholds['ear_night_adjustment'] = 0.02
        thresholds['calibration_confidence'] = 1.0
        
        # Solo para log informativo
        if self.metrics_to_calibrate['ear_values']:
            ear_values = np.array(self.metrics_to_calibrate['ear_values'])
            self.logger.info(f"EAR promedio en fotos: {np.mean(ear_values):.3f} (ignorado)")
        
        self.logger.info("Aplicando valores FIJOS estándar:")
        self.logger.info("  - Umbral EAR: 0.20")
        self.logger.info("  - Tiempo microsueño: 1.5s")
        self.logger.info("  - Frames confirmación: 4")
        
        return thresholds
    
    def _calculate_statistics(self):
        """Calcula estadísticas de las métricas"""
        stats = {}
        
        if self.metrics_to_calibrate['ear_values']:
            ear_values = np.array(self.metrics_to_calibrate['ear_values'])
            stats['ear_stats'] = {
                'mean': float(np.mean(ear_values)),
                'std': float(np.std(ear_values)),
                'min': float(np.min(ear_values)),
                'max': float(np.max(ear_values)),
                'percentiles': {
                    '25': float(np.percentile(ear_values, 25)),
                    '50': float(np.percentile(ear_values, 50)),
                    '75': float(np.percentile(ear_values, 75))
                }
            }
        
        if self.metrics_to_calibrate['lighting_conditions']:
            light_values = np.array(self.metrics_to_calibrate['lighting_conditions'])
            stats['lighting_stats'] = {
                'mean': float(np.mean(light_values)),
                'std': float(np.std(light_values))
            }
        
        return stats
    
    def _save_calibration(self, operator_id, calibration_data):
        """Guarda la calibración en archivo JSON"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Guardar archivo
            calibration_path = os.path.join(operator_dir, "fatigue_baseline.json")
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
            calibration_path = os.path.join(self.baseline_dir, operator_id, "fatigue_baseline.json")
            
            if os.path.exists(calibration_path):
                with open(calibration_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"No existe calibración para operador {operator_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error cargando calibración: {e}")
            return None
    
    def get_thresholds(self, operator_id):
        """Obtiene umbrales para un operador (calibrados o default)"""
        calibration = self.load_calibration(operator_id)
        
        if calibration and 'thresholds' in calibration:
            self.logger.info(f"Usando umbrales calibrados para {operator_id}")
            return calibration['thresholds']
        else:
            self.logger.info(f"Usando umbrales por defecto para {operator_id}")
            return self.default_thresholds.copy()
    
    # Métodos auxiliares
    def _get_eye_points(self, landmarks, start, end):
        """Obtiene puntos del ojo desde landmarks"""
        return [(landmarks.part(i).x, landmarks.part(i).y) for i in range(start, end)]
    
    def _calculate_ear(self, eye):
        """Calcula Eye Aspect Ratio"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C) if C > 0 else 0
    
    def calibrate_from_extracted_data(self, operator_id, data, photos_count):
        """
        Genera calibración desde datos ya extraídos por el MasterCalibrationManager.
        
        Args:
            operator_id: ID del operador
            data: Diccionario con métricas específicas de fatiga
            photos_count: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Generando calibración de fatiga desde datos extraídos para {operator_id}")
        
        try:
            # Usar los datos ya extraídos
            ear_values = np.array(data.get('ear_values', []))
            light_levels = np.array(data.get('light_levels', []))
            
            if len(ear_values) == 0:
                self.logger.error("No hay valores EAR para calibrar")
                return False
            
            # Calcular umbrales personalizados
            thresholds = {
                'ear_threshold': float(np.percentile(ear_values, 20) + (np.percentile(ear_values, 80) - np.percentile(ear_values, 20)) * 0.3),
                'ear_night_adjustment': float(min(0.05, np.std(ear_values) * 0.5)),
                'microsleep_threshold': 1.5,
                'blink_rate_normal': 15,
                'calibration_confidence': min(1.0, photos_count / 4.0)
            }
            
            # Ajustar umbral si los valores son muy bajos o altos
            if thresholds['ear_threshold'] < 0.15:
                thresholds['ear_threshold'] = 0.15
            elif thresholds['ear_threshold'] > 0.35:
                thresholds['ear_threshold'] = 0.35
                
            # Calcular estadísticas
            statistics = {
                'ear_stats': {
                    'mean': float(np.mean(ear_values)),
                    'std': float(np.std(ear_values)),
                    'min': float(np.min(ear_values)),
                    'max': float(np.max(ear_values)),
                    'percentiles': {
                        '25': float(np.percentile(ear_values, 25)),
                        '50': float(np.percentile(ear_values, 50)),
                        '75': float(np.percentile(ear_values, 75))
                    }
                }
            }
            
            if len(light_levels) > 0:
                statistics['lighting_stats'] = {
                    'mean': float(np.mean(light_levels)),
                    'std': float(np.std(light_levels))
                }
            
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
            self.logger.error(f"Error generando calibración de fatiga: {e}")
            import traceback
            traceback.print_exc()
            return False