"""
Módulo de Gestión de Calibración
================================
Maneja la calibración y baselines personalizados por operador.
"""

import json
import os
import time
import numpy as np
from datetime import datetime
from collections import deque
import logging

class CalibrationManager:
    def __init__(self, operators_dir="operators"):
        """
        Inicializa el gestor de calibración.
        
        Args:
            operators_dir: Directorio donde se guardan los datos de operadores
        """
        self.operators_dir = operators_dir
        self.logger = logging.getLogger('CalibrationManager')
        
        # Estado de calibración
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_duration = 30  # segundos
        self.calibration_samples = deque()
        
        # Operador actual
        self.current_operator_id = None
        self.current_operator_name = None
        self.current_baseline = None
        
        # Métricas a calibrar
        self.metrics_buffer = {
            'eye_openness': deque(maxlen=1000),
            'eyebrow_distance': deque(maxlen=1000),
            'mouth_ratio': deque(maxlen=1000),
            'face_width': deque(maxlen=1000),
            'jaw_tension': deque(maxlen=1000),
            'blink_count': 0,
            'samples_count': 0
        }
        
    def load_or_create_baseline(self, operator_id, operator_name):
        """
        Carga el baseline existente o inicia calibración para nuevo operador.
        
        Args:
            operator_id: ID/DNI del operador
            operator_name: Nombre del operador
            
        Returns:
            bool: True si existe baseline, False si necesita calibración
        """
        self.current_operator_id = operator_id
        self.current_operator_name = operator_name
        
        # Crear directorio del operador si no existe
        operator_path = os.path.join(self.operators_dir, operator_id)
        os.makedirs(operator_path, exist_ok=True)
        
        # Buscar baseline existente
        baseline_path = os.path.join(operator_path, "baseline.json")
        
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    self.current_baseline = json.load(f)
                
                # Verificar antigüedad
                last_calibration = self.current_baseline['calibration_info']['last_calibration']
                last_date = datetime.fromisoformat(last_calibration)
                days_old = (datetime.now() - last_date).days
                
                if days_old > 30:
                    self.logger.warning(f"Baseline antiguo ({days_old} días) para {operator_name}")
                
                self.logger.info(f"Baseline cargado para {operator_name} ({operator_id})")
                return True
                
            except Exception as e:
                self.logger.error(f"Error cargando baseline: {e}")
                
        # No existe baseline, iniciar calibración
        self.start_calibration()
        return False
    
    def start_calibration(self):
        """Inicia el proceso de calibración"""
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.metrics_buffer = {
            'eye_openness': deque(maxlen=1000),
            'eyebrow_distance': deque(maxlen=1000),
            'mouth_ratio': deque(maxlen=1000),
            'face_width': deque(maxlen=1000),
            'jaw_tension': deque(maxlen=1000),
            'blink_count': 0,
            'samples_count': 0
        }
        self.logger.info(f"Calibración iniciada para {self.current_operator_name}")
    
    def add_calibration_sample(self, face_landmarks):
        """
        Agrega una muestra de calibración.
        
        Args:
            face_landmarks: Landmarks faciales detectados
        """
        if not self.is_calibrating:
            return
        
        # Verificar si terminó el tiempo de calibración
        elapsed = time.time() - self.calibration_start_time
        if elapsed >= self.calibration_duration:
            self._finish_calibration()
            return
        
        # Extraer métricas
        metrics = self._extract_metrics(face_landmarks)
        if metrics:
            # Agregar a buffers
            self.metrics_buffer['eye_openness'].append(metrics['eye_openness'])
            self.metrics_buffer['eyebrow_distance'].append(metrics['eyebrow_distance'])
            self.metrics_buffer['mouth_ratio'].append(metrics['mouth_ratio'])
            self.metrics_buffer['face_width'].append(metrics['face_width'])
            self.metrics_buffer['jaw_tension'].append(metrics['jaw_tension'])
            
            # Detectar parpadeos
            if metrics['eye_openness'] < 0.15:
                self.metrics_buffer['blink_count'] += 1
            
            self.metrics_buffer['samples_count'] += 1
    
    def _extract_metrics(self, landmarks):
        """Extrae métricas faciales de los landmarks"""
        try:
            metrics = {}
            
            # Apertura de ojos
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_ear = self._calculate_ear(landmarks['left_eye'])
                right_ear = self._calculate_ear(landmarks['right_eye'])
                metrics['eye_openness'] = (left_ear + right_ear) / 2
            
            # Distancia entre cejas (normalizada)
            if 'left_eyebrow' in landmarks and 'right_eyebrow' in landmarks:
                left_brow = landmarks['left_eyebrow']
                right_brow = landmarks['right_eyebrow']
                brow_dist = self._distance(left_brow[-1], right_brow[0])
                
                # Normalizar por ancho de cara
                face_width = self._get_face_width(landmarks)
                metrics['face_width'] = face_width
                metrics['eyebrow_distance'] = brow_dist / face_width if face_width > 0 else 0
            
            # Ratio de boca
            if 'top_lip' in landmarks and 'bottom_lip' in landmarks:
                mouth_width = self._distance(landmarks['top_lip'][0], landmarks['top_lip'][-1])
                mouth_height = self._distance(landmarks['top_lip'][3], landmarks['bottom_lip'][3])
                metrics['mouth_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Tensión mandibular
            if 'chin' in landmarks:
                jaw_points = landmarks['chin']
                center = jaw_points[8] if len(jaw_points) > 8 else jaw_points[len(jaw_points)//2]
                metrics['jaw_tension'] = self._calculate_jaw_symmetry(jaw_points, center)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error extrayendo métricas: {e}")
            return None
    
    def _finish_calibration(self):
        """Finaliza la calibración y guarda el baseline"""
        self.is_calibrating = False
        
        if self.metrics_buffer['samples_count'] < 100:
            self.logger.warning("Calibración con pocas muestras")
            return
        
        # Calcular promedios y estadísticas
        baseline_data = {
            "operator_id": self.current_operator_id,
            "operator_name": self.current_operator_name,
            "calibration_info": {
                "first_calibration": datetime.now().isoformat(),
                "last_calibration": datetime.now().isoformat(),
                "total_calibrations": 1,
                "samples_collected": self.metrics_buffer['samples_count']
            },
            "facial_metrics": {
                "eye_measurements": {
                    "eye_openness_avg": float(np.mean(self.metrics_buffer['eye_openness'])),
                    "eye_openness_std": float(np.std(self.metrics_buffer['eye_openness'])),
                    "blink_rate_per_minute": (self.metrics_buffer['blink_count'] / self.calibration_duration) * 60
                },
                "eyebrow_measurements": {
                    "eyebrow_distance_avg": float(np.mean(self.metrics_buffer['eyebrow_distance'])),
                    "eyebrow_distance_std": float(np.std(self.metrics_buffer['eyebrow_distance']))
                },
                "mouth_measurements": {
                    "mouth_ratio_avg": float(np.mean(self.metrics_buffer['mouth_ratio'])),
                    "mouth_ratio_std": float(np.std(self.metrics_buffer['mouth_ratio']))
                },
                "face_structure": {
                    "face_width_avg": float(np.mean(self.metrics_buffer['face_width'])),
                    "jaw_tension_avg": float(np.mean(self.metrics_buffer['jaw_tension']))
                }
            },
            "thresholds": self._calculate_thresholds()
        }
        
        # Guardar baseline
        baseline_path = os.path.join(self.operators_dir, self.current_operator_id, "baseline.json")
        try:
            with open(baseline_path, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            
            self.current_baseline = baseline_data
            self.logger.info(f"Baseline guardado para {self.current_operator_name}")
            
        except Exception as e:
            self.logger.error(f"Error guardando baseline: {e}")
    
    def _calculate_thresholds(self):
        """Calcula umbrales adaptativos basados en las métricas"""
        eye_openness_avg = np.mean(self.metrics_buffer['eye_openness'])
        eyebrow_dist_avg = np.mean(self.metrics_buffer['eyebrow_distance'])
        
        return {
            "fatigue": {
                "microsleep_threshold": float(eye_openness_avg * 0.6),
                "severe_fatigue_threshold": float(eye_openness_avg * 0.4),
                "blink_duration_threshold": 0.3  # segundos
            },
            "stress": {
                "high_tension_threshold": float(eyebrow_dist_avg * 0.7),
                "movement_threshold": 0.05,
                "jaw_tension_threshold": 0.7
            }
        }
    
    def get_calibration_progress(self):
        """Obtiene el progreso de calibración en porcentaje"""
        if not self.is_calibrating:
            return 100
        
        elapsed = time.time() - self.calibration_start_time
        progress = min(100, int((elapsed / self.calibration_duration) * 100))
        return progress
    
    def get_current_baseline(self):
        """Retorna el baseline actual"""
        return self.current_baseline
    
    def get_baseline_date(self):
        """Retorna la fecha del baseline actual"""
        if self.current_baseline:
            return self.current_baseline['calibration_info']['last_calibration']
        return None
    
    def update_baseline(self, operator_id, new_metrics):
        """
        Actualiza parcialmente el baseline con nuevas métricas.
        Útil para ajustes incrementales.
        """
        baseline_path = os.path.join(self.operators_dir, operator_id, "baseline.json")
        
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    baseline = json.load(f)
                
                # Actualizar métricas específicas
                # TODO: Implementar lógica de actualización incremental
                
                # Actualizar fecha
                baseline['calibration_info']['last_calibration'] = datetime.now().isoformat()
                baseline['calibration_info']['total_calibrations'] += 1
                
                # Guardar
                with open(baseline_path, 'w', encoding='utf-8') as f:
                    json.dump(baseline, f, indent=2, ensure_ascii=False)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error actualizando baseline: {e}")
                
        return False
    
    def force_recalibration(self):
        """Fuerza una recalibración del operador actual"""
        if self.current_operator_id:
            self.start_calibration()
            return True
        return False
    
    # Métodos auxiliares
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0
        
        # Distancias verticales
        v1 = self._distance(eye_points[1], eye_points[5])
        v2 = self._distance(eye_points[2], eye_points[4])
        
        # Distancia horizontal
        h = self._distance(eye_points[0], eye_points[3])
        
        if h == 0:
            return 0
            
        return (v1 + v2) / (2.0 * h)
    
    def _distance(self, p1, p2):
        """Calcula distancia euclidiana entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _get_face_width(self, landmarks):
        """Obtiene el ancho de la cara"""
        if 'chin' in landmarks and len(landmarks['chin']) >= 17:
            return self._distance(landmarks['chin'][0], landmarks['chin'][16])
        return 100
    
    def _calculate_jaw_symmetry(self, jaw_points, center):
        """Calcula simetría de la mandíbula"""
        if len(jaw_points) < 17:
            return 0
        
        # Comparar distancias de puntos simétricos al centro
        symmetry_scores = []
        for i in range(8):
            left_dist = self._distance(jaw_points[i], center)
            right_dist = self._distance(jaw_points[16-i], center)
            symmetry = min(left_dist, right_dist) / max(left_dist, right_dist) if max(left_dist, right_dist) > 0 else 1
            symmetry_scores.append(symmetry)
        
        return 1.0 - np.mean(symmetry_scores)  # Invertir para que más asimetría = más tensión