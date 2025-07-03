"""
Módulo Avanzado de Análisis de Fatiga
=====================================
Complementa la detección de microsueños con análisis facial avanzado.
"""

import cv2
import numpy as np
from collections import deque
import time
import logging

class AdvancedFatigueAnalyzer:
    def __init__(self):
        """Inicializa el analizador avanzado de fatiga"""
        self.logger = logging.getLogger('AdvancedFatigueAnalyzer')
        
        # Configuración
        self.ear_threshold = 0.25
        self.blink_threshold = 0.2
        self.time_window = 60  # segundos
        
        # Historial de datos
        self.ear_history = deque(maxlen=30)
        self.blink_history = []
        self.head_pose_history = deque(maxlen=20)
        self.last_blink_time = 0
        self.last_ear = 0.3
        
        # Métricas de fatiga
        self.fatigue_score = 0
        self.fatigue_indicators = {
            'blink_rate': 0,
            'eye_closure': 0,
            'head_drooping': 0,
            'perclos': 0,  # Percentage of Eye Closure
            'microsleeps': 0
        }
        
        # Colores para visualización
        self.fatigue_colors = {
            'alert': (0, 255, 0),      # Verde
            'drowsy': (0, 165, 255),   # Naranja
            'fatigued': (0, 0, 255)    # Rojo
        }
        
    def analyze(self, frame, face_landmarks, face_location=None):
        """
        Analiza múltiples indicadores de fatiga.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales
            face_location: Ubicación del rostro (opcional)
            
        Returns:
            dict: Análisis completo de fatiga
        """
        current_time = time.time()
        
        # 1. Calcular EAR (Eye Aspect Ratio)
        left_ear = self._calculate_ear(face_landmarks.get('left_eye', []))
        right_ear = self._calculate_ear(face_landmarks.get('right_eye', []))
        avg_ear = (left_ear + right_ear) / 2
        
        # 2. Detectar parpadeos
        is_blink = self._detect_blink(avg_ear, current_time)
        
        # 3. Calcular PERCLOS (Percentage of Eye Closure)
        perclos = self._calculate_perclos(avg_ear)
        
        # 4. Detectar cabeceo/head drooping
        head_droop = self._detect_head_drooping(face_landmarks, face_location)
        
        # 5. Analizar frecuencia de parpadeo
        blink_rate = self._calculate_blink_rate(current_time)
        
        # 6. Detectar microsueños (ojos cerrados > 0.5s)
        microsleep = self._detect_microsleep(avg_ear, current_time)
        
        # Actualizar indicadores
        self.fatigue_indicators = {
            'blink_rate': blink_rate,
            'eye_closure': int((1 - avg_ear) * 100),
            'head_drooping': int(head_droop * 100),
            'perclos': int(perclos * 100),
            'microsleeps': len([t for t in self.blink_history 
                              if current_time - t < 60 and 
                              self._is_microsleep_blink(t)])
        }
        
        # Calcular score general de fatiga
        self.fatigue_score = self._calculate_fatigue_score()
        
        # Limpiar historial antiguo
        self._clean_old_data(current_time)
        
        return {
            'fatigue_score': self.fatigue_score,
            'fatigue_level': self._categorize_fatigue(self.fatigue_score),
            'indicators': self.fatigue_indicators,
            'ear_value': avg_ear,
            'is_blinking': is_blink,
            'is_microsleep': microsleep,
            'color': self._get_fatigue_color(self.fatigue_score),
            'recommendations': self._get_recommendations(self.fatigue_score)
        }
    
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        if len(eye_points) != 6:
            return 0.3  # Valor por defecto
        
        # Distancias verticales
        A = self._distance(eye_points[1], eye_points[5])
        B = self._distance(eye_points[2], eye_points[4])
        
        # Distancia horizontal
        C = self._distance(eye_points[0], eye_points[3])
        
        # EAR formula
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def _detect_blink(self, ear, current_time):
        """Detecta parpadeos"""
        is_blink = False
        
        # Detectar transición de abierto a cerrado
        if self.last_ear > self.blink_threshold and ear <= self.blink_threshold:
            # Evitar contar parpadeos muy seguidos (menos de 0.1s)
            if current_time - self.last_blink_time > 0.1:
                self.blink_history.append(current_time)
                self.last_blink_time = current_time
                is_blink = True
        
        self.last_ear = ear
        self.ear_history.append(ear)
        
        return is_blink
    
    def _calculate_perclos(self, current_ear):
        """
        Calcula PERCLOS - Porcentaje del tiempo con ojos cerrados
        en los últimos segundos.
        """
        if not self.ear_history:
            return 0
        
        # Contar frames con ojos cerrados (EAR bajo)
        closed_count = sum(1 for ear in self.ear_history if ear < self.ear_threshold)
        total_count = len(self.ear_history)
        
        perclos = (closed_count / total_count) if total_count > 0 else 0
        return perclos
    
    def _detect_head_drooping(self, landmarks, face_location):
        """Detecta si la cabeza está cayendo (señal de fatiga)"""
        if not face_location or 'nose_tip' not in landmarks:
            return 0.0
        
        # Analizar posición vertical de la nariz respecto al rostro
        nose_tip = landmarks['nose_tip'][0] if landmarks['nose_tip'] else None
        if not nose_tip:
            return 0.0
        
        top, right, bottom, left = face_location
        face_height = bottom - top
        nose_y_position = nose_tip[1]
        
        # Posición relativa de la nariz (0 = arriba, 1 = abajo)
        relative_nose_position = (nose_y_position - top) / face_height if face_height > 0 else 0.5
        
        # Guardar en historial
        self.head_pose_history.append(relative_nose_position)
        
        # Detectar tendencia descendente
        if len(self.head_pose_history) > 10:
            recent_positions = list(self.head_pose_history)[-10:]
            # Calcular pendiente (positiva = cabeza cayendo)
            slope = np.polyfit(range(len(recent_positions)), recent_positions, 1)[0]
            
            # Normalizar pendiente a 0-1
            droop_indicator = min(1.0, max(0.0, slope * 100))
            return droop_indicator
        
        return 0.0
    
    def _calculate_blink_rate(self, current_time):
        """Calcula parpadeos por minuto"""
        # Filtrar parpadeos en el último minuto
        recent_blinks = [t for t in self.blink_history if current_time - t < 60]
        
        # Calcular tasa
        blinks_per_minute = len(recent_blinks)
        
        return blinks_per_minute
    
    def _detect_microsleep(self, ear, current_time):
        """Detecta microsueños (ojos cerrados por tiempo prolongado)"""
        # Si EAR es muy bajo, contar tiempo
        if ear < self.ear_threshold:
            # Buscar cuánto tiempo llevan cerrados
            closed_duration = 0
            for i in range(len(self.ear_history)-1, -1, -1):
                if self.ear_history[i] < self.ear_threshold:
                    closed_duration += 1
                else:
                    break
            
            # Si llevan cerrados más de 15 frames (~0.5s a 30fps)
            if closed_duration > 15:
                return True
        
        return False
    
    def _is_microsleep_blink(self, blink_time):
        """Determina si un parpadeo fue realmente un microsueño"""
        # Verificar duración del cierre de ojos alrededor del parpadeo
        # (Esta es una simplificación, en producción sería más complejo)
        return False  # Por ahora retorna False
    
    def _calculate_fatigue_score(self):
        """Calcula score general de fatiga (0-100)"""
        weights = {
            'blink_rate': 0.20,
            'eye_closure': 0.30,
            'perclos': 0.25,
            'head_drooping': 0.15,
            'microsleeps': 0.10
        }
        
        # Normalizar indicadores
        normalized = {}
        
        # Frecuencia de parpadeo (normal: 15-20 por minuto)
        blink_rate = self.fatigue_indicators['blink_rate']
        if blink_rate < 10:  # Muy pocos parpadeos = fatiga
            normalized['blink_rate'] = (10 - blink_rate) * 10
        elif blink_rate > 25:  # Muchos parpadeos = fatiga
            normalized['blink_rate'] = min(100, (blink_rate - 25) * 5)
        else:
            normalized['blink_rate'] = 0
        
        # Otros indicadores ya están en porcentaje
        normalized['eye_closure'] = self.fatigue_indicators['eye_closure']
        normalized['perclos'] = self.fatigue_indicators['perclos'] * 2  # Amplificar
        normalized['head_drooping'] = self.fatigue_indicators['head_drooping']
        normalized['microsleeps'] = min(100, self.fatigue_indicators['microsleeps'] * 20)
        
        # Calcular score ponderado
        score = 0
        for indicator, weight in weights.items():
            score += normalized.get(indicator, 0) * weight
        
        return int(min(100, max(0, score)))
    
    def _categorize_fatigue(self, score):
        """Categoriza el nivel de fatiga"""
        if score < 30:
            return "alerta"
        elif score < 60:
            return "somnoliento"
        else:
            return "fatigado"
    
    def _get_fatigue_color(self, score):
        """Obtiene color según nivel de fatiga"""
        if score < 30:
            return self.fatigue_colors['alert']
        elif score < 60:
            return self.fatigue_colors['drowsy']
        else:
            return self.fatigue_colors['fatigued']
    
    def _get_recommendations(self, score):
        """Genera recomendaciones según fatiga"""
        if score < 30:
            return ["Estado de alerta normal"]
        elif score < 60:
            return ["Considere tomar un café o té",
                   "Realice ejercicios de estiramiento",
                   "Mejore la ventilación del área"]
        else:
            return ["⚠️ FATIGA ELEVADA DETECTADA",
                   "Tome un descanso de 15-20 minutos",
                   "Considere cambiar de operador",
                   "Evite operar maquinaria pesada"]
    
    def draw_fatigue_info(self, frame, fatigue_data, position=(10, 100)):
        """Dibuja información de fatiga en el frame"""
        if not fatigue_data:
            return frame
        
        score = fatigue_data['fatigue_score']
        level = fatigue_data['fatigue_level']
        color = fatigue_data['color']
        
        # Título principal
        title = f"Fatiga: {score}% ({level.upper()})"
        cv2.putText(frame, title, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barra de progreso
        bar_width = 200
        bar_height = 15
        bar_x = position[0]
        bar_y = position[1] + 25
        
        # Fondo
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progreso
        progress_width = int(bar_width * (score / 100))
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + progress_width, bar_y + bar_height),
                     color, -1)
        
        # Indicadores específicos
        y_offset = bar_y + bar_height + 20
        indicators = fatigue_data['indicators']
        
        # Mostrar indicadores clave
        key_indicators = [
            ('Parpadeos/min', indicators['blink_rate']),
            ('PERCLOS', f"{indicators['perclos']}%"),
            ('Microsueños', indicators['microsleeps'])
        ]
        
        for i, (label, value) in enumerate(key_indicators):
            text = f"{label}: {value}"
            cv2.putText(frame, text, (position[0], y_offset + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mostrar EAR actual
        if 'ear_value' in fatigue_data:
            ear_text = f"EAR: {fatigue_data['ear_value']:.2f}"
            cv2.putText(frame, ear_text, (position[0], y_offset + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def _distance(self, p1, p2):
        """Calcula distancia entre dos puntos"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _clean_old_data(self, current_time):
        """Limpia datos antiguos del historial"""
        # Limpiar parpadeos antiguos
        self.blink_history = [t for t in self.blink_history 
                             if current_time - t < self.time_window]
    
    def get_fatigue_report(self):
        """Genera reporte detallado de fatiga"""
        return {
            'score': self.fatigue_score,
            'level': self._categorize_fatigue(self.fatigue_score),
            'indicators': self.fatigue_indicators,
            'blink_count_last_minute': len([t for t in self.blink_history 
                                           if time.time() - t < 60]),
            'average_ear': np.mean(self.ear_history) if self.ear_history else 0.3,
            'recommendations': self._get_recommendations(self.fatigue_score)
        }
    
    def reset(self):
        """Reinicia el analizador"""
        self.ear_history.clear()
        self.blink_history.clear()
        self.head_pose_history.clear()
        self.fatigue_score = 0