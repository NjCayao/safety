"""
Módulo de Estimación de Pulso mediante rPPG
===========================================
Estima la frecuencia cardíaca usando fotopletismografía remota (rPPG).
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import time
import logging

class PulseEstimator:
    def __init__(self, fps=30, window_size=10, min_hr=40, max_hr=180):
        """
        Inicializa el estimador de pulso.
        
        Args:
            fps: Frames por segundo de la cámara
            window_size: Ventana de tiempo en segundos para análisis
            min_hr: Frecuencia cardíaca mínima esperada (BPM)
            max_hr: Frecuencia cardíaca máxima esperada (BPM)
        """
        self.logger = logging.getLogger('PulseEstimator')
        
        # Configuración
        self.fps = fps
        self.window_size = window_size
        self.min_hr = min_hr
        self.max_hr = max_hr
        
        # Buffers de señal
        self.buffer_size = self.fps * self.window_size
        self.green_channel_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # ROIs (Regions of Interest)
        self.roi_forehead = None
        self.roi_left_cheek = None
        self.roi_right_cheek = None
        
        # Estado del pulso
        self.current_bpm = 0
        self.confidence = 0
        self.signal_quality = 0
        self.last_valid_bpm = 72  # Valor por defecto
        
        # Historial para estabilización
        self.bpm_history = deque(maxlen=10)
        
        # Filtros
        self.setup_filters()
        
        # Modo de iluminación
        self.is_night_mode = False
        self.ir_compensation = 1.0
        
    def setup_filters(self):
        """Configura filtros para procesamiento de señal"""
        # Frecuencias de corte para filtro pasa-banda (0.8-3 Hz = 48-180 BPM)
        self.lowcut = 0.8
        self.highcut = 3.0
        
        # Crear filtro Butterworth
        nyquist = self.fps / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Coeficientes del filtro
        self.b, self.a = signal.butter(2, [low, high], btype='band')
        
    def extract_rois(self, frame, face_landmarks):
        """
        Extrae las regiones de interés (ROIs) del rostro.
        
        Args:
            frame: Imagen actual
            face_landmarks: Puntos faciales detectados
            
        Returns:
            bool: True si se extrajeron exitosamente
        """
        if not face_landmarks:
            return False
        
        try:
            # ROI de la frente (entre cejas y línea del cabello)
            left_eyebrow = face_landmarks.get('left_eyebrow', [])
            right_eyebrow = face_landmarks.get('right_eyebrow', [])
            
            if left_eyebrow and right_eyebrow:
                # Calcular centro de cejas
                eyebrow_center_y = int((left_eyebrow[2][1] + right_eyebrow[2][1]) / 2)
                forehead_top = eyebrow_center_y - 40  # 40 píxeles arriba
                forehead_bottom = eyebrow_center_y - 10
                forehead_left = left_eyebrow[0][0]
                forehead_right = right_eyebrow[-1][0]
                
                self.roi_forehead = (
                    max(0, forehead_left),
                    max(0, forehead_top),
                    forehead_right,
                    forehead_bottom
                )
            
            # ROI de mejillas
            nose_bridge = face_landmarks.get('nose_bridge', [])
            nose_tip = face_landmarks.get('nose_tip', [])
            
            if nose_bridge and nose_tip:
                # Mejilla izquierda
                cheek_size = 30
                left_cheek_x = nose_bridge[0][0] - 50
                left_cheek_y = nose_tip[0][1] - 20
                
                self.roi_left_cheek = (
                    max(0, left_cheek_x),
                    max(0, left_cheek_y),
                    left_cheek_x + cheek_size,
                    left_cheek_y + cheek_size
                )
                
                # Mejilla derecha
                right_cheek_x = nose_bridge[0][0] + 20
                right_cheek_y = nose_tip[0][1] - 20
                
                self.roi_right_cheek = (
                    max(0, right_cheek_x),
                    max(0, right_cheek_y),
                    right_cheek_x + cheek_size,
                    right_cheek_y + cheek_size
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extrayendo ROIs: {str(e)}")
            return False
    
    def process_frame(self, frame, face_landmarks):
        """
        Procesa un frame para extraer señal de pulso.
        
        Args:
            frame: Frame actual
            face_landmarks: Landmarks faciales
            
        Returns:
            dict: Resultados del procesamiento
        """
        current_time = time.time()
        
        # Extraer ROIs
        if not self.extract_rois(frame, face_landmarks):
            return self._get_default_result()
        
        # Detectar modo de iluminación
        self._detect_lighting_mode(frame)
        
        # Extraer señal de las ROIs
        signal_value = self._extract_signal_from_rois(frame)
        
        if signal_value is not None:
            # Agregar a buffer
            self.green_channel_buffer.append(signal_value)
            self.timestamps.append(current_time)
            
            # Calcular BPM si tenemos suficientes datos
            if len(self.green_channel_buffer) >= self.fps * 3:  # Mínimo 3 segundos
                bpm, confidence, quality = self._calculate_bpm()
                
                if confidence > 0.5:  # Confianza mínima
                    self.current_bpm = bpm
                    self.confidence = confidence
                    self.signal_quality = quality
                    self.last_valid_bpm = bpm
                    self.bpm_history.append(bpm)
        
        return {
            'bpm': self._get_stable_bpm(),
            'confidence': self.confidence,
            'signal_quality': self.signal_quality,
            'is_valid': self.confidence > 0.5,
            'mode': 'IR' if self.is_night_mode else 'RGB',
            'variability': self._calculate_hr_variability()
        }
    
    def _extract_signal_from_rois(self, frame):
        """Extrae señal de las ROIs"""
        try:
            signals = []
            weights = []
            
            # Extraer señal de cada ROI
            for roi, weight in [(self.roi_forehead, 0.5), 
                               (self.roi_left_cheek, 0.25), 
                               (self.roi_right_cheek, 0.25)]:
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    
                    # Validar coordenadas
                    if x2 > x1 and y2 > y1:
                        roi_pixels = frame[y1:y2, x1:x2]
                        
                        if roi_pixels.size > 0:
                            # Método CHROM para mejor extracción de señal
                            signal_value = self._chrom_method(roi_pixels)
                            
                            if signal_value is not None:
                                signals.append(signal_value)
                                weights.append(weight)
            
            # Combinar señales ponderadas
            if signals:
                weighted_signal = np.average(signals, weights=weights)
                return weighted_signal
            
        except Exception as e:
            self.logger.error(f"Error extrayendo señal: {str(e)}")
        
        return None
    
    def _chrom_method(self, roi):
        """
        Implementa el método CHROM para extracción de señal rPPG.
        Más robusto a cambios de iluminación.
        """
        try:
            # Separar canales RGB
            if len(roi.shape) == 3:
                R = np.mean(roi[:, :, 2])
                G = np.mean(roi[:, :, 1])
                B = np.mean(roi[:, :, 0])
                
                # Normalizar
                if R + G + B > 0:
                    r = R / (R + G + B)
                    g = G / (R + G + B)
                    b = B / (R + G + B)
                    
                    # Señal CHROM
                    X = 3*r - 2*g
                    Y = 1.5*r + g - 1.5*b
                    
                    # Combinar señales
                    alpha = np.std(X) / np.std(Y) if np.std(Y) > 0 else 1
                    signal = X - alpha * Y
                    
                    return signal
            else:
                # Modo escala de grises (IR)
                return np.mean(roi) * self.ir_compensation
                
        except Exception as e:
            self.logger.error(f"Error en método CHROM: {str(e)}")
            
        return None
    
    def _calculate_bpm(self):
        """Calcula BPM usando FFT"""
        try:
            # Convertir buffer a array numpy
            signal_array = np.array(self.green_channel_buffer)
            
            # Aplicar filtro pasa-banda
            filtered_signal = signal.filtfilt(self.b, self.a, signal_array)
            
            # Detrend para eliminar deriva
            detrended = signal.detrend(filtered_signal)
            
            # Normalizar
            normalized = (detrended - np.mean(detrended)) / np.std(detrended)
            
            # Aplicar ventana de Hamming
            windowed = normalized * np.hamming(len(normalized))
            
            # FFT
            fft_vals = np.abs(fft(windowed))
            freqs = fftfreq(len(windowed), 1/self.fps)
            
            # Encontrar frecuencias en rango de interés
            min_freq = self.min_hr / 60  # Convertir a Hz
            max_freq = self.max_hr / 60
            
            valid_indices = np.where((freqs > min_freq) & (freqs < max_freq))[0]
            
            if len(valid_indices) > 0:
                # Encontrar pico dominante
                valid_fft = fft_vals[valid_indices]
                valid_freqs = freqs[valid_indices]
                
                peak_idx = np.argmax(valid_fft)
                peak_freq = valid_freqs[peak_idx]
                
                # Convertir a BPM
                bpm = peak_freq * 60
                
                # Calcular confianza basada en la prominencia del pico
                peak_power = valid_fft[peak_idx]
                mean_power = np.mean(valid_fft)
                confidence = min(1.0, (peak_power / mean_power - 1) / 5)
                
                # Calcular calidad de señal
                snr = peak_power / (np.std(valid_fft) + 1e-10)
                quality = min(1.0, snr / 10)
                
                return bpm, confidence, quality
                
        except Exception as e:
            self.logger.error(f"Error calculando BPM: {str(e)}")
        
        return self.last_valid_bpm, 0, 0
    
    def _get_stable_bpm(self):
        """Obtiene BPM estabilizado usando historial"""
        if not self.bpm_history:
            return self.last_valid_bpm
        
        # Filtrar outliers
        recent_bpms = list(self.bpm_history)
        median_bpm = np.median(recent_bpms)
        
        # Eliminar valores muy alejados de la mediana
        filtered_bpms = [bpm for bpm in recent_bpms 
                        if abs(bpm - median_bpm) < 20]
        
        if filtered_bpms:
            return int(np.mean(filtered_bpms))
        
        return int(median_bpm)
    
    def _calculate_hr_variability(self):
        """Calcula variabilidad de frecuencia cardíaca (HRV simplificado)"""
        if len(self.bpm_history) < 5:
            return 0
        
        recent_bpms = list(self.bpm_history)[-5:]
        return np.std(recent_bpms)
    
    def _detect_lighting_mode(self, frame):
        """Detecta si estamos en modo nocturno/IR"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        avg_brightness = np.mean(gray)
        
        self.is_night_mode = avg_brightness < 50
        
        # Ajustar compensación para IR
        if self.is_night_mode:
            self.ir_compensation = 1.5  # Amplificar señal en modo IR
        else:
            self.ir_compensation = 1.0
    
    def draw_pulse_info(self, frame, pulse_data, position=(10, 300)):
        """Dibuja información del pulso en el frame"""
        if not pulse_data:
            return frame
        
        bpm = pulse_data['bpm']
        confidence = pulse_data['confidence']
        quality = pulse_data['signal_quality']
        
        # Color según BPM
        if 60 <= bpm <= 100:
            color = (0, 255, 0)  # Verde - normal
        elif 50 <= bpm < 60 or 100 < bpm <= 110:
            color = (0, 165, 255)  # Naranja - atención
        else:
            color = (0, 0, 255)  # Rojo - anormal
        
        # Texto principal
        text = f"Pulso: {bpm} BPM"
        cv2.putText(frame, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Barra de confianza
        bar_y = position[1] + 25
        bar_width = 150
        bar_height = 8
        
        # Fondo
        cv2.rectangle(frame, (position[0], bar_y),
                     (position[0] + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        
        # Progreso de confianza
        conf_width = int(bar_width * confidence)
        cv2.rectangle(frame, (position[0], bar_y),
                     (position[0] + conf_width, bar_y + bar_height),
                     (0, 255, 0) if confidence > 0.7 else (0, 165, 255), -1)
        
        # Texto de confianza
        conf_text = f"Confianza: {confidence:.0%}"
        cv2.putText(frame, conf_text, (position[0], bar_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Calidad de señal
        quality_text = f"Calidad: {'Alta' if quality > 0.7 else 'Media' if quality > 0.4 else 'Baja'}"
        cv2.putText(frame, quality_text, (position[0], bar_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Dibujar ROIs si están disponibles
        if pulse_data['is_valid']:
            self._draw_rois(frame)
        
        return frame
    
    def _draw_rois(self, frame):
        """Dibuja las ROIs en el frame"""
        # Color verde semi-transparente para ROIs
        overlay = frame.copy()
        
        for roi in [self.roi_forehead, self.roi_left_cheek, self.roi_right_cheek]:
            if roi is not None:
                x1, y1, x2, y2 = roi
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
        
        # Mezclar con transparencia
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    def _get_default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'bpm': self.last_valid_bpm,
            'confidence': 0,
            'signal_quality': 0,
            'is_valid': False,
            'mode': 'N/A',
            'variability': 0
        }
    
    def get_pulse_report(self):
        """Genera reporte del estado del pulso"""
        stable_bpm = self._get_stable_bpm()
        
        # Clasificar estado
        if 60 <= stable_bpm <= 100:
            status = "Normal"
            risk = "Bajo"
        elif 50 <= stable_bpm < 60:
            status = "Bradicardia leve"
            risk = "Medio"
        elif 100 < stable_bpm <= 110:
            status = "Taquicardia leve"
            risk = "Medio"
        else:
            status = "Anormal"
            risk = "Alto"
        
        return {
            'current_bpm': stable_bpm,
            'average_bpm': int(np.mean(self.bpm_history)) if self.bpm_history else stable_bpm,
            'variability': self._calculate_hr_variability(),
            'status': status,
            'risk_level': risk,
            'measurement_quality': self.signal_quality,
            'recommendations': self._get_recommendations(stable_bpm)
        }
    
    def _get_recommendations(self, bpm):
        """Genera recomendaciones basadas en el pulso"""
        if 60 <= bpm <= 100:
            return ["Frecuencia cardíaca normal", "Continúe monitoreando"]
        elif bpm < 50:
            return ["Bradicardia detectada", "Consulte con médico si persiste", 
                   "Evite actividades extenuantes"]
        elif bpm > 120:
            return ["Taquicardia detectada", "Tome un descanso", 
                   "Respire profundamente", "Hidratación adecuada"]
        else:
            return ["Frecuencia cardíaca fuera de rango normal", 
                   "Monitoreo continuo recomendado"]
    
    def reset(self):
        """Reinicia el estimador"""
        self.green_channel_buffer.clear()
        self.timestamps.clear()
        self.bpm_history.clear()
        self.current_bpm = 0
        self.confidence = 0
        self.signal_quality = 0