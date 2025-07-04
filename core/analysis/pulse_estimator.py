"""
Módulo de Estimación de Pulso mediante rPPG - Versión Optimizada
===============================================================
Estima la frecuencia cardíaca usando fotopletismografía remota con calibración.
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import time
import logging

class PulseEstimator:
    def __init__(self, fps=30, window_size=10, headless=False):
        """
        Inicializa el estimador de pulso.
        
        Args:
            fps: Frames por segundo de la cámara
            window_size: Ventana de tiempo en segundos para análisis
            headless: Si True, desactiva visualizaciones (modo servidor)
        """
        self.logger = logging.getLogger('PulseEstimator')
        self.headless = headless
        
        # Configuración
        self.fps = fps
        self.window_size = window_size
        self.min_hr = 40
        self.max_hr = 180
        
        # Buffers de señal
        self.buffer_size = self.fps * self.window_size
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        
        # Estado del pulso
        self.current_bpm = 0
        self.confidence = 0
        self.signal_quality = 0
        self.last_valid_bpm = 72  # Valor por defecto
        
        # Calibración
        self.baseline = None
        self.is_calibrated = False
        self.baseline_skin_tone = None
        self.baseline_roi_brightness = None
        
        # Historial para estabilización
        self.bpm_history = deque(maxlen=10)
        self.hrv_buffer = deque(maxlen=60)  # Para variabilidad
        
        # ROIs faciales
        self.forehead_roi = None
        self.left_cheek_roi = None
        self.right_cheek_roi = None
        
        # Configuración de filtros
        self.setup_filters()
        
        # Modo de iluminación
        self.lighting_mode = 'auto'
        self.last_update = time.time()
        
        # Configuración visual (solo si no es headless)
        if not self.headless:
            self.bar_color = (255, 0, 0)  # Rojo para pulso
            self.bar_width = 200
            self.bar_height = 30
    
    def setup_filters(self):
        """Configura filtros para procesamiento de señal"""
        # Frecuencias de corte para filtro pasa-banda (0.7-3.5 Hz = 42-210 BPM)
        self.lowcut = 0.7
        self.highcut = 3.5
        
        # Crear filtro Butterworth
        nyquist = self.fps / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        # Coeficientes del filtro
        self.b, self.a = signal.butter(3, [low, high], btype='band')
    
    def set_baseline(self, baseline):
        """
        Configura el baseline del operador.
        
        Args:
            baseline: Diccionario con métricas calibradas
        """
        if baseline:
            self.baseline = baseline
            
            # Extraer valores relevantes para pulso
            if 'environment_conditions' in baseline:
                self.baseline_roi_brightness = baseline['environment_conditions'].get('lighting_average', 125)
            
            self.is_calibrated = True
            self.logger.info("Baseline de pulso configurado")
    
    def process_frame(self, frame, face_landmarks):
        """
        Procesa un frame para extraer señal de pulso.
        
        Args:
            frame: Frame actual
            face_landmarks: Landmarks faciales
            
        Returns:
            dict: Resultados del procesamiento
        """
        self.last_update = time.time()
        
        if not self._extract_rois(frame, face_landmarks):
            return self._get_default_result()
        
        # Detectar condiciones de iluminación
        self._analyze_lighting_conditions(frame)
        
        # Extraer señal de las ROIs
        signal_value = self._extract_ppg_signal(frame)
        
        if signal_value is not None:
            # Agregar a buffer con timestamp
            self.signal_buffer.append(signal_value)
            self.timestamps.append(self.last_update)
            
            # Calcular BPM si tenemos suficientes datos
            if len(self.signal_buffer) >= self.fps * 2:  # Mínimo 3 segundos
                bpm, confidence, quality = self._calculate_bpm_advanced()
                
                if confidence > 0.5:
                    self.current_bpm = bpm
                    self.confidence = confidence
                    self.signal_quality = quality
                    self.last_valid_bpm = bpm
                    self.bpm_history.append(bpm)
                    
                    # Actualizar HRV
                    if len(self.bpm_history) > 2:
                        self.hrv_buffer.append({
                            'bpm': bpm,
                            'timestamp': self.last_update
                        })
        
        return {
            'bpm': self._get_stable_bpm(),
            'confidence': self.confidence,
            'signal_quality': self.signal_quality,
            'is_valid': self.confidence > 0.5,
            'hrv': self._calculate_hrv(),
            'is_calibrated': self.is_calibrated,
            'lighting_mode': self.lighting_mode
        }
    
    def _extract_rois(self, frame, face_landmarks):
        """Extrae regiones de interés optimizadas para rPPG"""
        if not face_landmarks:
            return False
        
        try:
            # Forehead ROI (mejor para rPPG)
            if 'left_eyebrow' in face_landmarks and 'right_eyebrow' in face_landmarks:
                left_brow = face_landmarks['left_eyebrow']
                right_brow = face_landmarks['right_eyebrow']
                
                # Centro entre cejas
                brow_center_x = (left_brow[2][0] + right_brow[2][0]) // 2
                brow_center_y = (left_brow[2][1] + right_brow[2][1]) // 2
                
                # ROI de frente (más arriba de las cejas)
                roi_size = 40
                forehead_y = brow_center_y - 30
                
                self.forehead_roi = (
                    max(0, brow_center_x - roi_size),
                    max(0, forehead_y),
                    brow_center_x + roi_size,
                    forehead_y + roi_size
                )
            
            # Mejillas (backup ROIs)
            if 'nose_tip' in face_landmarks:
                nose_tip = face_landmarks['nose_tip'][2]
                
                # Mejilla izquierda
                self.left_cheek_roi = (
                    max(0, nose_tip[0] - 60),
                    max(0, nose_tip[1] - 20),
                    nose_tip[0] - 20,
                    nose_tip[1] + 20
                )
                
                # Mejilla derecha
                self.right_cheek_roi = (
                    nose_tip[0] + 20,
                    max(0, nose_tip[1] - 20),
                    nose_tip[0] + 60,
                    nose_tip[1] + 20
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error extrayendo ROIs: {e}")
            return False
    
    def _extract_ppg_signal(self, frame):
        """Extrae señal PPG usando método optimizado"""
        try:
            signals = []
            weights = []
            
            # Procesar cada ROI
            for roi, weight in [(self.forehead_roi, 0.6), 
                              (self.left_cheek_roi, 0.2),
                              (self.right_cheek_roi, 0.2)]:
                if roi is not None:
                    x1, y1, x2, y2 = roi
                    
                    # Validar coordenadas
                    if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0:
                        roi_pixels = frame[y1:y2, x1:x2]
                        
                        if roi_pixels.size > 0:
                            # Método CHROM mejorado
                            signal_value = self._chrom_method_optimized(roi_pixels)
                            
                            if signal_value is not None:
                                signals.append(signal_value)
                                weights.append(weight)
            
            # Combinar señales
            if signals:
                if self.is_calibrated and self.baseline_roi_brightness:
                    # Ajustar por condiciones de iluminación
                    current_brightness = np.mean([np.mean(frame[roi[1]:roi[3], roi[0]:roi[2]]) 
                                                 for roi in [self.forehead_roi] if roi])
                    brightness_factor = current_brightness / self.baseline_roi_brightness
                    brightness_factor = np.clip(brightness_factor, 0.5, 2.0)
                else:
                    brightness_factor = 1.0
                
                weighted_signal = np.average(signals, weights=weights) * brightness_factor
                return weighted_signal
                
        except Exception as e:
            self.logger.error(f"Error extrayendo señal PPG: {e}")
            
        return None
    
    def _chrom_method_optimized(self, roi):
        """Método CHROM optimizado para extracción de señal rPPG"""
        try:
            # Usar solo píxeles de piel (eliminar pelo, ojos, etc.)
            skin_mask = self._create_skin_mask(roi)
            
            if np.sum(skin_mask) < roi.size * 0.3:  # Menos del 30% es piel
                return None
            
            # Aplicar máscara
            masked_roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
            
            # Extraer canales RGB
            if len(roi.shape) == 3:
                # Promediar solo píxeles de piel
                R = np.mean(masked_roi[:, :, 2][skin_mask > 0])
                G = np.mean(masked_roi[:, :, 1][skin_mask > 0])
                B = np.mean(masked_roi[:, :, 0][skin_mask > 0])
                
                # Normalizar
                sum_rgb = R + G + B
                if sum_rgb > 0:
                    r = R / sum_rgb
                    g = G / sum_rgb
                    b = B / sum_rgb
                    
                    # CHROM signals
                    X = 3*r - 2*g
                    Y = 1.5*r + g - 1.5*b
                    
                    # Combinar con pesos adaptativos
                    if np.std(Y) > 0:
                        alpha = np.std(X) / np.std(Y)
                        signal = X - alpha * Y
                    else:
                        signal = X
                    
                    return signal
            else:
                # Modo escala de grises (cámara IR)
                return np.mean(roi[skin_mask > 0])
                
        except Exception as e:
            self.logger.debug(f"Error en CHROM: {e}")
            
        return None
    
    def _create_skin_mask(self, roi):
        """Crea máscara para detectar píxeles de piel"""
        try:
            # Convertir a HSV para mejor detección de piel
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Rango de piel en HSV (ajustado para diversidad)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Crear máscara
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Limpiar ruido
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
            
        except:
            # Si falla, usar toda la ROI
            return np.ones(roi.shape[:2], dtype=np.uint8) * 255
    
    def _calculate_bpm_advanced(self):
        """Calcula BPM usando FFT con validación avanzada"""
        try:
            # Convertir buffer a array
            signal_array = np.array(self.signal_buffer)
            
            # Pre-procesamiento
            # 1. Detrend
            detrended = signal.detrend(signal_array)
            
            # 2. Filtro pasa-banda
            filtered = signal.filtfilt(self.b, self.a, detrended)
            
            # 3. Normalizar
            normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)
            
            # 4. Ventana de Hamming
            windowed = normalized * np.hamming(len(normalized))
            
            # FFT
            fft_vals = np.abs(fft(windowed))
            freqs = fftfreq(len(windowed), 1/self.fps)
            
            # Buscar picos en rango de interés
            min_freq = self.min_hr / 60
            max_freq = self.max_hr / 60
            
            valid_indices = np.where((freqs > min_freq) & (freqs < max_freq))[0]
            
            if len(valid_indices) > 0:
                valid_fft = fft_vals[valid_indices]
                valid_freqs = freqs[valid_indices]
                
                # Encontrar múltiples picos
                peaks, properties = signal.find_peaks(valid_fft, 
                                                     height=np.max(valid_fft)*0.3,
                                                     distance=5)
                
                if len(peaks) > 0:
                    # Usar el pico más prominente
                    main_peak_idx = peaks[np.argmax(valid_fft[peaks])]
                    peak_freq = valid_freqs[main_peak_idx]
                    
                    # Convertir a BPM
                    bpm = peak_freq * 60
                    
                    # Validar con armónicos
                    confidence = self._validate_with_harmonics(valid_fft, valid_freqs, peak_freq)
                    
                    # Calidad de señal
                    snr = self._calculate_snr(valid_fft, main_peak_idx)
                    quality = min(1.0, snr / 10)
                    
                    return int(bpm), confidence, quality
                    
        except Exception as e:
            self.logger.error(f"Error calculando BPM: {e}")
        
        return self.last_valid_bpm, 0, 0
    
    def _validate_with_harmonics(self, fft_vals, freqs, fundamental_freq):
        """Valida la frecuencia fundamental verificando armónicos"""
        confidence = 0.5  # Base
        
        # Buscar primer armónico (2x frecuencia)
        harmonic_freq = fundamental_freq * 2
        harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
        
        if harmonic_idx < len(fft_vals):
            # Si existe armónico significativo, aumentar confianza
            if fft_vals[harmonic_idx] > np.mean(fft_vals) * 2:
                confidence += 0.3
        
        # Verificar que no hay picos más fuertes en frecuencias no relacionadas
        fundamental_idx = np.argmin(np.abs(freqs - fundamental_freq))
        if fundamental_idx < len(fft_vals):
            if fft_vals[fundamental_idx] == np.max(fft_vals):
                confidence += 0.2
        
        return min(1.0, confidence)
    
    def _calculate_snr(self, fft_vals, peak_idx):
        """Calcula relación señal-ruido"""
        if peak_idx >= len(fft_vals):
            return 0
            
        signal_power = fft_vals[peak_idx]
        
        # Ruido: promedio excluyendo el pico y sus vecinos
        noise_indices = list(range(len(fft_vals)))
        for i in range(max(0, peak_idx-2), min(len(fft_vals), peak_idx+3)):
            if i in noise_indices:
                noise_indices.remove(i)
        
        if noise_indices:
            noise_power = np.mean(fft_vals[noise_indices])
            snr = signal_power / (noise_power + 1e-10)
            return snr
        
        return 0
    
    def _get_stable_bpm(self):
        """Obtiene BPM estabilizado usando historial"""
        if not self.bpm_history:
            return self.last_valid_bpm
        
        # Filtrar outliers usando MAD (Median Absolute Deviation)
        recent_bpms = list(self.bpm_history)
        median_bpm = np.median(recent_bpms)
        mad = np.median([abs(bpm - median_bpm) for bpm in recent_bpms])
        
        # Filtrar valores dentro de 2.5 MAD
        threshold = 2.5 * mad if mad > 0 else 10
        filtered_bpms = [bpm for bpm in recent_bpms 
                        if abs(bpm - median_bpm) <= threshold]
        
        if filtered_bpms:
            # Media ponderada (valores más recientes tienen más peso)
            weights = np.linspace(0.5, 1.0, len(filtered_bpms))
            weighted_avg = np.average(filtered_bpms, weights=weights)
            return int(weighted_avg)
        
        return int(median_bpm)
    
    def _calculate_hrv(self):
        """Calcula variabilidad de frecuencia cardíaca simplificada"""
        if len(self.hrv_buffer) < 5:
            return 0
        
        # Calcular intervalos RR (tiempo entre latidos)
        recent_data = list(self.hrv_buffer)[-10:]
        rr_intervals = []
        
        for i in range(1, len(recent_data)):
            if recent_data[i]['bpm'] > 0 and recent_data[i-1]['bpm'] > 0:
                # Aproximar intervalo RR desde BPM
                rr_interval = 60000 / recent_data[i]['bpm']  # en ms
                rr_intervals.append(rr_interval)
        
        if len(rr_intervals) >= 2:
            # RMSSD simplificado
            differences = [abs(rr_intervals[i] - rr_intervals[i-1]) 
                          for i in range(1, len(rr_intervals))]
            rmssd = np.sqrt(np.mean(np.square(differences)))
            
            # Normalizar a escala 0-100
            hrv_score = min(100, (rmssd / 50) * 100)
            return int(hrv_score)
        
        return 0
    
    def _analyze_lighting_conditions(self, frame):
        """Analiza condiciones de iluminación"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        avg_brightness = np.mean(gray)
        
        if avg_brightness < 50:
            self.lighting_mode = 'low_light'
        elif avg_brightness > 200:
            self.lighting_mode = 'high_light'
        else:
            self.lighting_mode = 'normal'
    
    def draw_pulse_bar(self, frame, pulse_data, position=(None, None)):
        """
        Dibuja barra simple con BPM.
        
        Args:
            frame: Frame donde dibujar
            pulse_data: Datos de pulso
            position: (x, y) posición de la barra
        """
        if self.headless or not pulse_data:
            return frame
        
        h, w = frame.shape[:2]
        
        # Posición por defecto
        if position[0] is None:
            bar_x = w - self.bar_width - 20
        else:
            bar_x = position[0]
            
        if position[1] is None:
            bar_y = 100  # Debajo de emoción
        else:
            bar_y = position[1]
        
        bpm = pulse_data['bpm']
        confidence = pulse_data['confidence']
        is_valid = pulse_data['is_valid']
        
        # Color según BPM
        if 60 <= bpm <= 100:
            color = (0, 255, 0)  # Verde
        elif 50 <= bpm < 60 or 100 < bpm <= 110:
            color = (0, 165, 255)  # Naranja
        else:
            color = (0, 0, 255)  # Rojo
        
        # Texto principal
        if is_valid:
            text = f"❤️ PULSO: {bpm} BPM"
        else:
            text = f"❤️ PULSO: Midiendo..."
            color = (150, 150, 150)
        
        cv2.putText(frame, text, (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Barra de confianza
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + self.bar_width, bar_y + self.bar_height),
                     (50, 50, 50), -1)
        
        if is_valid:
            conf_width = int(self.bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y),
                         (bar_x + conf_width, bar_y + self.bar_height),
                         color, -1)
        
        cv2.rectangle(frame, (bar_x, bar_y),
                     (bar_x + self.bar_width, bar_y + self.bar_height),
                     (200, 200, 200), 1)
        
        # Mostrar HRV si está disponible
        if is_valid and 'hrv' in pulse_data and pulse_data['hrv'] > 0:
            hrv_text = f"HRV: {pulse_data['hrv']}"
            cv2.putText(frame, hrv_text, (bar_x + self.bar_width + 10, bar_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def draw_pulse_info(self, frame, pulse_data, position=(10, 300)):
        """Método de compatibilidad - usa draw_pulse_bar"""
        return self.draw_pulse_bar(frame, pulse_data, position)
    
    def get_pulse_report(self):
        """
        Genera reporte del estado del pulso para servidor.
        
        Returns:
            dict: Reporte JSON
        """
        stable_bpm = self._get_stable_bpm()
        
        # Clasificar estado
        if 60 <= stable_bpm <= 100:
            status = 'normal'
            risk = 'low'
        elif 50 <= stable_bpm < 60:
            status = 'bradycardia'
            risk = 'medium'
        elif 100 < stable_bpm <= 120:
            status = 'tachycardia_mild'
            risk = 'medium'
        elif stable_bpm > 120:
            status = 'tachycardia'
            risk = 'high'
        else:
            status = 'abnormal'
            risk = 'high'
        
        # Calcular estadísticas
        if self.bpm_history:
            avg_bpm = int(np.mean(self.bpm_history))
            min_bpm = int(np.min(self.bpm_history))
            max_bpm = int(np.max(self.bpm_history))
        else:
            avg_bpm = min_bpm = max_bpm = stable_bpm
        
        return {
            'timestamp': self.last_update,
            'current_bpm': stable_bpm,
            'statistics': {
                'average_bpm': avg_bpm,
                'min_bpm': min_bpm,
                'max_bpm': max_bpm,
                'samples': len(self.bpm_history)
            },
            'hrv': self._calculate_hrv(),
            'status': status,
            'risk_level': risk,
            'confidence': self.confidence,
            'signal_quality': self.signal_quality,
            'is_calibrated': self.is_calibrated,
            'lighting_conditions': self.lighting_mode,
            'recommendations': self._get_recommendations(stable_bpm, status)
        }
    
    def _get_recommendations(self, bpm, status):
        """Genera recomendaciones basadas en el pulso"""
        recommendations = []
        
        if status == 'normal':
            recommendations.append("Frecuencia cardíaca normal")
        elif status == 'bradycardia':
            recommendations.append("Frecuencia cardíaca baja - Monitorear")
            recommendations.append("Si presenta mareos, consultar médico")
        elif status == 'tachycardia_mild':
            recommendations.append("Frecuencia cardíaca elevada")
            recommendations.append("Tomar un descanso y respirar profundo")
        elif status == 'tachycardia':
            recommendations.append("Frecuencia cardíaca muy alta")
            recommendations.append("Detener actividad y descansar")
            recommendations.append("Si persiste, buscar atención médica")
        
        # HRV bajo indica estrés
        hrv = self._calculate_hrv()
        if hrv < 20 and hrv > 0:
            recommendations.append("Variabilidad cardíaca baja - Posible estrés")
        
        return recommendations[:3]  # Máximo 3 recomendaciones
    
    def _get_default_result(self):
        """Resultado por defecto cuando no hay datos"""
        return {
            'bpm': self.last_valid_bpm,
            'confidence': 0,
            'signal_quality': 0,
            'is_valid': False,
            'hrv': 0,
            'is_calibrated': self.is_calibrated,
            'lighting_mode': 'unknown'
        }
    
    def reset(self):
        """Reinicia el estimador"""
        self.signal_buffer.clear()
        self.timestamps.clear()
        self.bpm_history.clear()
        self.hrv_buffer.clear()
        self.current_bpm = 0
        self.confidence = 0
        self.signal_quality = 0