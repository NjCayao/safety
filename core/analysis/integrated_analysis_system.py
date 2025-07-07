"""
Sistema Integrado de Análisis - Versión Compatible
=================================================
Compatible con versiones anteriores de los módulos.
"""

import time
import logging

# Importar módulos de análisis
from .fatigue_detector import FatigueDetector
from .stress_analyzer import StressAnalyzer
from .pulse_estimator import PulseEstimator
from .emotion_analyzer import EmotionAnalyzer
from .anomaly_detector import AnomalyDetector
from .calibration_manager import CalibrationManager
from .analysis_dashboard import AnalysisDashboard

class IntegratedAnalysisSystem:
    def __init__(self, operators_dir="operators", headless=False):
        """
        Inicializa el sistema integrado de análisis.
        
        Args:
            operators_dir: Directorio donde se guardan los datos de operadores
            headless: Si True, modo servidor sin GUI
        """
        self.operators_dir = operators_dir
        self.headless = headless
        self.logger = logging.getLogger('IntegratedAnalysis')
        
        # Inicializar gestor de calibración
        self.calibration_manager = CalibrationManager(operators_dir)
        
        # Inicializar módulos de análisis con compatibilidad
        self.logger.info("Inicializando módulos de análisis...")
        
        # FatigueDetector
        try:
            self.fatigue_detector = FatigueDetector(headless=headless)
        except TypeError:
            self.fatigue_detector = FatigueDetector()
            self.logger.info("FatigueDetector inicializado sin headless")
        
        # StressAnalyzer
        try:
            self.stress_analyzer = StressAnalyzer(headless=headless)
        except TypeError:
            self.stress_analyzer = StressAnalyzer()
            self.logger.info("StressAnalyzer inicializado sin headless")
        
        # PulseEstimator
        try:
            self.pulse_estimator = PulseEstimator(headless=headless)
        except TypeError:
            self.pulse_estimator = PulseEstimator()
            self.logger.info("PulseEstimator inicializado sin headless")
        
        # EmotionAnalyzer
        try:
            self.emotion_analyzer = EmotionAnalyzer(headless=headless)
        except TypeError:
            self.emotion_analyzer = EmotionAnalyzer()
            self.logger.info("EmotionAnalyzer inicializado sin headless")
        
        # AnomalyDetector
        try:
            self.anomaly_detector = AnomalyDetector(headless=headless)
        except TypeError:
            self.anomaly_detector = AnomalyDetector()
            self.logger.info("AnomalyDetector inicializado sin headless")
        
        # Dashboard (solo si no es headless)
        self.dashboard = AnalysisDashboard(panel_width=300, position='right') if not headless else None
        
        # Estado actual
        self.current_operator = None
        self.is_calibrated = False
        self.analysis_enabled = True
        
        # Historial para tendencias
        self.analysis_history = []
        self.max_history_size = 300  # ~10 segundos a 30fps
        
        # Estadísticas
        self.stats = {
            'frames_analyzed': 0,
            'calibrations_completed': 0,
            'alerts_generated': 0,
            'start_time': time.time()
        }
        
        self.logger.info("Sistema de análisis integrado inicializado")

        # NUEVO: Gestor de reportes
        from core.reports.report_manager import get_report_manager
        self.report_manager = get_report_manager()
        
        # NUEVO: Configuración de reportes
        self.report_config = {
            'enabled': True,
            'cooldown_seconds': 60,  # 1 minuto entre reportes del mismo tipo
            'include_frame': True,
            
            # Umbrales críticos (ambos deben cumplirse)
            'critical_stress_threshold': 80,
            'critical_fatigue_threshold': 80,
            'critical_duration': 10,  # segundos
            
            # Umbrales para pulso
            'pulse_high_threshold': 120,
            'pulse_low_threshold': 50,
            'pulse_duration': 10,  # segundos
            
            # Umbrales para anomalías
            'intoxication_threshold': 40,
            'intoxication_duration': 8,
            'neurological_threshold': 30,
            'neurological_duration': 5,
            'erratic_threshold': 50,
            'erratic_duration': 10
        }
        
        #  Control de reportes
        self.last_report_time = {}
        self.condition_start_times = {}
        
        #  Frame actual para reportes
        self._current_frame = None
    
    def analyze_operator(self, frame, face_landmarks, face_location, operator_info=None):
        """
        Analiza un operador con todos los módulos disponibles.
        
        Args:
            frame: Frame actual de video
            face_landmarks: Landmarks faciales detectados
            face_location: Ubicación del rostro (top, right, bottom, left)
            operator_info: Información del operador {'id': '12345678', 'name': 'Juan'}
            
        Returns:
            tuple: (frame_con_dashboard, resultados_análisis)
        """
        #  Guardar frame actual para reportes
        self._current_frame = frame

        # Verificar si hay operador
        if not operator_info:
            return frame, {'status': 'no_operator'}
        
        # Verificar si cambió el operador
        if not self.current_operator or self.current_operator['id'] != operator_info['id']:
            self._handle_operator_change(operator_info)
        
        # Si está calibrando
        if self.calibration_manager.is_calibrating:
            self.calibration_manager.add_calibration_sample(face_landmarks)
            progress = self.calibration_manager.get_calibration_progress()
            
            # Mostrar progreso de calibración
            if self.dashboard:
                frame = self.dashboard.draw_calibration_progress(frame, progress, operator_info['name'])
            
            return frame, {
                'status': 'calibrating',
                'progress': progress,
                'operator': operator_info
            }       

        # Análisis normal
        if not self.analysis_enabled:
            return frame, {'status': 'disabled'}
        
        # Recopilar resultados de todos los análisis
        analysis_results = {
            'operator': self.current_operator,
            'timestamp': time.time(),
            'frame_number': self.stats['frames_analyzed'],
            'analysis': {}
        }
        
        # Ejecutar cada módulo de análisis
        try:
            # 1. Detección de fatiga
            fatigue_result = self.fatigue_detector.analyze(face_landmarks)
            analysis_results['analysis']['fatigue'] = fatigue_result
            
            # 2. Análisis de estrés
            stress_result = self.stress_analyzer.analyze(frame, face_landmarks)
            analysis_results['analysis']['stress'] = stress_result
            
            # 3. Estimación de pulso
            pulse_result = self.pulse_estimator.process_frame(frame, face_landmarks)
            analysis_results['analysis']['pulse'] = pulse_result
            
            # 4. Análisis de emociones
            emotion_result = self.emotion_analyzer.analyze(frame, face_landmarks)
            analysis_results['analysis']['emotion'] = emotion_result
            
            # 5. Detección de anomalías (necesita datos de emoción)
            anomaly_result = self.anomaly_detector.analyze(frame, face_landmarks, emotion_result)
            analysis_results['analysis']['anomaly'] = anomaly_result
            
        except Exception as e:
            self.logger.error(f"Error en análisis: {e}")
            import traceback
            traceback.print_exc()
            return frame, {'status': 'error', 'message': str(e)}
        
        # Evaluar estado general
        overall_assessment = self._evaluate_overall_state(analysis_results['analysis'])
        analysis_results['overall_assessment'] = overall_assessment
        
        # Generar alertas si es necesario
        alerts = self._generate_alerts(analysis_results['analysis'])
        if alerts:
            analysis_results['alerts'] = alerts
            self.stats['alerts_generated'] += len(alerts)
        
        # Generar recomendaciones
        recommendations = self._generate_recommendations(analysis_results['analysis'])
        if recommendations:
            analysis_results['recommendations'] = recommendations
        
        # Agregar al historial
        self._update_history(analysis_results)
        
        # Renderizar dashboard si no es headless
        if self.dashboard and not self.headless:
            frame = self.dashboard.render(frame, analysis_results)

        # NUEVO: Verificar condiciones para reportes DESPUÉS de tener los resultados
        self._check_report_conditions(analysis_results)
        
        # Actualizar estadísticas
        self.stats['frames_analyzed'] += 1
        
        return frame, analysis_results
    
    def _handle_operator_change(self, operator_info):
        """Maneja el cambio de operador"""
        self.current_operator = operator_info
        self.logger.info(f"Cambio de operador: {operator_info['name']} ({operator_info['id']})")
        
        # Resetear módulos (con manejo de errores)
        try:
            self.fatigue_detector.reset()
        except:
            pass
            
        try:
            self.stress_analyzer.reset()
        except:
            pass
            
        try:
            self.pulse_estimator.reset()
        except:
            pass
            
        try:
            self.emotion_analyzer.reset()
        except:
            pass
            
        try:
            self.anomaly_detector.reset()
        except:
            pass
        
        # Cargar o crear baseline
        self.is_calibrated = self.calibration_manager.load_or_create_baseline(
            operator_info['id'],
            operator_info['name']
        )
        
        # Si existe baseline, configurar módulos
        if self.is_calibrated:
            baseline = self.calibration_manager.get_current_baseline()
            
            # Configurar cada módulo si tiene el método set_baseline
            if hasattr(self.fatigue_detector, 'set_baseline'):
                self.fatigue_detector.set_baseline(baseline)
                
            if hasattr(self.stress_analyzer, 'set_baseline'):
                self.stress_analyzer.set_baseline(baseline)
                
            self.logger.info(f"Baseline cargado para {operator_info['name']}")
        else:
            self.logger.info(f"Iniciando calibración para {operator_info['name']}")
            self.stats['calibrations_completed'] += 1
        
        # Limpiar historial
        self.analysis_history.clear()
    
    def _evaluate_overall_state(self, analysis_data):
        """
        Evalúa el estado general del operador basado en todos los análisis.
        """
        # Pesos para cada componente
        weights = {
            'fatigue': 0.30,
            'stress': 0.25,
            'emotion': 0.20,
            'anomaly': 0.25
        }
        
        # Calcular score ponderado (0-100, donde 100 es el peor estado)
        risk_score = 0
        
        # Fatiga
        if 'fatigue' in analysis_data:
            fatigue_level = analysis_data['fatigue'].get('fatigue_percentage', 0)
            risk_score += fatigue_level * weights['fatigue']
        
        # Estrés
        if 'stress' in analysis_data:
            stress_level = analysis_data['stress'].get('stress_level', 0)
            risk_score += stress_level * weights['stress']
        
        # Emociones negativas
        if 'emotion' in analysis_data:
            negative_emotions = ['angry', 'sad', 'fear', 'disgust']
            emotion_score = 0
            emotions = analysis_data['emotion'].get('emotions', {})
            for emotion in negative_emotions:
                emotion_score += emotions.get(emotion, 0)
            emotion_score = min(100, emotion_score * 2)  # Amplificar y limitar
            risk_score += emotion_score * weights['emotion']
        
        # Anomalías
        if 'anomaly' in analysis_data:
            anomaly_score = analysis_data['anomaly'].get('anomaly_score', 0)
            risk_score += anomaly_score * weights['anomaly']
        
        # Determinar estado
        if risk_score < 25:
            status = "ÓPTIMO"
            color = (0, 255, 0)  # Verde
        elif risk_score < 50:
            status = "NORMAL"
            color = (0, 255, 255)  # Amarillo
        elif risk_score < 75:
            status = "ATENCIÓN"
            color = (0, 165, 255)  # Naranja
        else:
            status = "CRÍTICO"
            color = (0, 0, 255)  # Rojo
        
        return {
            'score': int(risk_score),
            'status': status,
            'color': color,
            'timestamp': time.time()
        }
    
    def _generate_alerts(self, analysis_data):
        """Genera alertas basadas en los resultados del análisis"""
        alerts = []
        
        # Alertas de fatiga
        if 'fatigue' in analysis_data:
            fatigue = analysis_data['fatigue']
            if fatigue.get('is_critical', False):
                alerts.append({
                    'type': 'fatigue',
                    'level': 'critical',
                    'message': f"Fatiga crítica: {fatigue.get('fatigue_percentage', 0)}%"
                })
            elif fatigue.get('microsleep_detected', False):
                alerts.append({
                    'type': 'microsleep',
                    'level': 'high',
                    'message': f"Microsueños detectados: {fatigue.get('microsleep_count', 0)}"
                })
        
        # Alertas de estrés
        if 'stress' in analysis_data:
            stress = analysis_data['stress']
            if stress.get('stress_level', 0) > 80:
                alerts.append({
                    'type': 'stress',
                    'level': 'high',
                    'message': f"Estrés crítico: {stress.get('stress_level', 0)}%"
                })
        
        # Alertas de pulso
        if 'pulse' in analysis_data:
            pulse = analysis_data['pulse']
            if pulse.get('is_valid', False):
                bpm = pulse.get('bpm', 0)
                if bpm > 120:
                    alerts.append({
                        'type': 'pulse',
                        'level': 'warning',
                        'message': f"Frecuencia cardíaca elevada: {bpm} BPM"
                    })
                elif bpm < 50 and bpm > 0:
                    alerts.append({
                        'type': 'pulse',
                        'level': 'warning',
                        'message': f"Frecuencia cardíaca baja: {bpm} BPM"
                    })
        
        # Alertas de anomalías
        if 'anomaly' in analysis_data:
            anomaly = analysis_data['anomaly']
            if anomaly.get('requires_immediate_attention', False):
                alerts.extend(anomaly.get('alerts', []))
        
        return alerts
    
    def _generate_recommendations(self, analysis_data):
        """Genera recomendaciones basadas en el análisis"""
        recommendations = []
        
        # Recomendaciones por fatiga
        if 'fatigue' in analysis_data:
            fatigue_level = analysis_data['fatigue'].get('fatigue_percentage', 0)
            if fatigue_level > 80:
                recommendations.append("Tome un descanso inmediato de 15-20 minutos")
            elif fatigue_level > 60:
                recommendations.append("Considere tomar un descanso corto")
        
        # Recomendaciones por estrés
        if 'stress' in analysis_data:
            stress_level = analysis_data['stress'].get('stress_level', 0)
            if stress_level > 70:
                recommendations.append("Realice ejercicios de respiración profunda")
        
        # Recomendaciones combinadas
        fatigue = analysis_data.get('fatigue', {}).get('fatigue_percentage', 0)
        stress = analysis_data.get('stress', {}).get('stress_level', 0)
        
        if fatigue > 60 and stress > 60:
            recommendations.insert(0, "Estado crítico: Solicite relevo inmediato")
        
        return recommendations
    
    def _update_history(self, analysis_results):
        """Actualiza el historial de análisis"""
        # Agregar al historial
        self.analysis_history.append({
            'timestamp': analysis_results['timestamp'],
            'fatigue': analysis_results['analysis'].get('fatigue', {}).get('fatigue_percentage', 0),
            'stress': analysis_results['analysis'].get('stress', {}).get('stress_level', 0),
            'overall_score': analysis_results.get('overall_assessment', {}).get('score', 0)
        })
        
        # Limitar tamaño del historial
        if len(self.analysis_history) > self.max_history_size:
            self.analysis_history.pop(0)
    
    def get_analysis_report(self):
        """
        Genera un reporte completo del análisis para el servidor.
        
        Returns:
            dict: Reporte JSON con todos los datos de análisis
        """
        if not self.current_operator:
            return {'status': 'no_operator'}
        
        # Calcular estadísticas del historial
        history_stats = self._calculate_history_stats()
        
        report = {
            'device_info': {
                'analysis_version': '1.0.0',
                'headless_mode': self.headless,
                'uptime_seconds': int(time.time() - self.stats['start_time'])
            },
            'operator': {
                'id': self.current_operator['id'],
                'name': self.current_operator['name'],
                'is_calibrated': self.is_calibrated,
                'calibration_date': self.calibration_manager.get_baseline_date()
            },
            'current_analysis': {},
            'statistics': {
                'frames_analyzed': self.stats['frames_analyzed'],
                'alerts_generated': self.stats['alerts_generated'],
                'calibrations_completed': self.stats['calibrations_completed'],
                'average_fatigue': history_stats['avg_fatigue'],
                'average_stress': history_stats['avg_stress'],
                'trend': history_stats['trend']
            },
            'timestamp': time.time()
        }
        
        # Obtener reportes de cada módulo si tienen el método
        if hasattr(self.fatigue_detector, 'get_fatigue_report_for_server'):
            report['current_analysis']['fatigue'] = self.fatigue_detector.get_fatigue_report_for_server()
        
        if hasattr(self.stress_analyzer, 'get_stress_report_for_server'):
            report['current_analysis']['stress'] = self.stress_analyzer.get_stress_report_for_server()
        elif hasattr(self.stress_analyzer, 'get_stress_report'):
            report['current_analysis']['stress'] = self.stress_analyzer.get_stress_report()
        
        if hasattr(self.pulse_estimator, 'get_pulse_report'):
            report['current_analysis']['pulse'] = self.pulse_estimator.get_pulse_report()
        
        if hasattr(self.emotion_analyzer, 'get_emotion_report'):
            report['current_analysis']['emotion'] = self.emotion_analyzer.get_emotion_report()
        
        if hasattr(self.anomaly_detector, 'get_anomaly_report'):
            report['current_analysis']['anomaly'] = self.anomaly_detector.get_anomaly_report()
        
        return report
    
    def _calculate_history_stats(self):
        """Calcula estadísticas del historial"""
        if not self.analysis_history:
            return {
                'avg_fatigue': 0,
                'avg_stress': 0,
                'trend': 'stable'
            }
        
        # Promedios
        fatigue_values = [h['fatigue'] for h in self.analysis_history]
        stress_values = [h['stress'] for h in self.analysis_history]
        
        avg_fatigue = sum(fatigue_values) / len(fatigue_values)
        avg_stress = sum(stress_values) / len(stress_values)
        
        # Tendencia (últimos 30 vs primeros 30 valores)
        if len(self.analysis_history) > 60:
            recent_avg = sum(fatigue_values[-30:]) / 30
            old_avg = sum(fatigue_values[:30]) / 30
            
            if recent_avg > old_avg + 10:
                trend = 'worsening'
            elif recent_avg < old_avg - 10:
                trend = 'improving'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'avg_fatigue': round(avg_fatigue, 1),
            'avg_stress': round(avg_stress, 1),
            'trend': trend
        }
    
    def enable_analysis(self):
        """Habilita el análisis"""
        self.analysis_enabled = True
        self.logger.info("Análisis habilitado")
    
    def disable_analysis(self):
        """Deshabilita el análisis temporalmente"""
        self.analysis_enabled = False
        self.logger.info("Análisis deshabilitado")
    
    def force_recalibration(self):
        """Fuerza una recalibración del operador actual"""
        if self.current_operator:
            self.logger.info(f"Forzando recalibración para {self.current_operator['name']}")
            return self.calibration_manager.force_recalibration()
        return False
    
    def get_dashboard_config(self):
        """Obtiene la configuración actual del dashboard"""
        if self.dashboard:
            return self.dashboard.get_config()
        return None
    
    def update_dashboard_config(self, config):
        """Actualiza la configuración del dashboard"""
        if self.dashboard:
            self.dashboard.update_config(config)
            return True
        return False
    
    def _check_report_conditions(self, analysis_results):
        """Verifica si se deben generar reportes basados en las condiciones"""
        if not self.report_config['enabled'] or not self.current_operator:
            return
        
        current_time = time.time()
        analysis_data = analysis_results.get('analysis', {})
        
        # 1. Verificar condición crítica: Estrés >= 80% Y Fatiga >= 80%
        if 'stress' in analysis_data and 'fatigue' in analysis_data:
            stress_level = analysis_data['stress'].get('stress_level', 0)
            fatigue_level = analysis_data['fatigue'].get('fatigue_percentage', 0)
            
            if (stress_level >= self.report_config['critical_stress_threshold'] and 
                fatigue_level >= self.report_config['critical_fatigue_threshold']):
                
                # Iniciar o verificar duración
                condition_key = 'critical_stress_fatigue'
                if condition_key not in self.condition_start_times:
                    self.condition_start_times[condition_key] = current_time
                    self.logger.warning(f"Condición crítica iniciada: Estrés {stress_level}%, Fatiga {fatigue_level}%")
                else:
                    duration = current_time - self.condition_start_times[condition_key]
                    if duration >= self.report_config['critical_duration']:
                        # Verificar cooldown
                        if self._check_cooldown(condition_key, current_time):
                            self._generate_critical_report(
                                'critical_stress_fatigue',
                                {
                                    'stress_level': stress_level,
                                    'fatigue_level': fatigue_level,
                                    'duration': duration
                                }
                            )
                            # Reset timer
                            del self.condition_start_times[condition_key]
            else:
                # Condición no se cumple, resetear timer
                if 'critical_stress_fatigue' in self.condition_start_times:
                    del self.condition_start_times['critical_stress_fatigue']
        
        # 2. Verificar pulso anormal
        if 'pulse' in analysis_data:
            pulse_data = analysis_data['pulse']
            if pulse_data.get('is_valid', False):
                bpm = pulse_data.get('bpm', 0)
                
                if (bpm > self.report_config['pulse_high_threshold'] or 
                    (bpm < self.report_config['pulse_low_threshold'] and bpm > 0)):
                    
                    condition_key = 'abnormal_pulse'
                    if condition_key not in self.condition_start_times:
                        self.condition_start_times[condition_key] = current_time
                        self.logger.warning(f"Pulso anormal detectado: {bpm} BPM")
                    else:
                        duration = current_time - self.condition_start_times[condition_key]
                        if duration >= self.report_config['pulse_duration']:
                            if self._check_cooldown(condition_key, current_time):
                                self._generate_critical_report(
                                    'abnormal_pulse',
                                    {
                                        'bpm': bpm,
                                        'duration': duration,
                                        'type': 'high' if bpm > 100 else 'low'
                                    }
                                )
                                del self.condition_start_times[condition_key]
                else:
                    if 'abnormal_pulse' in self.condition_start_times:
                        del self.condition_start_times['abnormal_pulse']
        
        # 3. Verificar anomalías
        if 'anomaly' in analysis_data:
            anomaly_data = analysis_data['anomaly']
            indicators = anomaly_data.get('indicators', {})
            
            # Intoxicación
            if 'intoxication' in indicators:
                level = indicators['intoxication'].get('level', 0)
                self._check_anomaly_condition(
                    'intoxication', level,
                    self.report_config['intoxication_threshold'],
                    self.report_config['intoxication_duration'],
                    current_time
                )
            
            # Neurológico
            if 'neurological' in indicators:
                level = indicators['neurological'].get('level', 0)
                self._check_anomaly_condition(
                    'neurological', level,
                    self.report_config['neurological_threshold'],
                    self.report_config['neurological_duration'],
                    current_time
                )
            
            # Comportamiento errático
            if 'erratic' in indicators:
                level = indicators['erratic'].get('level', 0)
                self._check_anomaly_condition(
                    'erratic', level,
                    self.report_config['erratic_threshold'],
                    self.report_config['erratic_duration'],
                    current_time
                )
    
    def _check_anomaly_condition(self, anomaly_type, level, threshold, duration_required, current_time):
        """Verifica una condición de anomalía específica"""
        condition_key = f'anomaly_{anomaly_type}'
        
        if level >= threshold:
            if condition_key not in self.condition_start_times:
                self.condition_start_times[condition_key] = current_time
                self.logger.warning(f"Anomalía {anomaly_type} iniciada: {level}%")
            else:
                duration = current_time - self.condition_start_times[condition_key]
                if duration >= duration_required:
                    if self._check_cooldown(condition_key, current_time):
                        self._generate_critical_report(
                            f'anomaly_{anomaly_type}',
                            {
                                'type': anomaly_type,
                                'level': level,
                                'duration': duration
                            }
                        )
                        del self.condition_start_times[condition_key]
        else:
            if condition_key in self.condition_start_times:
                del self.condition_start_times[condition_key]
    
    def _check_cooldown(self, report_type, current_time):
        """Verifica si ha pasado suficiente tiempo desde el último reporte"""
        last_time = self.last_report_time.get(report_type, 0)
        if current_time - last_time >= self.report_config['cooldown_seconds']:
            return True
        return False
    
    def _generate_critical_report(self, event_type, event_data):
        """Genera un reporte crítico con fotografía"""
        if not self._current_frame is None and not self.current_operator:
            return
        
        try:
            # Preparar datos del evento
            report_data = {
                'event_type': event_type,
                'event_data': event_data,
                'timestamp': time.time(),
                'operator_id': self.current_operator['id'],
                'operator_name': self.current_operator['name'],
                'analysis_summary': {
                    'frames_analyzed': self.stats['frames_analyzed'],
                    'session_duration': time.time() - self.stats['start_time']
                }
            }
            
            # Generar reporte con frame
            report = self.report_manager.generate_report(
                module_name='analysis',
                event_type=event_type,
                data=report_data,
                frame=self._current_frame if self.report_config['include_frame'] else None,
                operator_info=self.current_operator
            )
            
            if report:
                self.last_report_time[event_type] = time.time()
                self.logger.info(f"Reporte crítico generado: {report['id']}")
                self.stats['alerts_generated'] += 1
                
        except Exception as e:
            self.logger.error(f"Error generando reporte: {e}")
            import traceback
            traceback.print_exc()
    
    def update_report_config(self, new_config):
        """Actualiza la configuración de reportes"""
        self.report_config.update(new_config)
        self.logger.info("Configuración de reportes actualizada")