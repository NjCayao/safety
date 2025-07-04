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