import cv2
import os
import time
import logging
import traceback
import dlib
import gc
import psutil
from datetime import datetime
from collections import deque

# 🆕 NUEVO: Importar sistema de configuración Y sincronización
try:
    from config.config_manager import get_config, has_gui, is_development, is_production
    CONFIG_AVAILABLE = True
    print("✅ Sistema de configuración cargado")
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ Sistema de configuración no disponible, usando valores por defecto")

# 🆕 NUEVO: Importar clientes de sincronización
try:
    from sync.config_sync_client import get_config_sync_client
    from sync.heartbeat_sender import get_heartbeat_sender
    from sync.device_auth import get_device_authenticator
    SYNC_AVAILABLE = True
    print("✅ Sistema de sincronización cargado")
except ImportError:
    SYNC_AVAILABLE = False
    print("⚠️ Sistema de sincronización no disponible")

# Importar módulos individuales
from core.camera_module import CameraModule
from core.face_recognition_module import FaceRecognitionModule
from core.fatigue_detection import FatigueDetector
from core.bostezo_detection import BostezosDetector
from core.distraction_detection import DistractionDetector
from core.alarm_module import AlarmModule
from core.behavior_detection_module import BehaviorDetectionModule

# Configuración de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATORS_DIR = os.path.join(BASE_DIR, "operators")
MODEL_DIR = os.path.join(BASE_DIR, "assets/models")
AUDIO_DIR = os.path.join(BASE_DIR, "assets/audio")
REPORTS_DIR = os.path.join(BASE_DIR, "output/reports")
LOGS_DIR = os.path.join(BASE_DIR, "output/logs")

# Asegurar que existan los directorios
for directory in [OPERATORS_DIR, MODEL_DIR, AUDIO_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# 🆕 NUEVO: Configuración de logging con configuración externa
if CONFIG_AVAILABLE:
    log_level = get_config('logging.level', 'INFO')
    log_format = get_config('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    log_level = 'INFO'
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format=log_format,
    filename=os.path.join(LOGS_DIR, f'safety_system_{datetime.now().strftime("%Y%m%d")}.log'),
    filemode='a'
)
logger = logging.getLogger('MainSystem')

class PerformanceOptimizer:
    """
    🆕 NUEVO: Optimizador de rendimiento para Raspberry Pi
    """
    def __init__(self, is_production=False):
        self.is_production = is_production
        self.cpu_threshold = 80 if is_production else 90
        self.memory_threshold = 75 if is_production else 85
        self.temp_threshold = 70  # °C para Raspberry Pi
        
        # Historial de métricas
        self.metrics_history = deque(maxlen=30)  # Últimos 30 segundos
        
        # Estados de optimización
        self.optimization_level = 0  # 0=normal, 1=medium, 2=aggressive
        self.last_optimization_time = 0
        
        # Contadores para scheduling
        self.frame_counter = 0
        
        self.logger = logging.getLogger('PerformanceOptimizer')
        
    def update_metrics(self):
        """Actualiza métricas del sistema"""
        try:
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'temperature': self._get_temperature()
            }
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error actualizando métricas: {str(e)}")
            return None
    
    def _get_temperature(self):
        """Obtiene temperatura del sistema"""
        if not self.is_production:
            return 0
            
        try:
            # Raspberry Pi
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    return int(f.read().strip()) / 1000.0
        except:
            pass
        return 0
    
    def should_optimize(self):
        """Determina si necesita optimizar el rendimiento"""
        if not self.metrics_history:
            return False
            
        current = self.metrics_history[-1]
        
        # Verificar condiciones críticas
        critical_cpu = current['cpu_percent'] > self.cpu_threshold
        critical_memory = current['memory_percent'] > self.memory_threshold
        critical_temp = current['temperature'] > self.temp_threshold
        
        return critical_cpu or critical_memory or critical_temp
    
    def get_optimization_level(self):
        """Determina el nivel de optimización necesario"""
        if not self.metrics_history:
            return 0
            
        current = self.metrics_history[-1]
        level = 0
        
        # Calcular nivel basado en métricas
        if current['cpu_percent'] > self.cpu_threshold:
            level += 1
        if current['memory_percent'] > self.memory_threshold:
            level += 1
        if current['temperature'] > self.temp_threshold:
            level += 1
            
        # Máximo nivel 2
        return min(2, level)
    
    def should_process_detector(self, detector_name, frame_count):
        """
        🎯 SCHEDULER INTELIGENTE: Decide qué detector procesar en cada frame
        """
        optimization_level = self.get_optimization_level()
        
        # Nivel 0: Procesamiento normal (todos los detectores cada frame)
        if optimization_level == 0:
            return True
            
        # Nivel 1: Procesamiento alternado
        elif optimization_level == 1:
            if detector_name == "face_recognition":
                return True  # Siempre procesar reconocimiento facial
            elif detector_name == "fatigue":
                return frame_count % 2 == 0  # Cada 2 frames
            elif detector_name == "behavior":
                return frame_count % 3 == 0  # Cada 3 frames
            elif detector_name == "yawn":
                return frame_count % 2 == 1  # Alternado con fatiga
            elif detector_name == "distraction":
                return frame_count % 4 == 0  # Cada 4 frames
                
        # Nivel 2: Procesamiento muy selectivo
        elif optimization_level == 2:
            if detector_name == "face_recognition":
                return frame_count % 2 == 0  # Reducir incluso reconocimiento facial
            elif detector_name == "fatigue":
                return frame_count % 4 == 0  # Cada 4 frames
            elif detector_name == "behavior":
                return frame_count % 6 == 0  # Cada 6 frames
            elif detector_name == "yawn":
                return frame_count % 5 == 0  # Cada 5 frames
            elif detector_name == "distraction":
                return frame_count % 8 == 0  # Cada 8 frames
                
        return False
    
    def cleanup_memory(self):
        """Limpia memoria proactivamente"""
        try:
            # Forzar garbage collection
            collected = gc.collect()
            self.logger.info(f"Memoria liberada: {collected} objetos")
            
            return True
        except Exception as e:
            self.logger.error(f"Error limpiando memoria: {str(e)}")
            return False
    
    def get_status_report(self):
        """Genera reporte de estado del optimizador"""
        if not self.metrics_history:
            return "Sin métricas disponibles"
            
        current = self.metrics_history[-1]
        opt_level = self.get_optimization_level()
        
        return f"""
=== Estado del Optimizador ===
CPU: {current['cpu_percent']:.1f}% (límite: {self.cpu_threshold}%)
Memoria: {current['memory_percent']:.1f}% (límite: {self.memory_threshold}%)
Temperatura: {current['temperature']:.1f}°C (límite: {self.temp_threshold}°C)
Nivel de optimización: {opt_level}/2
Modo: {'PRODUCCIÓN (Pi)' if self.is_production else 'DESARROLLO'}
"""

class SafetySystem:
    def __init__(self):
        """Inicializa el sistema de seguridad optimizado para Pi"""
        self.logger = logging.getLogger('SafetySystem')
        self.logger.info("Iniciando sistema de seguridad optimizado")

        # 🆕 NUEVO: Cargar configuración y detectar entorno
        if CONFIG_AVAILABLE:
            self.show_gui = has_gui()
            self.is_dev_mode = is_development()
            self.is_prod_mode = is_production()
            self.alert_cooldown = get_config('alerts.cooldown_time', 5)
            
            # 🆕 NUEVO: Configuración específica de optimización
            self.enable_optimization = get_config('system.auto_optimization', True)
            self.performance_monitoring = get_config('system.performance_monitoring', True)
            
            print(f"🔧 Configuración cargada:")
            print(f"   - Modo: {'PRODUCCIÓN (Pi)' if self.is_prod_mode else 'DESARROLLO'}")
            print(f"   - GUI: {'HABILITADA' if self.show_gui else 'DESHABILITADA (headless)'}")
            print(f"   - Optimización: {'HABILITADA' if self.enable_optimization else 'DESHABILITADA'}")
            print(f"   - Monitoreo rendimiento: {'HABILITADO' if self.performance_monitoring else 'DESHABILITADO'}")
        else:
            # ✅ FALLBACK: Valores por defecto
            self.show_gui = True
            self.is_dev_mode = True
            self.is_prod_mode = False
            self.alert_cooldown = 5
            self.enable_optimization = True
            self.performance_monitoring = True
            print("⚠️ Usando configuración por defecto")

        # 🆕 NUEVO: Inicializar optimizador de rendimiento
        self.optimizer = PerformanceOptimizer(self.is_prod_mode) if self.enable_optimization else None
        
        # 🆕 NUEVO: Inicializar clientes de sincronización
        self.config_sync_client = None
        self.heartbeat_sender = None
        self.device_authenticator = None
        
        if SYNC_AVAILABLE:
            try:
                self.config_sync_client = get_config_sync_client()
                self.heartbeat_sender = get_heartbeat_sender()
                self.device_authenticator = get_device_authenticator()
                
                # Configurar callback para cambios de configuración
                self.config_sync_client.add_config_change_callback(self._on_config_changed)
                
                print("🔄 Sincronización inicializada:")
                print(f"   - Device ID: {self.device_authenticator.get_device_id()}")
                print(f"   - Autenticado: {'SÍ' if self.device_authenticator.is_authenticated() else 'NO'}")
                
            except Exception as e:
                self.logger.error(f"Error inicializando sincronización: {e}")
                print(f"⚠️ Error en sincronización: {e}")
        
        # Definir el directorio de reportes
        self.reports_dir = REPORTS_DIR
        
        # Estado del sistema
        self.is_running = False
        self.current_operator = None
        
        # 🆕 NUEVO: Control de frames para scheduling
        self.frame_counter = 0
        self.last_metrics_update = 0
        self.metrics_update_interval = 1.0  # Actualizar métricas cada segundo
        
        # 🆕 NUEVO: Control de reconfiguración dinámica
        self.config_reload_pending = False
        self.last_config_reload = 0
        
        # Tiempos para alertas
        self.last_alert_times = {
            "fatigue": 0, "cell_phone": 0, "cigarette": 0, "unauthorized": 0,
            "multiple_fatigue": 0, "yawn": 0, "multiple_yawns": 0, "distraction": 0,
            "distraction_level1": 0, "distraction_level2": 0, "multiple_distractions": 0,
            "phone_3s": 0, "phone_7s": 0, "smoking_pattern": 0, "smoking_7s": 0
        }
        
        # Estado de comportamientos (para manejar audio)
        self.behavior_audio_played = {"cell_phone": False, "cigarette": False}
        self.behavior_detection_start = {}
        
        # Variable para controlar si se reproducen sonidos del sistema principal
        self.play_system_audio = True
        
        # 🆕 NUEVO: Métricas de rendimiento
        self.performance_stats = {
            'frames_processed': 0,
            'detections_skipped': 0,
            'optimizations_applied': 0,
            'memory_cleanups': 0,
            'config_reloads': 0
        }
        
        # Inicializar módulos
        self.camera = CameraModule()
        self.face_recognizer = FaceRecognitionModule(OPERATORS_DIR)
        
        # Ruta al detector de landmarks faciales
        landmark_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
        
        # Inicializar detectores individuales
        self.face_detector = None
        self.landmark_predictor = None
        self.fatigue_detector = FatigueDetector(landmark_path)
        self.bostezo_detector = BostezosDetector()
        self.distraction_detector = DistractionDetector()
        self.behavior_detector = BehaviorDetectionModule(MODEL_DIR, AUDIO_DIR)
        self.alarm = AlarmModule(AUDIO_DIR)
        
        # 🆕 NUEVO: Configurar ventana solo si GUI está habilitada
        if self.show_gui:
            cv2.namedWindow("Sistema de Seguridad", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sistema de Seguridad", 800, 600)
            print("🖥️ Ventana gráfica configurada")
        else:
            print("🖥️ Modo headless - Sin interfaz gráfica")
    
    def initialize(self):
        """Inicializa todos los módulos del sistema"""
        logger.info("Inicializando módulos del sistema")
        print("Inicializando sistema de seguridad optimizado...")
        
        # Inicializar cámara
        if not self.camera.initialize():
            logger.error("Error al inicializar cámara")
            print("ERROR: No se pudo inicializar la cámara")
            return False
        
        # Cargar operadores
        if not self.face_recognizer.load_operators():
            logger.warning("No se pudieron cargar operadores")
            print("ADVERTENCIA: No se pudieron cargar operadores")
        
        # Inicializar detector facial y predictor de landmarks
        try:
            landmark_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(landmark_path)
            print("Detector facial y predictor de landmarks inicializados correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar detector facial: {str(e)}")
            print(f"ERROR: No se pudo inicializar el detector facial: {str(e)}")
            return False
        
        # Inicializar módulo de alarma
        if not self.alarm.initialize():
            logger.warning("Error al inicializar módulo de alarma")
            print("ADVERTENCIA: No se pudo inicializar el módulo de alarma")
        
        # 🆕 NUEVO: Inicializar sincronización
        if SYNC_AVAILABLE:
            try:
                # Iniciar cliente de configuración
                if self.config_sync_client:
                    self.config_sync_client.start()
                    print("🔄 Cliente de configuración iniciado")
                
                # Iniciar heartbeats
                if self.heartbeat_sender:
                    self.heartbeat_sender.start()
                    print("💓 Heartbeats iniciados")
                
                # Probar conexión inicial
                if self.device_authenticator:
                    test_result = self.device_authenticator.test_connection()
                    if test_result['success']:
                        print(f"🌐 Conexión con servidor: ✅ {test_result['message']}")
                    else:
                        print(f"🌐 Conexión con servidor: ⚠️ {test_result['message']}")
                
            except Exception as e:
                logger.error(f"Error inicializando sincronización: {e}")
                print(f"⚠️ Error en sincronización: {e}")
        
        print("Sistema inicializado correctamente")
        return True
    
    def _on_config_changed(self, old_config, new_config):
        """
        🆕 NUEVO: Callback cuando cambia la configuración remotamente
        """
        try:
            self.logger.info("Configuración actualizada remotamente")
            print("🔄 CONFIGURACIÓN ACTUALIZADA desde servidor")
            
            # Marcar para recarga en el siguiente ciclo
            self.config_reload_pending = True
            self.performance_stats['config_reloads'] += 1
            
            # Mostrar resumen de cambios principales
            self._log_config_changes(old_config, new_config)
            
        except Exception as e:
            self.logger.error(f"Error procesando cambio de configuración: {e}")
    
    def _log_config_changes(self, old_config, new_config):
        """Registra los cambios de configuración principales"""
        try:
            changes = []
            
            # Verificar cambios en secciones críticas
            critical_sections = ['camera', 'fatigue', 'behavior']
            
            for section in critical_sections:
                if section in old_config and section in new_config:
                    old_section = old_config[section]
                    new_section = new_config[section]
                    
                    for key, new_value in new_section.items():
                        old_value = old_section.get(key)
                        if old_value != new_value:
                            changes.append(f"{section}.{key}: {old_value} → {new_value}")
            
            if changes:
                self.logger.info(f"Cambios aplicados: {'; '.join(changes[:5])}")  # Solo primeros 5
                if not self.show_gui:  # En modo headless, mostrar en consola
                    print(f"   Cambios: {'; '.join(changes[:3])}")
            
        except Exception as e:
            self.logger.warning(f"Error registrando cambios de configuración: {e}")
    
    def _should_update_metrics(self, current_time):
        """Determina si debe actualizar métricas de rendimiento"""
        return (self.performance_monitoring and 
                current_time - self.last_metrics_update > self.metrics_update_interval)
    
    def _process_performance_optimization(self, current_time):
        """Procesa optimización de rendimiento"""
        if not self.optimizer:
            return
            
        # Actualizar métricas
        metrics = self.optimizer.update_metrics()
        self.last_metrics_update = current_time
        
        if metrics:
            # Verificar si necesita optimización
            if self.optimizer.should_optimize():
                self.performance_stats['optimizations_applied'] += 1
                
                # Limpiar memoria si es necesario
                if metrics['memory_percent'] > 80:
                    if self.optimizer.cleanup_memory():
                        self.performance_stats['memory_cleanups'] += 1
                
                # Log de optimización (solo en modo development o cada 30 segundos en production)
                if self.is_dev_mode or (current_time % 30 < 1):
                    opt_level = self.optimizer.get_optimization_level()
                    self.logger.info(f"Optimización aplicada - Nivel: {opt_level}, "
                                   f"CPU: {metrics['cpu_percent']:.1f}%, "
                                   f"RAM: {metrics['memory_percent']:.1f}%, "
                                   f"Temp: {metrics['temperature']:.1f}°C")
    
    def _should_process_detector(self, detector_name):
        """Determina si debe procesar un detector específico en este frame"""
        if not self.optimizer:
            return True
            
        should_process = self.optimizer.should_process_detector(detector_name, self.frame_counter)
        
        if not should_process:
            self.performance_stats['detections_skipped'] += 1
            
        return should_process
    
    def _process_config_reload(self):
        """
        🆕 NUEVO: Procesa recarga de configuración si está pendiente
        """
        if not self.config_reload_pending:
            return
        
        current_time = time.time()
        
        # Evitar recargas muy frecuentes
        if current_time - self.last_config_reload < 5:  # Mínimo 5 segundos entre recargas
            return
        
        try:
            self.logger.info("Aplicando recarga de configuración...")
            
            # Recargar configuración en config manager
            if CONFIG_AVAILABLE:
                from config.config_manager import get_config_manager
                config_manager = get_config_manager()
                config_manager.reload()
            
            # Actualizar configuraciones locales que no requieren reinicio
            self.alert_cooldown = get_config('alerts.cooldown_time', 5) if CONFIG_AVAILABLE else 5
            
            # Marcar como procesada
            self.config_reload_pending = False
            self.last_config_reload = current_time
            
            print("🔄 Configuración recargada exitosamente")
            
        except Exception as e:
            self.logger.error(f"Error recargando configuración: {e}")
            print(f"⚠️ Error recargando configuración: {e}")
    
    def start(self):
        """Inicia el sistema de seguridad optimizado"""
        logger.info("Sistema de seguridad optimizado iniciado")
        
        if not self.initialize():
            logger.error("Error al inicializar el sistema")
            return
        
        self.is_running = True
        
        # Para medir FPS
        prev_time = time.time()
        fps_frame_count = 0
        
        # 🆕 NUEVO: Mostrar información de inicio optimizada
        print(f"\n🚀 SISTEMA INICIADO - MODO {'PRODUCCIÓN (Pi)' if self.is_prod_mode else 'DESARROLLO'}")
        print(f"   GUI: {'HABILITADA' if self.show_gui else 'DESHABILITADA'}")
        print(f"   Optimización: {'HABILITADA' if self.enable_optimization else 'DESHABILITADA'}")
        print(f"   Sincronización: {'HABILITADA' if SYNC_AVAILABLE else 'DESHABILITADA'}")
        if self.show_gui:
            print("   Presiona 'q' para salir")
        else:
            print("   Presiona Ctrl+C para salir")
        print("-" * 60)
        
        try:
            while self.is_running:
                try:
                    # Incrementar contador de frames
                    self.frame_counter += 1
                    self.performance_stats['frames_processed'] += 1
                    current_time = time.time()
                    
                    # 🆕 NUEVO: Procesar recarga de configuración si está pendiente
                    self._process_config_reload()
                    
                    # 🆕 NUEVO: Procesar optimización de rendimiento
                    if self._should_update_metrics(current_time):
                        self._process_performance_optimization(current_time)
                    
                    # Capturar frame
                    frame = self.camera.get_frame()
                    
                    if frame is None:
                        logger.error("Error al capturar frame")
                        time.sleep(0.1)
                        continue
                    
                    # Calcular FPS
                    fps_frame_count += 1
                    if current_time - prev_time >= 1.0:
                        fps = fps_frame_count / (current_time - prev_time)
                        fps_frame_count = 0
                        prev_time = current_time
                    else:
                        fps = 0
                    
                    # 🆕 NUEVO: Mostrar FPS solo si GUI está habilitada
                    if self.show_gui:
                        # Mostrar información de optimización si está habilitada
                        if self.optimizer:
                            opt_level = self.optimizer.get_optimization_level()
                            color = (0, 255, 0) if opt_level == 0 else (0, 165, 255) if opt_level == 1 else (0, 0, 255)
                            status_text = f"FPS: {fps:.1f} | Opt: L{opt_level}"
                            
                            # Agregar información de sincronización
                            if SYNC_AVAILABLE and self.device_authenticator:
                                sync_status = "🟢" if self.device_authenticator.is_authenticated() else "🔴"
                                status_text += f" | Sync: {sync_status}"
                            
                            cv2.putText(frame, status_text, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        else:
                            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 🆕 NUEVO: Reconocimiento facial con scheduling inteligente
                    if self._should_process_detector("face_recognition"):
                        if self.current_operator is None:
                            operator = self.face_recognizer.identify_operator(frame)
                            if operator:
                                logger.info(f"Operador reconocido: {operator['name']} (ID: {operator['id']})")
                                self.current_operator = operator
                                print(f"👤 Operador reconocido: {operator['name']}")
                                
                                # Saludar al operador
                                if self.play_system_audio:
                                    self.alarm.play_alarm_threaded("greeting")
                        else:
                            # Verificar cambio de operador cada 5 segundos aproximadamente
                            if self.frame_counter % 150 == 0:  # Asumiendo ~30 FPS
                                operator = self.face_recognizer.identify_operator(frame)
                                if operator and operator['id'] != self.current_operator['id']:
                                    logger.warning(f"Cambio de operador detectado: {operator['name']}")
                                    self.current_operator = operator
                                    print(f"🔄 Cambio de operador: {operator['name']}")
                                    
                                    if self.play_system_audio:
                                        self.alarm.play_alarm_threaded("greeting")
                    
                    # Mostrar información del operador
                    self.face_recognizer.draw_operator_info(frame, self.current_operator)
                    
                    # Detección facial y landmarks (una sola vez, compartido entre detectores)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector(gray, 0)
                    
                    # Solo procesar si se detecta un rostro y hay operador
                    if self.current_operator and faces:
                        landmarks = self.landmark_predictor(gray, faces[0])
                        
                        # 🆕 NUEVO: Procesamiento con scheduling inteligente
                        
                        # Detección de fatiga (microsueños)
                        if self._should_process_detector("fatigue"):
                            fatigue_detected, multiple_fatigue, frame = self.fatigue_detector.detect(frame)
                            self._handle_fatigue_alerts(fatigue_detected, multiple_fatigue, frame, current_time)
                        
                        # Detección de bostezos
                        if self._should_process_detector("yawn"):
                            is_yawning, multiple_yawns = self.bostezo_detector.detect(landmarks, frame)
                            self._handle_yawn_alerts(is_yawning, multiple_yawns, frame, current_time)
                        
                        # Detección de distracciones
                        if self._should_process_detector("distraction"):
                            distraction, multiple_distractions = self.distraction_detector.detect(landmarks, frame)
                            self._handle_distraction_alerts(distraction, multiple_distractions, frame, current_time)
                        
                        # Detección de comportamientos peligrosos
                        if self._should_process_detector("behavior"):
                            face_locations = [(faces[0].top(), faces[0].right(), faces[0].bottom(), faces[0].left())]
                            behaviors, frame, behavior_alerts = self.behavior_detector.detect_behaviors(frame, face_locations)
                            self._handle_behavior_alerts(behavior_alerts, frame, current_time)
                    
                    # 🆕 NUEVO: Mostrar frame solo si GUI está habilitada
                    if self.show_gui:
                        cv2.imshow("Sistema de Seguridad", frame)
                        
                        # Salir si se presiona 'q'
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("👋 Saliendo del sistema...")
                            break
                    else:
                        # En modo headless, pequeña pausa para no saturar CPU
                        time.sleep(0.05)
                        
                        # Log periódico en modo headless
                        if self.frame_counter % 300 == 0:  # Cada ~10 segundos en Pi
                            status = f"📊 Frame: {self.frame_counter} | FPS: {fps:.1f}"
                            if self.optimizer:
                                opt_level = self.optimizer.get_optimization_level()
                                status += f" | Opt: L{opt_level}"
                            if self.current_operator:
                                status += f" | Op: {self.current_operator['name']}"
                            if SYNC_AVAILABLE and self.device_authenticator:
                                sync_status = "✅" if self.device_authenticator.is_authenticated() else "❌"
                                status += f" | Sync: {sync_status}"
                            print(status)
                    
                except Exception as e:
                    logger.error(f"Error en bucle principal: {str(e)}")
                    traceback.print_exc()
                    time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Sistema detenido por el usuario")
            print("\n👋 Sistema detenido por el usuario")
        finally:
            self.stop()
    
    # 🆕 NUEVO: Métodos auxiliares para manejo de alertas (extraídos para mejor organización)
    
    def _handle_fatigue_alerts(self, fatigue_detected, multiple_fatigue, frame, current_time):
        """Maneja alertas de fatiga"""
        if fatigue_detected and current_time - self.last_alert_times.get("fatigue", 0) > self.alert_cooldown:
            logger.warning(f"Fatiga detectada para operador {self.current_operator['name']}")
            print(f"⚠️ FATIGA DETECTADA - {self.current_operator['name']}")
            
            if self.play_system_audio:
                self.alarm.play_alarm_threaded("fatigue")
            
            self.generate_report(frame, "fatigue", self.current_operator)
            self.last_alert_times["fatigue"] = current_time
        
        if multiple_fatigue and current_time - self.last_alert_times.get("multiple_fatigue", 0) > self.alert_cooldown:
            logger.warning(f"Múltiples episodios de fatiga para operador {self.current_operator['name']}")
            print(f"🚨 MÚLTIPLE FATIGA - {self.current_operator['name']}")
            
            if self.play_system_audio:
                self.alarm.play_alarm_threaded("recomendacion")
            
            self.generate_report(frame, "multiple_fatigue", self.current_operator)
            self.last_alert_times["multiple_fatigue"] = current_time
    
    def _handle_yawn_alerts(self, is_yawning, multiple_yawns, frame, current_time):
        """Maneja alertas de bostezos"""
        if is_yawning and current_time - self.last_alert_times.get("yawn", 0) > self.alert_cooldown:
            logger.warning(f"Bostezo detectado para operador {self.current_operator['name']}")
            print(f"🥱 BOSTEZO - {self.current_operator['name']}")
            
            self.generate_report(frame, "yawn", self.current_operator)
            self.last_alert_times["yawn"] = current_time
        
        if multiple_yawns and current_time - self.last_alert_times.get("multiple_yawns", 0) > self.alert_cooldown:
            logger.warning(f"Múltiples bostezos detectados para operador {self.current_operator['name']}")
            print(f"🥱🥱 MÚLTIPLES BOSTEZOS - {self.current_operator['name']}")
            
            self.generate_report(frame, "yawn", self.current_operator)
            self.last_alert_times["multiple_yawns"] = current_time
    
    def _handle_distraction_alerts(self, distraction, multiple_distractions, frame, current_time):
        """Maneja alertas de distracciones"""
        if distraction:
            distraction_status = self.distraction_detector.get_status()
            current_level = distraction_status['current_alert_level']
            
            if current_level == 1 and current_time != self.last_alert_times.get("distraction_level1", 0):
                if current_time - self.last_alert_times.get("distraction", 0) > self.alert_cooldown:
                    logger.warning(f"Distracción nivel 1 detectada para operador {self.current_operator['name']}")
                    print(f"👀 DISTRACCIÓN NIVEL 1 - {self.current_operator['name']}")
                    
                    details = {
                        'direction': distraction_status['direction'],
                        'duration': self.distraction_detector.config['level1_time'],
                        'level': 1
                    }
                    
                    self.generate_report(frame, "distraction", self.current_operator, details)
                    self.last_alert_times["distraction_level1"] = current_time
                    self.last_alert_times["distraction"] = current_time
            
            elif current_level == 2 and current_time != self.last_alert_times.get("distraction_level2", 0):
                if current_time - self.last_alert_times.get("distraction", 0) > self.alert_cooldown:
                    logger.warning(f"Distracción nivel 2 detectada para operador {self.current_operator['name']}")
                    print(f"👀👀 DISTRACCIÓN NIVEL 2 - {self.current_operator['name']}")
                    
                    details = {
                        'direction': distraction_status['direction'],
                        'duration': self.distraction_detector.config['level2_time'],
                        'level': 2
                    }
                    
                    self.generate_report(frame, "distraction", self.current_operator, details)
                    self.last_alert_times["distraction_level2"] = current_time
                    self.last_alert_times["distraction"] = current_time
        
        if multiple_distractions and current_time - self.last_alert_times.get("multiple_distractions", 0) > self.alert_cooldown:
            logger.warning(f"Múltiples distracciones detectadas para operador {self.current_operator['name']}")
            print(f"👀🔄 MÚLTIPLES DISTRACCIONES - {self.current_operator['name']}")
            
            distraction_status = self.distraction_detector.get_status()
            details = {
                'count': distraction_status['total_distractions'],
                'time_window': '10 minutos'
            }
            
            self.generate_report(frame, "multiple_distractions", self.current_operator, details)
            self.last_alert_times["multiple_distractions"] = current_time
    
    def _handle_behavior_alerts(self, behavior_alerts, frame, current_time):
        """Maneja alertas de comportamientos peligrosos"""
        for alert_type, behavior, value in behavior_alerts:
            if current_time - self.last_alert_times.get(alert_type, 0) > self.alert_cooldown:
                logger.warning(f"Alerta de comportamiento: {alert_type} - {behavior}")
                print(f"📱 COMPORTAMIENTO PELIGROSO: {behavior.upper()} - {self.current_operator['name']}")
                
                details = {'behavior': behavior}
                
                if isinstance(value, float):
                    details['duration'] = value
                elif isinstance(value, int):
                    details['count'] = value
                
                self.generate_report(frame, alert_type, self.current_operator, details)
                self.last_alert_times[alert_type] = current_time
                
                if '7s' in alert_type:
                    frame = self.behavior_detector.draw_behavior_alert(frame, behavior, 1.0)
    
    def generate_report(self, frame, alert_type, operator_info=None, details=None):
        """Genera un reporte para una alerta detectada"""
        if self.current_operator is None and alert_type != "unauthorized":
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            operator_id = self.current_operator["id"] if self.current_operator else "unknown"
            
            alert_type_db = alert_type
            if "phone" in alert_type:
                alert_type_db = "phone"
            elif "smoking" in alert_type:
                alert_type_db = "smoking"
            elif "multiple_distractions" in alert_type:
                alert_type_db = "distraction"
            
            filename = f"{operator_id}_{alert_type_db}_{timestamp}.jpg"
            
            # Guardar imagen
            image_path = os.path.join(self.reports_dir, filename)
            cv2.imwrite(image_path, frame)

            # Crear archivo de texto con detalles
            details_file = os.path.join(self.reports_dir, f"{operator_id}_{alert_type_db}_{timestamp}.txt")
            with open(details_file, 'w') as f:
                f.write(f"Reporte de Seguridad - {alert_type.upper()}\n")
                f.write(f"Fecha y hora: {timestamp}\n")
                f.write(f"Operador: {operator_info['name'] if operator_info else 'Desconocido'}\n")
                f.write(f"Frame procesado: {self.frame_counter}\n")
                
                # Información del optimizador si está disponible
                if self.optimizer:
                    f.write(f"Nivel de optimización: {self.optimizer.get_optimization_level()}\n")
                
                # 🆕 NUEVO: Información de sincronización
                if SYNC_AVAILABLE and self.device_authenticator:
                    f.write(f"Device ID: {self.device_authenticator.get_device_id()}\n")
                    f.write(f"Autenticado: {'Sí' if self.device_authenticator.is_authenticated() else 'No'}\n")
                
                # Detalles específicos según el tipo de alerta
                if details:
                    for key, value in details.items():
                        f.write(f"{key.replace('_', ' ').title()}: {value}\n")
                
                f.write("\nAcción recomendada: El operador debe cesar el comportamiento peligroso inmediatamente.\n")
            
            self.logger.info(f"Reporte generado: {image_path} y {details_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error al generar reporte: {str(e)}")
            traceback.print_exc()
            return False
    
    def stop(self):
        """Detiene el sistema y libera recursos"""
        logger.info("Deteniendo sistema optimizado")
        print("🛑 Deteniendo sistema...")
        self.is_running = False
        
        # 🆕 NUEVO: Detener clientes de sincronización
        if SYNC_AVAILABLE:
            try:
                if self.config_sync_client:
                    self.config_sync_client.stop()
                    print("🔄 Cliente de configuración detenido")
                
                if self.heartbeat_sender:
                    self.heartbeat_sender.stop()
                    print("💓 Heartbeats detenidos")
                
            except Exception as e:
                self.logger.error(f"Error deteniendo sincronización: {e}")
        
        self.camera.release()
        
        # 🆕 NUEVO: Solo destruir ventanas si GUI estaba habilitada
        if self.show_gui:
            cv2.destroyAllWindows()
        
        # 🆕 NUEVO: Mostrar estadísticas finales
        if self.performance_stats['frames_processed'] > 0:
            print("\n📊 ESTADÍSTICAS FINALES:")
            print(f"   Frames procesados: {self.performance_stats['frames_processed']}")
            print(f"   Detecciones omitidas (optimización): {self.performance_stats['detections_skipped']}")
            print(f"   Optimizaciones aplicadas: {self.performance_stats['optimizations_applied']}")
            print(f"   Limpiezas de memoria: {self.performance_stats['memory_cleanups']}")
            print(f"   Recargas de configuración: {self.performance_stats['config_reloads']}")
            
            if self.optimizer:
                print(self.optimizer.get_status_report())
        
        print("✅ Sistema detenido correctamente")

if __name__ == "__main__":
    try:
        system = SafetySystem()
        system.start()
    except Exception as e:
        logger.critical(f"Error crítico en el sistema: {str(e)}")
        traceback.print_exc()
        
        input("\nPresiona Enter para salir...")