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

# Sistema de configuración
try:
    from config.config_manager import get_config, has_gui, is_development, is_production
    CONFIG_AVAILABLE = True
    print("✅ Sistema de configuración cargado")
except ImportError:
    CONFIG_AVAILABLE = False
    print("⚠️ Sistema de configuración no disponible, usando valores por defecto")

# Clientes de sincronización
try:
    from sync.config_sync_client import get_config_sync_client
    from sync.heartbeat_sender import get_heartbeat_sender
    from sync.device_auth import get_device_authenticator
    SYNC_AVAILABLE = True
    print("✅ Sistema de sincronización cargado")
except ImportError:
    SYNC_AVAILABLE = False
    print("⚠️ Sistema de sincronización no disponible")

# Importar módulos básicos
from core.camera_module import CameraModule
from core.alarm_module import AlarmModule

# NUEVO: Importar sistemas integrados
from core.face_recognition.integrated_face_system import IntegratedFaceSystem
from core.fatigue.integrated_fatigue_system import IntegratedFatigueSystem
from core.behavior.integrated_behavior_system import IntegratedBehaviorSystem
from core.distraction.integrated_distraction_system import IntegratedDistractionSystem
from core.yawn.integrated_yawn_system import IntegratedYawnSystem

# Importar MasterDashboard
from core.master_dashboard import MasterDashboard

# OPCIONAL: Sistema de análisis si está disponible
try:
    from core.analysis.integrated_analysis_system import IntegratedAnalysisSystem
    ANALYSIS_AVAILABLE = True
    print("✅ Sistema de análisis disponible")
except ImportError:
    ANALYSIS_AVAILABLE = False
    print("⚠️ Sistema de análisis no disponible")

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

# Configuración de logging
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
    """Optimizador de rendimiento para Raspberry Pi"""
    def __init__(self, is_production=False):
        self.is_production = is_production
        self.cpu_threshold = 80 if is_production else 90
        self.memory_threshold = 75 if is_production else 85
        self.temp_threshold = 70
        self.metrics_history = deque(maxlen=30)
        self.optimization_level = 0
        self.last_optimization_time = 0
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
        if current['cpu_percent'] > self.cpu_threshold:
            level += 1
        if current['memory_percent'] > self.memory_threshold:
            level += 1
        if current['temperature'] > self.temp_threshold:
            level += 1
        return min(2, level)
    
    def should_process_detector(self, detector_name, frame_count):
        """SCHEDULER INTELIGENTE: Decide qué detector procesar en cada frame"""
        optimization_level = self.get_optimization_level()
        
        if optimization_level == 0:
            return True
        elif optimization_level == 1:
            if detector_name == "face_recognition":
                return True
            elif detector_name == "fatigue":
                return frame_count % 2 == 0
            elif detector_name == "behavior":
                return frame_count % 3 == 0
            elif detector_name == "yawn":
                return frame_count % 2 == 1
            elif detector_name == "distraction":
                return frame_count % 4 == 0
            elif detector_name == "analysis":
                return frame_count % 5 == 0
        elif optimization_level == 2:
            if detector_name == "face_recognition":
                return frame_count % 2 == 0
            elif detector_name == "fatigue":
                return frame_count % 4 == 0
            elif detector_name == "behavior":
                return frame_count % 6 == 0
            elif detector_name == "yawn":
                return frame_count % 5 == 0
            elif detector_name == "distraction":
                return frame_count % 8 == 0
            elif detector_name == "analysis":
                return frame_count % 10 == 0
        return False
    
    def cleanup_memory(self):
        """Limpia memoria proactivamente"""
        try:
            collected = gc.collect()
            self.logger.info(f"Memoria liberada: {collected} objetos")
            return True
        except Exception as e:
            self.logger.error(f"Error limpiando memoria: {str(e)}")
            return False

class SafetySystem:
    def __init__(self):
        """Inicializa el sistema de seguridad con todos los integradores"""
        self.logger = logging.getLogger('SafetySystem')
        self.logger.info("Iniciando sistema de seguridad integrado")

        # Cargar configuración
        if CONFIG_AVAILABLE:
            self.show_gui = has_gui()
            self.is_dev_mode = is_development()
            self.is_prod_mode = is_production()
            self.alert_cooldown = get_config('alerts.cooldown_time', 5)
            self.enable_optimization = get_config('system.auto_optimization', True)
            self.performance_monitoring = get_config('system.performance_monitoring', True)
            
            print(f"🔧 Configuración cargada:")
            print(f"   - Modo: {'PRODUCCIÓN (Pi)' if self.is_prod_mode else 'DESARROLLO'}")
            print(f"   - GUI: {'HABILITADA' if self.show_gui else 'DESHABILITADA (headless)'}")
            print(f"   - Optimización: {'HABILITADA' if self.enable_optimization else 'DESHABILITADA'}")
        else:
            self.show_gui = True
            self.is_dev_mode = True
            self.is_prod_mode = False
            self.alert_cooldown = 5
            self.enable_optimization = True
            self.performance_monitoring = True

        # Inicializar optimizador
        self.optimizer = PerformanceOptimizer(self.is_prod_mode) if self.enable_optimization else None
        
        # Inicializar sincronización
        self.config_sync_client = None
        self.heartbeat_sender = None
        self.device_authenticator = None
        
        if SYNC_AVAILABLE:
            try:
                self.config_sync_client = get_config_sync_client()
                self.heartbeat_sender = get_heartbeat_sender()
                self.device_authenticator = get_device_authenticator()
                print("🔄 Sincronización inicializada")
            except Exception as e:
                self.logger.error(f"Error inicializando sincronización: {e}")
        
        # Estado del sistema
        self.is_running = False
        self.current_operator = None
        self.frame_counter = 0
        self.last_metrics_update = 0
        self.metrics_update_interval = 1.0
        
        # Inicializar módulos básicos
        self.camera = CameraModule()
        
        # Detectores dlib
        self.face_detector = None
        self.landmark_predictor = None
        landmark_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
        
        # NUEVO: Inicializar sistemas integrados
        self.face_system = IntegratedFaceSystem(
            operators_dir=OPERATORS_DIR,
            dashboard_position='right'
        )
        
        self.fatigue_system = IntegratedFatigueSystem(
            operators_dir=OPERATORS_DIR,
            model_path=landmark_path,
            headless=not self.show_gui
        )
        
        self.behavior_system = IntegratedBehaviorSystem(
            model_dir=MODEL_DIR,
            audio_dir=AUDIO_DIR,
            operators_dir=OPERATORS_DIR
        )
        
        self.distraction_system = IntegratedDistractionSystem(
            operators_dir=OPERATORS_DIR,
            dashboard_position='right'
        )
        
        self.yawn_system = IntegratedYawnSystem(
            operators_dir=OPERATORS_DIR,
            dashboard_position='right'
        )
        
        # IMPORTANTE: Desactivar dashboards individuales
        self.face_system.enable_dashboard(False)
        self.distraction_system.enable_dashboard(False)
        self.yawn_system.enable_dashboard(False)
        
        # Sistema de análisis (opcional)
        self.analysis_system = None
        if ANALYSIS_AVAILABLE:
            try:
                self.analysis_system = IntegratedAnalysisSystem(
                    operators_dir=OPERATORS_DIR,
                    headless=not self.show_gui
                )
                print("✅ Sistema de análisis inicializado")
            except Exception as e:
                self.logger.error(f"Error inicializando análisis: {e}")
                self.analysis_system = None
        
        # NUEVO: MasterDashboard con AnalysisDashboard habilitado
        self.master_dashboard = MasterDashboard(
            width=350,
            position='left',
            enable_analysis_dashboard=True  # CAMBIO: Habilitado para ver el dashboard
        )
        
        # Configurar ventana si GUI está habilitada
        if self.show_gui:
            cv2.namedWindow("Sistema de Seguridad", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Sistema de Seguridad", 1200, 800)
            print("🖥️ Ventana gráfica configurada")
        else:
            print("🖥️ Modo headless - Sin interfaz gráfica")
        
        # Estadísticas
        self.performance_stats = {
            'frames_processed': 0,
            'detections_skipped': 0,
            'optimizations_applied': 0,
            'memory_cleanups': 0
        }
    
    def initialize(self):
        """Inicializa todos los módulos del sistema"""
        logger.info("Inicializando módulos del sistema integrado")
        print("Inicializando sistema de seguridad integrado...")
        
        # Inicializar cámara
        if not self.camera.initialize():
            logger.error("Error al inicializar cámara")
            return False
        
        # Inicializar detector facial y predictor de landmarks
        try:
            landmark_path = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(landmark_path)
            print("✅ Detector facial y landmarks inicializados")
        except Exception as e:
            logger.error(f"Error al inicializar detector facial: {str(e)}")
            return False
        
        # Inicializar sincronización si está disponible
        if SYNC_AVAILABLE:
            try:
                if self.config_sync_client:
                    self.config_sync_client.start()
                    print("🔄 Cliente de configuración iniciado")
                
                if self.heartbeat_sender:
                    self.heartbeat_sender.start()
                    print("💓 Heartbeats iniciados")
                
            except Exception as e:
                logger.error(f"Error inicializando sincronización: {e}")
        
        print("✅ Sistema inicializado correctamente")
        return True
    
    def _convert_landmarks_to_dict(self, landmarks):
        """Convierte landmarks de dlib a formato diccionario para el sistema de análisis"""
        landmarks_dict = {
            'chin': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)],  # CAMBIO: 'jaw' → 'chin'
            'left_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)],  # CAMBIO: orden
            'right_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)],
            'nose_bridge': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)],
            'nose_tip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(31, 36)],  # CAMBIO: 'lower_nose' → 'nose_tip'
            'left_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],  # CAMBIO: orden
            'right_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
            'top_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 55)],  # CAMBIO: división correcta
            'bottom_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(54, 60)],  # CAMBIO: división correcta
            'outer_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)],  # Mantener compatibilidad
            'inner_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(60, 68)]   # Mantener compatibilidad
        }
        return landmarks_dict
    
    def start(self):
        """Inicia el sistema de seguridad integrado"""
        logger.info("Sistema de seguridad integrado iniciado")
        
        if not self.initialize():
            logger.error("Error al inicializar el sistema")
            return
        
        self.is_running = True
        
        # Para medir FPS
        prev_time = time.time()
        fps_frame_count = 0
        
        print(f"\n🚀 SISTEMA INICIADO - MODO {'PRODUCCIÓN (Pi)' if self.is_prod_mode else 'DESARROLLO'}")
        print(f"   Presiona {'q' if self.show_gui else 'Ctrl+C'} para salir")
        print("-" * 60)
        
        try:
            while self.is_running:
                try:
                    self.frame_counter += 1
                    self.performance_stats['frames_processed'] += 1
                    current_time = time.time()
                    
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
                    
                    # NUEVO: Procesar con sistemas integrados
                    frame_with_dashboards = self._process_integrated_frame(frame, current_time, fps)
                    
                    # Mostrar frame si GUI está habilitada
                    if self.show_gui:
                        cv2.imshow("Sistema de Seguridad", frame_with_dashboards)
                        
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            print("👋 Saliendo del sistema...")
                            break
                    else:
                        # En modo headless, pequeña pausa
                        time.sleep(0.05)
                        
                        # Log periódico
                        if self.frame_counter % 300 == 0:
                            self._log_headless_status(fps)
                    
                except Exception as e:
                    logger.error(f"Error en bucle principal: {str(e)}")
                    traceback.print_exc()
                    time.sleep(0.1)
            
        except KeyboardInterrupt:
            logger.info("Sistema detenido por el usuario")
            print("\n👋 Sistema detenido por el usuario")
        finally:
            self.stop()
    
    def _process_integrated_frame(self, frame, current_time, fps):
        """
        Procesa un frame con todos los sistemas integrados.
        IMPORTANTE: Aplica dashboards incluso en modo headless para capturas.
        """
        # Crear copia del frame original
        original_frame = frame.copy()
        
        # Variables para resultados
        face_result = None
        fatigue_result = None
        behavior_result = None
        distraction_result = None
        yawn_result = None
        analysis_result = None
        
        # 1. RECONOCIMIENTO FACIAL (siempre se ejecuta)
        if self._should_process_detector("face_recognition"):
            face_result = self.face_system.identify_and_analyze(frame)
            
            # Actualizar frame con dashboard de reconocimiento
            if face_result and 'frame' in face_result:
                frame = face_result['frame']
            
            # Actualizar operador actual
            if face_result and face_result.get('operator_info'):
                operator_info = face_result['operator_info']
                
                if operator_info.get('is_registered', False):
                    # Actualizar operador en todos los sistemas
                    if not self.current_operator or self.current_operator['id'] != operator_info['id']:
                        self.current_operator = operator_info
                        self._update_operator_in_all_systems(operator_info)
                else:
                    # Operador no registrado
                    self.current_operator = None
        
        # Si hay operador registrado, procesar otros análisis
        if self.current_operator:
            # Detectar rostros y landmarks (una sola vez)
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 0)
            
            if faces:
                # Usar el primer rostro detectado
                face = faces[0]
                landmarks = self.landmark_predictor(gray, face)
                face_location = (face.top(), face.right(), face.bottom(), face.left())
                
                # 2. DETECCIÓN DE FATIGA
                if self._should_process_detector("fatigue"):
                    fatigue_result = self.fatigue_system.analyze_frame(frame, landmarks)
                    # El frame ya viene procesado del sistema de fatiga
                    if fatigue_result and 'frame' in fatigue_result:
                        frame = fatigue_result['frame']
                
                # 3. DETECCIÓN DE COMPORTAMIENTOS
                if self._should_process_detector("behavior"):
                    behavior_result = self.behavior_system.analyze_frame(
                        frame, 
                        [face_location]
                    )
                    if behavior_result and 'frame' in behavior_result:
                        frame = behavior_result['frame']
                
                # 4. DETECCIÓN DE DISTRACCIONES
                if self._should_process_detector("distraction"):
                    distraction_result = self.distraction_system.analyze_frame(
                        frame, 
                        landmarks
                    )
                    if distraction_result and 'frame' in distraction_result:
                        frame = distraction_result['frame']
                
                # 5. DETECCIÓN DE BOSTEZOS
                if self._should_process_detector("yawn"):
                    yawn_result = self.yawn_system.analyze_frame(
                        frame, 
                        landmarks
                    )
                    if yawn_result and 'frame' in yawn_result:
                        frame = yawn_result['frame']
                
                # 6. ANÁLISIS AVANZADO (si está disponible)
                if self.analysis_system and self._should_process_detector("analysis"):
                    try:
                        # CONVERTIR LANDMARKS AL FORMATO ESPERADO
                        face_landmarks_dict = self._convert_landmarks_to_dict(landmarks)
                        
                        analysis_result = self.analysis_system.analyze_operator(
                            frame,
                            face_landmarks_dict,  # Usar el diccionario convertido
                            face_location,
                            self.current_operator
                        )
                        # analysis_result retorna (frame, results)
                        if analysis_result:
                            frame = analysis_result[0]
                            analysis_result = analysis_result[1]
                    except Exception as e:
                        self.logger.error(f"Error en análisis: {e}")
                        # Si falla, continuar sin análisis
                        analysis_result = None
        
        # 7. APLICAR MASTER DASHBOARD
        # IMPORTANTE: Esto se hace SIEMPRE, incluso en modo headless
        frame_final = self.master_dashboard.render(
            frame,
            fatigue_result=fatigue_result,
            behavior_result=behavior_result,
            face_result=face_result,
            distraction_result=distraction_result,
            yawn_result=yawn_result,
            analysis_data=analysis_result
        )
        # frame_final = frame
        
        # 8. Agregar información de estado si es necesario
        if self.optimizer:
            opt_level = self.optimizer.get_optimization_level()
            color = (0, 255, 0) if opt_level == 0 else (0, 165, 255) if opt_level == 1 else (0, 0, 255)
            status_text = f"FPS: {fps:.1f} | Opt: L{opt_level}"
            
            if SYNC_AVAILABLE and self.device_authenticator:
                sync_status = "OK" if self.device_authenticator.is_authenticated() else "X"
                status_text += f" | Sync: {sync_status}"
            
            cv2.putText(frame_final, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # IMPORTANTE: El frame_final tiene TODOS los dashboards aplicados
        # Esto es lo que se mostrará en GUI Y lo que se guardará en reportes
        return frame_final
    
    def _update_operator_in_all_systems(self, operator_info):
        """Actualiza el operador en todos los sistemas"""
        self.logger.info(f"Actualizando operador en todos los sistemas: {operator_info['name']}")
        
        # Actualizar en cada sistema
        self.fatigue_system.set_operator(operator_info)
        self.behavior_system.set_operator(operator_info)
        self.distraction_system.set_operator(operator_info)
        self.yawn_system.set_operator(operator_info)
        
        if self.analysis_system:
            # El sistema de análisis no tiene set_operator, 
            # se actualiza automáticamente en analyze_operator
            pass
    
    def _should_process_detector(self, detector_name):
        """Determina si debe procesar un detector específico"""
        if not self.optimizer:
            return True
        
        should_process = self.optimizer.should_process_detector(
            detector_name, 
            self.frame_counter
        )
        
        if not should_process:
            self.performance_stats['detections_skipped'] += 1
        
        return should_process
    
    def _log_headless_status(self, fps):
        """Log de estado para modo headless"""
        status = f"📊 Frame: {self.frame_counter} | FPS: {fps:.1f}"
        
        if self.optimizer:
            opt_level = self.optimizer.get_optimization_level()
            status += f" | Opt: L{opt_level}"
        
        if self.current_operator:
            status += f" | Op: {self.current_operator['name']}"
        else:
            status += " | Op: NO REGISTRADO"
        
        if SYNC_AVAILABLE and self.device_authenticator:
            sync_status = "✅" if self.device_authenticator.is_authenticated() else "❌"
            status += f" | Sync: {sync_status}"
        
        print(status)
    
    def stop(self):
        """Detiene el sistema y libera recursos"""
        logger.info("Deteniendo sistema integrado")
        print("🛑 Deteniendo sistema...")
        self.is_running = False
        
        # Detener sincronización
        if SYNC_AVAILABLE:
            try:
                if self.config_sync_client:
                    self.config_sync_client.stop()
                
                if self.heartbeat_sender:
                    self.heartbeat_sender.stop()
                    
            except Exception as e:
                self.logger.error(f"Error deteniendo sincronización: {e}")
        
        # Resetear sistemas integrados
        self.face_system.reset()
        self.fatigue_system.reset()
        self.behavior_system.reset()
        self.distraction_system.reset()
        self.yawn_system.reset()
        
        if self.analysis_system:
            # El sistema de análisis no tiene reset()
            pass
        
        # Liberar cámara
        self.camera.release()
        
        # Destruir ventanas si GUI estaba habilitada
        if self.show_gui:
            cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        if self.performance_stats['frames_processed'] > 0:
            print("\n📊 ESTADÍSTICAS FINALES:")
            print(f"   Frames procesados: {self.performance_stats['frames_processed']}")
            print(f"   Detecciones omitidas: {self.performance_stats['detections_skipped']}")
            print(f"   Optimizaciones aplicadas: {self.performance_stats['optimizations_applied']}")
            print(f"   Limpiezas de memoria: {self.performance_stats['memory_cleanups']}")
        
        print("✅ Sistema detenido correctamente")

if __name__ == "__main__":
    try:
        system = SafetySystem()
        system.start()
    except Exception as e:
        logger.critical(f"Error crítico en el sistema: {str(e)}")
        traceback.print_exc()
        input("\nPresiona Enter para salir...")