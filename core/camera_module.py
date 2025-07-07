import cv2
import logging
import time
import threading
import queue
import os
import psutil

# 🆕 NUEVO: Importar sistema de configuración
try:
    from config.config_manager import get_config, is_production
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Sistema de configuración no disponible para CameraModule, usando valores por defecto")

class CameraModule:
    def __init__(self):
        """Inicializa el módulo de cámara con configuración adaptativa"""
        
        # 🆕 NUEVO: Cargar configuración externa (con fallbacks seguros)
        if CONFIG_AVAILABLE:
            # === CONFIGURACIÓN DESDE ARCHIVOS YAML ===
            
            # Configuración básica de cámara
            self.config = {
                'camera_index': get_config('camera.index', 0),
                'width': get_config('camera.width', 640),
                'height': get_config('camera.height', 480),
                'fps': get_config('camera.fps', 30),
                
                # Configuración avanzada
                'brightness': get_config('camera.brightness', 0),
                'contrast': get_config('camera.contrast', 0),
                'saturation': get_config('camera.saturation', 0),
                'exposure': get_config('camera.exposure', -1),
                
                # Buffer y latencia
                'buffer_size': get_config('camera.buffer_size', 1),
                'capture_timeout': get_config('camera.capture_timeout', 5),
                
                # Configuración para Raspberry Pi
                'use_threading': get_config('camera.use_threading', True),
                'warmup_time': get_config('camera.warmup_time', 2),
                
                # Optimización automática
                'auto_optimization': get_config('system.auto_optimization', True),
                'performance_monitoring': get_config('system.performance_monitoring', True),
            }
            
            # 🆕 NUEVO: Detectar si estamos en producción (Raspberry Pi)
            self.is_production = is_production()
            
            print(f"✅ CameraModule - Configuración cargada:")
            print(f"   - Resolución: {self.config['width']}x{self.config['height']}")
            print(f"   - FPS objetivo: {self.config['fps']}")
            print(f"   - Modo: {'PRODUCCIÓN (Pi)' if self.is_production else 'DESARROLLO'}")
            print(f"   - Threading: {self.config['use_threading']}")
            print(f"   - Optimización automática: {self.config['auto_optimization']}")
        else:
            # ✅ FALLBACK: Configuración por defecto
            self.config = {
                'camera_index': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'brightness': 0,
                'contrast': 0,
                'saturation': 0,
                'exposure': -1,
                'buffer_size': 1,
                'capture_timeout': 5,
                'use_threading': True,
                'warmup_time': 2,
                'auto_optimization': True,
                'performance_monitoring': True,
            }
            self.is_production = False  # Asumir desarrollo si no hay config
            print("⚠️ CameraModule usando configuración por defecto")
        
        # Estado del módulo
        self.camera = None
        self.logger = logging.getLogger('CameraModule')
        self.is_initialized = False
        
        # Threading para captura
        self.frame_thread = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_thread = False
        
        # Monitoreo de rendimiento
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0
        self.cpu_usage = 0
        self.temperature = 0
        
        # Optimización adaptativa
        self.performance_history = []
        self.last_optimization_time = 0
        self.optimization_interval = 30  # segundos
        
        # Estado de calidad de imagen
        self.image_quality_score = 100
        self.last_frame_time = 0
        
        print("=== Inicializando Módulo de Cámara ===")
        print(f"Índice de cámara: {self.config['camera_index']}")
        print(f"Resolución objetivo: {self.config['width']}x{self.config['height']}")
        print(f"FPS objetivo: {self.config['fps']}")
        print(f"Tiempo de calentamiento: {self.config['warmup_time']} segundos")
        print(f"Optimización automática: {'Habilitada' if self.config['auto_optimization'] else 'Deshabilitada'}")
    
    def initialize(self):
        """Inicializa la cámara con los parámetros especificados"""
        try:
            self.logger.info("Inicializando cámara...")
            
            # Crear objeto de captura
            self.camera = cv2.VideoCapture(self.config['camera_index'])
            
            if not self.camera.isOpened():
                self.logger.error(f"No se pudo abrir la cámara con índice {self.config['camera_index']}")
                return False
            
            # Configurar parámetros básicos
            self._apply_camera_settings()
            
            # Tiempo de calentamiento para que la cámara se estabilice
            self.logger.info(f"Calentando cámara por {self.config['warmup_time']} segundos...")
            warmup_start = time.time()
            
            # Capturar algunos frames durante el calentamiento
            while time.time() - warmup_start < self.config['warmup_time']:
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Error durante calentamiento de cámara")
                time.sleep(0.1)
            
            # Verificar que podemos capturar frames
            ret, test_frame = self.camera.read()
            if not ret:
                self.logger.error("No se puede capturar frames de la cámara")
                return False
            
            # Verificar resolución actual
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Cámara inicializada: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Iniciar thread de captura si está habilitado
            if self.config['use_threading']:
                self._start_capture_thread()
            
            self.is_initialized = True
            
            # Realizar optimización inicial si está habilitada
            if self.config['auto_optimization']:
                self._optimize_for_hardware()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error al inicializar cámara: {str(e)}")
            return False
    
    def _apply_camera_settings(self):
        """Aplica la configuración a la cámara"""
        try:
            # Configurar resolución
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            
            # Configurar FPS
            self.camera.set(cv2.CAP_PROP_FPS, self.config['fps'])
            
            # Configurar buffer para menor latencia
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, self.config['buffer_size'])
            
            # Configuraciones avanzadas (si están soportadas)
            if self.config['brightness'] != 0:
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.config['brightness'])
            
            if self.config['contrast'] != 0:
                self.camera.set(cv2.CAP_PROP_CONTRAST, self.config['contrast'])
            
            if self.config['saturation'] != 0:
                self.camera.set(cv2.CAP_PROP_SATURATION, self.config['saturation'])
            
            if self.config['exposure'] != -1:
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.config['exposure'])
            
            self.logger.info("Configuración de cámara aplicada")
            
        except Exception as e:
            self.logger.warning(f"Error al aplicar configuración: {str(e)}")
    
    def _start_capture_thread(self):
        """Inicia el hilo de captura de frames"""
        self.stop_thread = False
        self.frame_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.frame_thread.start()
        self.logger.info("Hilo de captura iniciado")
    
    def _capture_frames(self):
        """Hilo que captura frames continuamente"""
        while not self.stop_thread and self.camera is not None:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Limpiar queue viejo si está lleno
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    
                    # Añadir frame nuevo
                    try:
                        self.frame_queue.put(frame, timeout=0.01)
                        self._update_performance_metrics()
                    except queue.Full:
                        pass  # Queue lleno, continuar
                else:
                    time.sleep(0.01)  # Pequeña pausa si no hay frame
                    
            except Exception as e:
                self.logger.error(f"Error en hilo de captura: {str(e)}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Captura y devuelve un frame de la cámara"""
        if not self.is_initialized:
            if not self.initialize():
                return None
        
        try:
            if self.config['use_threading'] and self.frame_thread and self.frame_thread.is_alive():
                # Usar frame del queue si está disponible
                try:
                    frame = self.frame_queue.get(timeout=self.config['capture_timeout'])
                    self.last_frame_time = time.time()
                    return frame
                except queue.Empty:
                    self.logger.warning("Timeout esperando frame del hilo")
                    return None
            else:
                # Captura directa
                ret, frame = self.camera.read()
                if ret:
                    self.last_frame_time = time.time()
                    self._update_performance_metrics()
                    return frame
                else:
                    self.logger.warning("Error al capturar frame directamente")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error al obtener frame: {str(e)}")
            return None
    
    def _update_performance_metrics(self):
        """Actualiza métricas de rendimiento"""
        if not self.config['performance_monitoring']:
            return
            
        current_time = time.time()
        self.frame_count += 1
        
        # Calcular FPS cada segundo
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Actualizar métricas del sistema
            self._update_system_metrics()
            
            # Realizar optimización automática si es necesario
            if (self.config['auto_optimization'] and 
                current_time - self.last_optimization_time > self.optimization_interval):
                self._check_and_optimize()
                self.last_optimization_time = current_time
    
    def _update_system_metrics(self):
        """Actualiza métricas del sistema"""
        try:
            # CPU usage
            self.cpu_usage = psutil.cpu_percent(interval=None)
            
            # Temperatura (solo en Raspberry Pi)
            if self.is_production:
                self.temperature = self._get_pi_temperature()
            
            # Calidad de imagen (estimada)
            self._estimate_image_quality()
            
        except Exception as e:
            self.logger.debug(f"Error actualizando métricas: {str(e)}")
    
    def _get_pi_temperature(self):
        """Obtiene la temperatura de la Raspberry Pi"""
        try:
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    return temp
        except:
            pass
        return 0
    
    def _estimate_image_quality(self):
        """Estima la calidad de la imagen basada en métricas"""
        try:
            # Calcular score basado en FPS y estabilidad
            fps_score = min(100, (self.current_fps / self.config['fps']) * 100)
            
            # Penalizar por CPU alto
            cpu_score = max(0, 100 - self.cpu_usage)
            
            # Penalizar por temperatura alta (solo en Pi)
            temp_score = 100
            if self.is_production and self.temperature > 0:
                if self.temperature > 70:  # Temperatura crítica
                    temp_score = max(0, 100 - (self.temperature - 70) * 5)
                elif self.temperature > 60:  # Temperatura alta
                    temp_score = max(80, 100 - (self.temperature - 60) * 2)
            
            # Score combinado
            self.image_quality_score = int((fps_score * 0.5 + cpu_score * 0.3 + temp_score * 0.2))
            
        except Exception as e:
            self.logger.debug(f"Error estimando calidad: {str(e)}")
    
    def _optimize_for_hardware(self):
        """Optimiza configuración según el hardware detectado"""
        if not self.config['auto_optimization']:
            return
            
        try:
            # Optimizaciones específicas para Raspberry Pi
            if self.is_production:
                self.logger.info("Aplicando optimizaciones para Raspberry Pi...")
                
                # Reducir resolución si CPU está alto
                if self.cpu_usage > 80:
                    new_width = max(320, self.config['width'] - 160)
                    new_height = max(240, self.config['height'] - 120)
                    
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
                    
                    self.logger.info(f"Resolución reducida a {new_width}x{new_height} por CPU alto")
                
                # Reducir FPS si temperatura está alta
                if self.temperature > 65:
                    new_fps = max(5, self.config['fps'] - 5)
                    self.camera.set(cv2.CAP_PROP_FPS, new_fps)
                    self.logger.info(f"FPS reducido a {new_fps} por temperatura alta ({self.temperature:.1f}°C)")
            
            else:
                # Optimizaciones para desarrollo (Windows/Linux con más recursos)
                self.logger.info("Aplicando optimizaciones para desarrollo...")
                
                # Puede usar configuración más alta si recursos están disponibles
                if self.cpu_usage < 50 and self.current_fps >= self.config['fps'] * 0.9:
                    # Sistema con recursos disponibles, mantener calidad alta
                    pass
                    
        except Exception as e:
            self.logger.warning(f"Error en optimización automática: {str(e)}")
    
    def _check_and_optimize(self):
        """Verifica rendimiento y optimiza si es necesario"""
        try:
            # Almacenar métricas en historial
            metrics = {
                'fps': self.current_fps,
                'cpu': self.cpu_usage,
                'temperature': self.temperature,
                'quality': self.image_quality_score,
                'timestamp': time.time()
            }
            
            self.performance_history.append(metrics)
            
            # Mantener solo los últimos 10 registros
            if len(self.performance_history) > 10:
                self.performance_history.pop(0)
            
            # Decidir si optimizar
            should_optimize = False
            
            # FPS muy bajo
            if self.current_fps < self.config['fps'] * 0.7:
                should_optimize = True
                self.logger.warning(f"FPS bajo detectado: {self.current_fps:.1f}/{self.config['fps']}")
            
            # CPU muy alto
            if self.cpu_usage > 85:
                should_optimize = True
                self.logger.warning(f"CPU alto detectado: {self.cpu_usage:.1f}%")
            
            # Temperatura crítica (solo Pi)
            if self.is_production and self.temperature > 70:
                should_optimize = True
                self.logger.warning(f"Temperatura crítica: {self.temperature:.1f}°C")
            
            if should_optimize:
                self._optimize_for_hardware()
                
        except Exception as e:
            self.logger.error(f"Error en verificación de rendimiento: {str(e)}")
    
    def get_status(self):
        """Retorna el estado actual de la cámara"""
        if not self.is_initialized:
            return {
                'initialized': False,
                'error': 'Cámara no inicializada'
            }
        
        try:
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            return {
                'initialized': True,
                'resolution': f"{actual_width}x{actual_height}",
                'target_fps': self.config['fps'],
                'actual_fps': self.current_fps,
                'cpu_usage': self.cpu_usage,
                'temperature': self.temperature,
                'image_quality_score': self.image_quality_score,
                'is_production': self.is_production,
                'threading_enabled': self.config['use_threading'],
                'thread_alive': self.frame_thread.is_alive() if self.frame_thread else False,
                'last_frame_time': self.last_frame_time,
                'optimization_enabled': self.config['auto_optimization']
            }
        except Exception as e:
            return {
                'initialized': True,
                'error': str(e)
            }
    
    def get_performance_report(self):
        """Genera reporte de rendimiento"""
        if not self.performance_history:
            return "No hay datos de rendimiento disponibles"
        
        # Calcular promedios
        avg_fps = sum(m['fps'] for m in self.performance_history) / len(self.performance_history)
        avg_cpu = sum(m['cpu'] for m in self.performance_history) / len(self.performance_history)
        avg_temp = sum(m['temperature'] for m in self.performance_history) / len(self.performance_history)
        avg_quality = sum(m['quality'] for m in self.performance_history) / len(self.performance_history)
        
        report = f"""
=== Reporte de Rendimiento de Cámara ===
FPS Promedio: {avg_fps:.1f}/{self.config['fps']} ({(avg_fps/self.config['fps']*100):.1f}%)
CPU Promedio: {avg_cpu:.1f}%
Temperatura Promedio: {avg_temp:.1f}°C
Calidad de Imagen: {avg_quality:.0f}/100
Modo: {'PRODUCCIÓN (Pi)' if self.is_production else 'DESARROLLO'}
Threading: {'Habilitado' if self.config['use_threading'] else 'Deshabilitado'}
Optimización: {'Habilitada' if self.config['auto_optimization'] else 'Deshabilitada'}
"""
        return report
    
    def update_config(self, new_config):
        """Actualiza la configuración en tiempo de ejecución"""
        try:
            self.config.update(new_config)
            
            # Aplicar cambios que se pueden hacer en tiempo real
            if self.is_initialized and self.camera:
                if 'fps' in new_config:
                    self.camera.set(cv2.CAP_PROP_FPS, self.config['fps'])
                
                if 'brightness' in new_config:
                    self.camera.set(cv2.CAP_PROP_BRIGHTNESS, self.config['brightness'])
                
                if 'contrast' in new_config:
                    self.camera.set(cv2.CAP_PROP_CONTRAST, self.config['contrast'])
                
                if 'exposure' in new_config:
                    self.camera.set(cv2.CAP_PROP_EXPOSURE, self.config['exposure'])
            
            self.logger.info("Configuración de cámara actualizada")
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando configuración: {str(e)}")
            return False
    
    def release(self):
        """Libera los recursos de la cámara"""
        try:
            # Detener hilo de captura
            if self.frame_thread and self.frame_thread.is_alive():
                self.stop_thread = True
                self.frame_thread.join(timeout=2)
                self.logger.info("Hilo de captura detenido")
            
            # Liberar cámara
            if self.camera is not None:
                self.camera.release()
                self.camera = None
                self.logger.info("Cámara liberada")
            
            # Limpiar queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.is_initialized = False
            
            # Mostrar reporte final si hay datos
            if self.performance_history:
                print(self.get_performance_report())
                
        except Exception as e:
            self.logger.error(f"Error liberando recursos: {str(e)}")
    
    def force_restart(self):
        """Fuerza un reinicio de la cámara"""
        self.logger.info("Forzando reinicio de cámara...")
        self.release()
        time.sleep(1)
        return self.initialize()
    
    def capture_test_image(self, filename=None):
        """Captura una imagen de prueba para verificar funcionamiento"""
        try:
            frame = self.get_frame()
            if frame is None:
                return False
            
            if filename is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"camera_test_{timestamp}.jpg"
            
            import cv2
            success = cv2.imwrite(filename, frame)
            
            if success:
                self.logger.info(f"Imagen de prueba guardada: {filename}")
                return filename
            else:
                self.logger.error("Error guardando imagen de prueba")
                return False
                
        except Exception as e:
            self.logger.error(f"Error capturando imagen de prueba: {str(e)}")
            return False