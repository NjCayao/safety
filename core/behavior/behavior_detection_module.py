import cv2
import numpy as np
import os
import logging
import time
from collections import deque
from core.alarm_module import AlarmModule

# 🆕 NUEVO: Importar sistema de configuración
try:
    from config.config_manager import get_config, has_gui, is_production
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    print("Sistema de configuración no disponible para BehaviorDetectionModule, usando valores por defecto")

class BehaviorDetectionModule:
    def __init__(self, model_dir="assets/models", audio_dir="assets/audio"):
        """
        🚀 FASE 3: Inicializa el módulo optimizado para Raspberry Pi
        """
        self.model_dir = model_dir
        self.audio_dir = audio_dir
        
        self.detection_buffer = {
            "cell phone": deque(maxlen=5),  # Buffer de 5 frames
            "cigarette": deque(maxlen=5)
        }
        self.stable_detections = {}

        # 🆕 NUEVO: Detectar si estamos en producción (Raspberry Pi)
        if CONFIG_AVAILABLE:
            self.is_production = is_production()
            self.show_gui = has_gui()
        else:
            self.is_production = False
            self.show_gui = True
        
        # 🆕 NUEVO: Cargar configuración optimizada para Pi
        if CONFIG_AVAILABLE:
            # === CONFIGURACIÓN OPTIMIZADA PARA RASPBERRY PI ===
            
            # Configuración base
            base_config = {
                'confidence_threshold': get_config('behavior.confidence_threshold', 0.15),
                'night_confidence_threshold': get_config('behavior.night_confidence_threshold', 0.10),
                'night_mode_threshold': get_config('behavior.night_mode_threshold', 50),
                'enable_night_mode': get_config('behavior.enable_night_mode', True),
                'phone_alert_threshold_1': get_config('behavior.phone_alert_threshold_1', 3),
                'phone_alert_threshold_2': get_config('behavior.phone_alert_threshold_2', 7),
                'cigarette_pattern_window': get_config('behavior.cigarette_pattern_window', 30),
                'cigarette_pattern_threshold': get_config('behavior.cigarette_pattern_threshold', 3),
                'cigarette_continuous_threshold': get_config('behavior.cigarette_continuous_threshold', 7),
                'face_proximity_factor': get_config('behavior.face_proximity_factor', 2),
                'detection_timeout': get_config('behavior.detection_timeout', 2.0),
                'audio_enabled': get_config('behavior.audio_enabled', True),
            }
            
            # 🆕 NUEVO: Configuración específica de optimización
            optimization_config = {
                # Optimización de procesamiento
                'enable_optimization': get_config('behavior.enable_optimization', True),
                'processing_interval': get_config('behavior.processing_interval', 2 if self.is_production else 1),
                'roi_enabled': get_config('behavior.roi_enabled', self.is_production),
                'roi_scale_factor': get_config('behavior.roi_scale_factor', 0.6),
                
                # Optimización de YOLO
                'yolo_input_size': get_config('behavior.yolo_input_size', 320 if self.is_production else 416),
                'nms_threshold': get_config('behavior.nms_threshold', 0.3),
                'max_detections': get_config('behavior.max_detections', 10),
                
                # Gestión de memoria
                'memory_optimization': get_config('behavior.memory_optimization', self.is_production),
                'frame_skip_threshold': get_config('behavior.frame_skip_threshold', 3),
                
                # Cache y predicciones
                'enable_prediction_cache': get_config('behavior.enable_prediction_cache', True),
                'cache_size': get_config('behavior.cache_size', 5),
                'similarity_threshold': get_config('behavior.similarity_threshold', 0.8),
            }
            
            # Combinar configuraciones
            self.config = {**base_config, **optimization_config}
            
            print(f"✅ BehaviorDetectionModule - Configuración optimizada cargada:")
            print(f"   - Modo: {'PRODUCCIÓN (Pi)' if self.is_production else 'DESARROLLO'}")
            print(f"   - Tamaño YOLO: {self.config['yolo_input_size']}px")
            print(f"   - Intervalo procesamiento: cada {self.config['processing_interval']} frames")
            print(f"   - ROI habilitado: {self.config['roi_enabled']}")
            print(f"   - Optimización memoria: {self.config['memory_optimization']}")
        else:
            # ✅ FALLBACK para desarrollo sin configuración
            self.config = {
                'confidence_threshold': 0.15, 'night_confidence_threshold': 0.10,
                'night_mode_threshold': 50, 'enable_night_mode': True,
                'phone_alert_threshold_1': 3, 'phone_alert_threshold_2': 7,
                'cigarette_pattern_window': 30, 'cigarette_pattern_threshold': 3,
                'cigarette_continuous_threshold': 7, 'face_proximity_factor': 2,
                'detection_timeout': 2.0, 'audio_enabled': True,
                'enable_optimization': True, 'processing_interval': 1,
                'roi_enabled': False, 'roi_scale_factor': 0.6,
                'yolo_input_size': 416, 'nms_threshold': 0.3, 'max_detections': 10,
                'memory_optimization': False, 'frame_skip_threshold': 3,
                'enable_prediction_cache': True, 'cache_size': 5, 'similarity_threshold': 0.8,
            }
            self.is_production = False
            self.show_gui = True
            print("⚠️ BehaviorDetectionModule usando configuración por defecto")
        
        # 🆕 NUEVO: Variables de optimización
        self.frame_counter = 0
        self.last_processing_frame = 0
        self.processing_times = deque(maxlen=10)
        
        # 🆕 NUEVO: Cache de predicciones para evitar recomputación
        self.prediction_cache = deque(maxlen=self.config['cache_size'])
        self.last_frame_hash = None
        
        # 🆕 NUEVO: ROI (Region of Interest) para reducir área de procesamiento
        self.roi_box = None
        self.roi_frame_counter = 0
        
        # 🆕 NUEVO: Control de memoria
        self.memory_cleanup_counter = 0
        self.memory_cleanup_interval = 100  # Cada 100 frames
        
        # === CONFIGURACIÓN DE COLORES Y AUDIO (igual que antes) ===
        if CONFIG_AVAILABLE:
            self.audio_keys = {
                "phone_3s": get_config('audio.files.telefono', 'telefono'),  # Sin .mp3
                "phone_7s": get_config('audio.files.comportamiento10s', 'comportamiento10s'),  # Sin .mp3
                "smoking_pattern": get_config('audio.files.cigarro', 'cigarro'),  # Sin .mp3
                "smoking_7s": get_config('audio.files.comportamiento10s', 'comportamiento10s')  # Sin .mp3
            }
        else:
            self.audio_keys = {
                "phone_3s": "telefono",  # Sin .mp3
                "phone_7s": "comportamiento10s",  # Sin .mp3
                "smoking_pattern": "cigarro",  # Sin .mp3
                "smoking_7s": "comportamiento10s"  # Sin .mp3
            }
        
        self.colors = {
            "cell phone": (0, 0, 255), "cigarette": (0, 165, 255),
            "text_normal": (255, 255, 255), "text_warning": (0, 165, 255),
            "text_critical": (0, 0, 255), "background": (0, 0, 0),
            "progress_bar_bg": (100, 100, 100), "progress_bar_fill": (0, 165, 255),
        }
        
        # === VARIABLES DE ESTADO INTERNO ===
        self.net = None
        self.classes = None
        self.target_classes = {
            "cell phone": {"id": None, "color": self.colors["cell phone"]},
            "cigarette": {"id": None, "color": self.colors["cigarette"]}
        }
        self.logger = logging.getLogger('BehaviorDetectionModule')
        
        # Inicializar módulo de alarma
        self.alarm = AlarmModule(audio_dir)
        if not self.alarm.initialize():
            self.logger.warning("No se pudo inicializar el módulo de alarma en BehaviorDetection")
        
        # Estado interno del módulo
        self.is_night_mode = False
        self.light_level = 0
        
        # Seguimiento de tiempo para comportamientos
        self.behavior_start_times = {}
        self.behavior_durations = {}
        self.last_detection_times = {}
        
        # Para detección de cigarro por frecuencia
        self.cigarette_detections = deque(maxlen=30)
        
        # Estados de reporte y audio
        self.report_states = {
            "phone_3s": False, "phone_7s": False,
            "smoking_pattern": False, "smoking_7s": False
        }
        self.audio_states = {
            "phone_3s": False, "phone_7s": False,
            "smoking_pattern": False, "smoking_7s": False
        }
        
        # Para logs en modo headless
        self._last_log_time = 0
        
        # Imprimir configuración
        self._print_optimization_config()
    
    def _print_optimization_config(self):
        """Imprime la configuración de optimización"""
        print("=== Detector de Comportamientos - Configuración Optimizada ===")
        print(f"Modo de ejecución: {'PRODUCCIÓN (Raspberry Pi)' if self.is_production else 'DESARROLLO'}")
        print(f"Optimización habilitada: {self.config['enable_optimization']}")
        
        if self.config['enable_optimization']:
            print(f"  • Tamaño de entrada YOLO: {self.config['yolo_input_size']}x{self.config['yolo_input_size']}")
            print(f"  • Intervalo de procesamiento: cada {self.config['processing_interval']} frames")
            print(f"  • ROI (Región de Interés): {'Habilitado' if self.config['roi_enabled'] else 'Deshabilitado'}")
            print(f"  • Cache de predicciones: {'Habilitado' if self.config['enable_prediction_cache'] else 'Deshabilitado'}")
            print(f"  • Gestión de memoria: {'Habilitada' if self.config['memory_optimization'] else 'Deshabilitada'}")
        
        print(f"GUI: {'Habilitada' if self.show_gui else 'Deshabilitada (headless)'}")
        print(f"Audio: {'Habilitado' if self.config['audio_enabled'] else 'Deshabilitado'}")
    
    def initialize(self):
        """🚀 FASE 3: Inicializa el modelo YOLO optimizado para Pi"""
        # Rutas a los archivos del modelo
        config_file = os.path.join(self.model_dir, "yolov3-tiny.cfg")
        weights_file = os.path.join(self.model_dir, "yolov3-tiny.weights")
        classes_file = os.path.join(self.model_dir, "coco.names")
        
        # Verificar archivos
        for file_path, file_name in [
            (config_file, "yolov3-tiny.cfg"),     # Cambiar aquí
            (weights_file, "yolov3-tiny.weights"), # Cambiar aquí
            (classes_file, "coco.names")
        ]:
            if not os.path.exists(file_path):
                self.logger.error(f"No se encontró: {file_path}")
                return False
        
        try:
            # Cargar nombres de clases
            with open(classes_file, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            # Mapear solo las clases que necesitamos (optimización)
            target_class_names = ["cell phone", "bottle"]  # bottle como sustituto de cigarette
            
            for i, class_name in enumerate(self.classes):
                if class_name == "cell phone":
                    self.target_classes["cell phone"]["id"] = i
                elif class_name == "bottle":  # Sustituto para cigarette
                    self.target_classes["cigarette"]["id"] = i
                    self.logger.info("Usando 'bottle' como sustituto para 'cigarette'")
            
            # 🆕 NUEVO: Cargar red con configuración optimizada para Pi
            self.net = cv2.dnn.readNetFromDarknet(config_file, weights_file)
            
            # 🆕 NUEVO: Configurar backend optimizado
            if self.is_production:
                # En Raspberry Pi, usar CPU optimizado
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("YOLO configurado para CPU (Raspberry Pi)")
            else:
                # En desarrollo, usar configuración estándar
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.logger.info("YOLO configurado para desarrollo")
            
            self.logger.info("Modelo YOLO optimizado cargado correctamente")
            return True
            
        except Exception as e:
            self.logger.error(f"Error al cargar modelo YOLO: {str(e)}")
            return False
    
    def _should_process_frame(self):
        """🎯 Determina si debe procesar este frame basado en optimización"""
        self.frame_counter += 1
        
        if not self.config['enable_optimization']:
            return True
        
        # Procesar solo cada N frames según configuración
        return (self.frame_counter - self.last_processing_frame) >= self.config['processing_interval']
    
    def _calculate_frame_similarity(self, frame):
        """📊 Calcula si el frame actual es similar al anterior (para cache)"""
        if not self.config['enable_prediction_cache']:
            return False
        
        # Calcular hash simple del frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (64, 64))  # Pequeño para rapidez
        frame_hash = hash(frame_resized.tobytes())
        
        # Comparar con frame anterior
        if self.last_frame_hash is not None:
            # Si es muy similar al anterior, usar cache
            similarity = abs(frame_hash - self.last_frame_hash) < 1000  # Umbral simple
            if similarity and self.prediction_cache:
                return True
        
        self.last_frame_hash = frame_hash
        return False
    
    def _get_cached_predictions(self):
        """📋 Obtiene predicciones del cache"""
        if self.prediction_cache:
            return self.prediction_cache[-1]  # Última predicción
        return None
    
    def _cache_predictions(self, detections, boxes, confidences):
        """💾 Guarda predicciones en cache"""
        if self.config['enable_prediction_cache']:
            cache_data = {
                'detections': detections.copy(),
                'boxes': [box.copy() for box in boxes],
                'confidences': confidences.copy(),
                'timestamp': time.time()
            }
            self.prediction_cache.append(cache_data)
    
    def _calculate_roi(self, face_locations, frame_shape):
        """🎯 Calcula Región de Interés (ROI) basada en rostros detectados"""
        if not self.config['roi_enabled'] or not face_locations:
            return None
        
        h, w = frame_shape[:2]
        
        # Calcular ROI expandida alrededor de los rostros
        min_x, min_y = w, h
        max_x, max_y = 0, 0
        
        for face_location in face_locations:
            if len(face_location) == 4:  # (top, right, bottom, left)
                top, right, bottom, left = face_location
                min_x = min(min_x, left)
                min_y = min(min_y, top)
                max_x = max(max_x, right)
                max_y = max(max_y, bottom)
        
        # Expandir ROI
        expansion = int(min(w, h) * (1 - self.config['roi_scale_factor']) / 2)
        roi_x1 = max(0, min_x - expansion)
        roi_y1 = max(0, min_y - expansion)
        roi_x2 = min(w, max_x + expansion)
        roi_y2 = min(h, max_y + expansion)
        
        return (roi_x1, roi_y1, roi_x2, roi_y2)
    
    def _memory_cleanup(self):
        """🧹 Limpieza de memoria optimizada"""
        if not self.config['memory_optimization']:
            return
        
        self.memory_cleanup_counter += 1
        
        if self.memory_cleanup_counter >= self.memory_cleanup_interval:
            # Limpiar cache antiguo
            current_time = time.time()
            if self.prediction_cache:
                # Remover predicciones muy antiguas (más de 5 segundos)
                while (self.prediction_cache and 
                       current_time - self.prediction_cache[0].get('timestamp', 0) > 5):
                    self.prediction_cache.popleft()
            
            # Limpiar detecciones de cigarro muy antiguas
            while (self.cigarette_detections and 
                   current_time - self.cigarette_detections[0] > self.config['cigarette_pattern_window'] * 2):
                self.cigarette_detections.popleft()
            
            # Resetear contador
            self.memory_cleanup_counter = 0
            
            # Forzar garbage collection en Pi
            if self.is_production:
                import gc
                gc.collect()
    
    def detect_behaviors(self, frame, face_locations=None):
        """
        🚀 FASE 3: Detecta comportamientos con optimizaciones para Raspberry Pi
        """
        alerts = []
        current_time = time.time()
        
        # 🆕 NUEVO: Verificar si debe procesar este frame
        if not self._should_process_frame():
            # Retornar última detección conocida si existe
            if hasattr(self, '_last_detections'):
                return self._last_detections, frame, []
            else:
                return [], frame, []
        
        # Marcar que procesamos este frame
        self.last_processing_frame = self.frame_counter
        processing_start_time = time.time()
        
        # Verificar que el modelo esté cargado
        if self.net is None or self.classes is None:
            if not self.initialize():
                return [], frame, alerts
        
        # 🆕 NUEVO: Verificar si podemos usar cache
        if self._calculate_frame_similarity(frame):
            cached_predictions = self._get_cached_predictions()
            if cached_predictions:
                self.logger.debug("Usando predicciones del cache")
                return cached_predictions['detections'], frame, []
        
        # Detectar condiciones de iluminación
        if self.config['enable_night_mode']:
            self._detect_lighting_conditions(frame)
        
        # 🆕 NUEVO: Calcular ROI para reducir área de procesamiento
        roi = self._calculate_roi(face_locations, frame.shape) if face_locations else None
        
        # Seleccionar frame o ROI para procesamiento
        if roi and self.config['roi_enabled']:
            roi_x1, roi_y1, roi_x2, roi_y2 = roi
            processing_frame = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            roi_offset = (roi_x1, roi_y1)
        else:
            processing_frame = frame
            roi_offset = (0, 0)
        
        # Mejorar la imagen según las condiciones de iluminación
        enhanced_frame = self._enhance_image(processing_frame)
        
        # 🆕 NUEVO: Usar tamaño de entrada optimizado para Pi
        input_size = self.config['yolo_input_size']
        blob = cv2.dnn.blobFromImage(enhanced_frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False)
        
        # Pasar blob por la red
        self.net.setInput(blob)
        
        # Obtener predicciones
        output_layers_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layers_names)
        
        # Procesar detecciones con optimizaciones
        detections, enhanced_frame = self._process_optimized_detections(
            layer_outputs, processing_frame, roi_offset, current_time
        )

        # Filtrar para quedarse solo con la mejor detección de cada tipo
        detections = self._filter_best_detection(detections)

       
        # 🆕 NUEVO: Guardar en cache
        boxes = []  # Simplificado para cache
        confidences = []
        self._cache_predictions(detections, boxes, confidences)
        
        # Guardar última detección
        # Estabilizar detecciones
        detections = self._stabilize_detections(detections, current_time)
        self._last_detections = detections
        
        # Procesar comportamientos detectados
        detected_behaviors = set(detection[0] for detection in detections)
        
        for behavior in detected_behaviors:
            if behavior == "cell phone":
                alerts.extend(self._process_cellphone_behavior(behavior, current_time))
            elif behavior == "cigarette":
                alerts.extend(self._process_cigarette_behavior(behavior, current_time))
        
        # Limpiar comportamientos no detectados
        self._cleanup_undetected_behaviors(detected_behaviors, current_time)
        
        # 🆕 NUEVO: Dibujar información solo si GUI está habilitada
        if self.show_gui:
            if roi and self.config['roi_enabled']:
                # Dibujar ROI en frame original
                roi_x1, roi_y1, roi_x2, roi_y2 = roi
                cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), 2)
                cv2.putText(frame, "ROI", (roi_x1, roi_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            frame = self._draw_behavior_timers(frame)
            frame = self._draw_optimization_info(frame, processing_start_time)
        else:
            # Log en modo headless
            if current_time - self._last_log_time > 10:
                active_behaviors = list(detected_behaviors)
                mode_str = "NOCHE" if self.is_night_mode else "DÍA"
                opt_info = f"ROI: {'Sí' if roi else 'No'}, Frames: {self.frame_counter}"
                print(f"📊 Comportamientos: {active_behaviors if active_behaviors else 'Ninguno'} | "
                      f"Modo: {mode_str} | {opt_info}")
                self._last_log_time = current_time
        
        # 🆕 NUEVO: Limpieza de memoria periódica
        self._memory_cleanup()
        
        # Registrar tiempo de procesamiento
        processing_time = time.time() - processing_start_time
        self.processing_times.append(processing_time)
        
        return detections, frame, alerts
    
    def _process_optimized_detections(self, layer_outputs, frame, roi_offset, current_time):
        """🎯 Procesa detecciones con optimizaciones para Pi"""
        height, width = frame.shape[:2]
        roi_x_offset, roi_y_offset = roi_offset
        
        # Inicializar listas
        boxes = []
        confidences = []
        class_ids = []
        
        # Determinar umbral de confianza según el modo
        current_threshold = (self.config['night_confidence_threshold'] if self.is_night_mode 
                           else self.config['confidence_threshold'])       
              
        # Procesar solo las clases objetivo (optimización)
        target_ids = [info["id"] for info in self.target_classes.values() if info["id"] is not None]       
        
        # Procesar cada detección
        detection_count = 0
        max_detections = self.config['max_detections']

        # DEBUG: Contador de detecciones
        total_detections = 0
        filtered_by_confidence = 0
        filtered_by_class = 0
        
        for output in layer_outputs:
            for detection in output:
                # 🆕 NUEVO: Límite de detecciones para ahorrar CPU
                if detection_count >= max_detections:
                    break
                
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # DEBUG: Ver qué se está filtrando
                if confidence <= current_threshold:
                    filtered_by_confidence += 1
                elif class_id not in target_ids:
                    filtered_by_class += 1

                # Filtrar por confianza y clases objetivo (solo las que necesitamos)
                if confidence > current_threshold and class_id in target_ids:
                    # Calcular coordenadas del objeto (con offset de ROI)
                    center_x = int(detection[0] * width) + roi_x_offset
                    center_y = int(detection[1] * height) + roi_y_offset
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Coordenadas de la esquina superior izquierda
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    detection_count += 1

        # DEBUG: Imprimir resumen
        # if total_detections > 0:
        #     print(f"DEBUG FILTROS: Total: {total_detections}, Filtrados por confianza: {filtered_by_confidence}, "
        #         f"Filtrados por clase: {filtered_by_class}, Pasaron: {len(boxes)}, Umbral: {current_threshold:.2f}")
        
        # Aplicar non-maximum suppression optimizado
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, current_threshold, self.config['nms_threshold'])
               
        detections = []
        
        # Verificar si hay detecciones
        if len(boxes) > 0 and len(indexes) > 0:
            try:
                # Manejar diferentes formatos de indexes
                if isinstance(indexes, np.ndarray) and indexes.ndim == 1:
                    indexes_flat = indexes
                else:
                    indexes_flat = indexes.flatten()
                
                for i in indexes_flat:
                    x, y, w, h = boxes[i]
                    class_id = class_ids[i]
                    confidence = confidences[i]
                    
                    # Mapear a nombre de clase
                    for target_name, target_info in self.target_classes.items():
                        if target_info["id"] == class_id:
                            # Actualizar tiempos de detección
                            self.last_detection_times[target_name] = current_time
                            
                            # 🆕 NUEVO: Dibujar solo si GUI está habilitada y necesario
                            if self.show_gui:
                                color = target_info["color"]
                                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                
                                # Etiqueta simplificada para mejor rendimiento
                                label = f"{target_name}: {confidence:.1f}"
                                cv2.putText(frame, label, (x, y - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            
                            # Añadir a detecciones
                            detections.append((target_name, confidence))
                            break
                            
            except Exception as e:
                self.logger.error(f"Error al procesar detecciones optimizadas: {str(e)}")
        
        return detections, frame
    
    def _filter_best_detection(self, detections):
        """Filtra y mantiene solo la mejor detección por tipo para evitar duplicados"""
        best_detections = {}
        
        # Agrupar por tipo y quedarse con la de mayor confianza
        for detection in detections:
            behavior_type = detection[0]
            confidence = detection[1]
            
            if behavior_type not in best_detections or confidence > best_detections[behavior_type][1]:
                best_detections[behavior_type] = detection
        
        # Convertir de vuelta a lista
        return list(best_detections.values())
        
    def _draw_optimization_info(self, frame, processing_start_time):
        """📊 Dibuja información de optimización en pantalla"""
        if not self.show_gui:
            return frame
        
        # Calcular tiempo de procesamiento actual
        current_processing_time = time.time() - processing_start_time
        
        # Tiempo promedio de procesamiento
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # Información de optimización (esquina superior derecha)
        h, w = frame.shape[:2]
        info_x = w - 250
        info_y = 30
        
        # Fondo semitransparente
        overlay = frame.copy()
        cv2.rectangle(overlay, (info_x - 10, info_y - 20), (w - 10, info_y + 100), 
                     self.colors["background"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Información de optimización
        opt_info = [
            f"Frame: {self.frame_counter}",
            f"Proc: {current_processing_time*1000:.1f}ms",
            f"Avg: {avg_processing_time*1000:.1f}ms",
            f"ROI: {'ON' if self.config['roi_enabled'] else 'OFF'}",
            f"Cache: {len(self.prediction_cache)}/{self.config['cache_size']}"
        ]
        
        for i, info in enumerate(opt_info):
            cv2.putText(frame, info, (info_x, info_y + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors["text_normal"], 1)
        
        return frame
    
    def _process_cellphone_behavior(self, behavior, current_time):
        """Procesa comportamiento de uso de celular (optimizado)"""
        alerts = []
        
        # Iniciar contador si es primera detección
        if behavior not in self.behavior_start_times:
            self.behavior_start_times[behavior] = current_time
            self.behavior_durations[behavior] = 0
        
        # Actualizar duración
        self.behavior_durations[behavior] = current_time - self.behavior_start_times[behavior]
        duration = self.behavior_durations[behavior]
        
        # Alertas según duración (usando nombres de BD)
        if duration >= self.config['phone_alert_threshold_1'] and not self.report_states["phone_3s"]:
            alerts.append(("phone_3s", behavior, duration))
            self.report_states["phone_3s"] = True

            # Reproducir audio si está habilitado
            if self.config['audio_enabled'] and not self.audio_states["phone_3s"]:
                audio_file = self.audio_keys["phone_3s"]
                # print(f"DEBUG: Intentando reproducir: {audio_file}")
                # print(f"DEBUG: AlarmModule disponible: {self.alarm is not None}")
                
                try:
                    result = self.alarm.play_alarm_threaded(audio_file)
                    # print(f"DEBUG: Resultado de reproducción: {result}")
                except Exception as e:
                    print(f"ERROR reproduciendo audio: {e}")
                    
                self.audio_states["phone_3s"] = True
        
        if duration >= self.config['phone_alert_threshold_2'] and not self.report_states["phone_7s"]:
            alerts.append(("phone_7s", behavior, duration))
            self.report_states["phone_7s"] = True
            
            # Reproducir audio crítico si está habilitado
            if self.config['audio_enabled'] and not self.audio_states["phone_7s"]:
                self.alarm.play_alarm_threaded(self.audio_keys["phone_7s"])
                self.audio_states["phone_7s"] = True
        
        return alerts
    
    def _process_cigarette_behavior(self, behavior, current_time):
        """Procesa comportamiento de fumar (optimizado)"""
        alerts = []
        
        # Agregar detección actual
        self.cigarette_detections.append(current_time)
        
        # Limpiar detecciones antiguas de manera más eficiente
        cutoff_time = current_time - self.config['cigarette_pattern_window']
        while self.cigarette_detections and self.cigarette_detections[0] < cutoff_time:
            self.cigarette_detections.popleft()
        
        # Contar detecciones en ventana
        detection_count = len(self.cigarette_detections)
        
        # Alerta por patrón de frecuencia
        if detection_count >= self.config['cigarette_pattern_threshold'] and not self.report_states["smoking_pattern"]:
            alerts.append(("smoking_pattern", behavior, detection_count))
            self.report_states["smoking_pattern"] = True
            
            if self.config['audio_enabled'] and not self.audio_states["smoking_pattern"]:
                self.alarm.play_alarm_threaded(self.audio_keys["smoking_pattern"])
                self.audio_states["smoking_pattern"] = True
        
        # Mantener duración para detección continua anormal
        if behavior not in self.behavior_start_times:
            self.behavior_start_times[behavior] = current_time
            self.behavior_durations[behavior] = 0
        
        self.behavior_durations[behavior] = current_time - self.behavior_start_times[behavior]
        duration = self.behavior_durations[behavior]
        
        # Alerta por duración continua anormal
        if duration >= self.config['cigarette_continuous_threshold'] and not self.report_states["smoking_7s"]:
            alerts.append(("smoking_7s", behavior, duration))
            self.report_states["smoking_7s"] = True
            
            if self.config['audio_enabled'] and not self.audio_states["smoking_7s"]:
                self.alarm.play_alarm_threaded(self.audio_keys["smoking_7s"])
                self.audio_states["smoking_7s"] = True
        
        return alerts
    
    def _cleanup_undetected_behaviors(self, detected_behaviors, current_time):
        """Limpia comportamientos que ya no se detectan (optimizado)"""
        behaviors_to_remove = []
        
        for behavior in list(self.behavior_start_times.keys()):  # Crear lista para evitar modificación durante iteración
            if behavior not in detected_behaviors:
                # Verificar si ha pasado suficiente tiempo sin detección
                if behavior in self.last_detection_times:
                    time_since_last = current_time - self.last_detection_times[behavior]
                    if time_since_last > self.config['detection_timeout']:
                        behaviors_to_remove.append(behavior)
        
        # Limpiar comportamientos de manera eficiente
        for behavior in behaviors_to_remove:
            # Remover de diccionarios
            self.behavior_start_times.pop(behavior, None)
            self.behavior_durations.pop(behavior, None)
            
            # Resetear estados de manera eficiente
            if behavior == "cell phone":
                states_to_reset = ["phone_3s", "phone_7s"]
            elif behavior == "cigarette":
                states_to_reset = ["smoking_pattern", "smoking_7s"]
            else:
                continue
            
            for state in states_to_reset:
                self.report_states[state] = False
                self.audio_states[state] = False
    
    def _draw_behavior_timers(self, frame):
        """Dibuja contadores de tiempo optimizados"""
        if not self.show_gui:
            return frame
            
        h, w = frame.shape[:2]
        y_offset = int(h * 0.5)
        x_position = w - 220
        
        # Fondo semitransparente más eficiente
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_position - 10, y_offset - 50), 
                     (w - 10, y_offset + 100), self.colors["background"], -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Título
        cv2.putText(frame, "COMPORTAMIENTOS", (x_position, y_offset - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors["text_normal"], 1)
        
        # Mostrar duración de celular
        if "cell phone" in self.behavior_durations:
            duration = self.behavior_durations["cell phone"]
            color = self._get_timer_color(duration, "phone")
            text = f"Celular: {duration:.1f}s"
            cv2.putText(frame, text, (x_position, y_offset + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Mostrar información de cigarro (optimizada)
        if "cigarette" in self.behavior_durations or len(self.cigarette_detections) > 0:
            y_offset += 30
            
            detection_count = len(self.cigarette_detections)
            color = (self.colors["text_critical"] if detection_count >= self.config['cigarette_pattern_threshold'] 
                    else self.colors["text_normal"])
            text = f"Cigarro: {detection_count}/{self.config['cigarette_pattern_threshold']}"
            cv2.putText(frame, text, (x_position, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Duración continua si existe
            if "cigarette" in self.behavior_durations:
                duration = self.behavior_durations["cigarette"]
                if duration > 1.0:
                    y_offset += 20
                    color = self._get_timer_color(duration, "cigarette")
                    text = f"Continuo: {duration:.1f}s"
                    cv2.putText(frame, text, (x_position, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def _get_timer_color(self, duration, behavior_type):
        """Obtiene color según duración y tipo de comportamiento"""
        if behavior_type == "phone":
            if duration < self.config['phone_alert_threshold_1']:
                return self.colors["text_normal"]
            elif duration < self.config['phone_alert_threshold_2']:
                return self.colors["text_warning"]
            else:
                return self.colors["text_critical"]
        elif behavior_type == "cigarette":
            if duration < self.config['cigarette_continuous_threshold']:
                return self.colors["text_normal"]
            else:
                return self.colors["text_critical"]
        else:
            return self.colors["text_normal"]
    
    def _detect_lighting_conditions(self, frame):
        """Detecta condiciones de iluminación (optimizado)"""
        if not self.config['enable_night_mode']:
            return
            
        # Usar submuestreo para calcular iluminación más rápido
        h, w = frame.shape[:2]
        sample_frame = frame[::4, ::4]  # Submuestrear por 4
        
        if len(sample_frame.shape) == 3:
            gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = sample_frame
            
        self.light_level = np.mean(gray)
        
        previous_mode = self.is_night_mode
        self.is_night_mode = self.light_level < self.config['night_mode_threshold']
        
        if previous_mode != self.is_night_mode:
            mode_str = "NOCTURNO (IR)" if self.is_night_mode else "DIURNO"
            self.logger.info(f"Cambio a modo {mode_str} (Nivel de luz: {self.light_level:.1f})")
    
    def _enhance_image(self, frame):
        """Mejora imagen de manera optimizada"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.is_night_mode:
                # Mejoras más ligeras para Pi
                enhanced_gray = cv2.equalizeHist(gray)
                # Blur más ligero
                enhanced_gray = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
                enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            else:
                # CLAHE más ligero para Pi
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
                enhanced_gray = clahe.apply(gray)
                enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        else:
            if self.is_night_mode:
                enhanced = cv2.equalizeHist(frame)
                enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            else:
                clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4,4))
                enhanced = clahe.apply(frame)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def get_optimization_status(self):
        """📊 Obtiene estado de optimización para monitoreo"""
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        return {
            'frames_processed': self.frame_counter,
            'optimization_enabled': self.config['enable_optimization'],
            'processing_interval': self.config['processing_interval'],
            'roi_enabled': self.config['roi_enabled'],
            'cache_enabled': self.config['enable_prediction_cache'],
            'cache_size': len(self.prediction_cache),
            'avg_processing_time_ms': avg_processing_time * 1000,
            'memory_optimization': self.config['memory_optimization'],
            'yolo_input_size': self.config['yolo_input_size'],
            'is_production_mode': self.is_production
        }
    
    def update_optimization_config(self, new_config):
        """Actualiza configuración de optimización en tiempo real"""
        try:
            optimization_keys = [
                'processing_interval', 'roi_enabled', 'roi_scale_factor',
                'enable_prediction_cache', 'cache_size', 'memory_optimization'
            ]
            
            for key in optimization_keys:
                if key in new_config:
                    self.config[key] = new_config[key]
            
            # Ajustar cache si cambió el tamaño
            if 'cache_size' in new_config:
                self.prediction_cache = deque(list(self.prediction_cache), maxlen=new_config['cache_size'])
            
            self.logger.info("Configuración de optimización actualizada")
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando configuración de optimización: {str(e)}")
            return False
    
    def draw_behavior_alert(self, frame, behavior, confidence):
        """Dibuja alerta visual optimizada"""
        if not self.show_gui:
            return frame
            
        # Alerta más simple para mejor rendimiento
        alert_height = 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], alert_height), 
                     self.colors["text_critical"], -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Mensaje simplificado
        alert_text = f"ALERTA: {behavior.upper()}"
        cv2.putText(frame, alert_text, (10, alert_height//2 + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["text_normal"], 2)
        
        return frame
    
    # Mantener métodos de compatibilidad
    def update_config(self, new_config):
        """Actualiza configuración general"""
        self.config.update(new_config)
        self.logger.info("Configuración actualizada")
    
    def get_config(self):
        """Retorna configuración actual"""
        return {**self.config, **self.get_optimization_status()}
    
    def _stabilize_detections(self, detections, current_time):
        """Estabiliza las detecciones para evitar parpadeo"""
        stabilized = []
        
        # Actualizar buffer con detecciones actuales
        detected_types = set(d[0] for d in detections)
        
        for behavior in ["cell phone", "cigarette"]:
            # Agregar True si detectado, False si no
            self.detection_buffer[behavior].append(behavior in detected_types)
            
            # Calcular si la detección es estable (presente en mayoría de frames)
            if len(self.detection_buffer[behavior]) >= 3:
                detection_count = sum(self.detection_buffer[behavior])
                is_stable = detection_count >= 3  # Al menos 3 de 5 frames
                
                if is_stable and behavior in detected_types:
                    # Encontrar la detección actual
                    for det in detections:
                        if det[0] == behavior:
                            stabilized.append(det)
                            self.stable_detections[behavior] = current_time
                            break
                elif behavior in self.stable_detections:
                    # Mantener detección por un tiempo extra
                    time_since_stable = current_time - self.stable_detections[behavior]
                    if time_since_stable < 0.5:  # Mantener por 0.5 segundos
                        stabilized.append((behavior, 0.5))  # Confianza artificial
                    else:
                        self.stable_detections.pop(behavior, None)
        
        return stabilized
