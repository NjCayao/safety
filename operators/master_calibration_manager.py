"""
Master Calibration Manager
=========================
Coordina la calibración de todos los módulos desde un punto central.
"""
import os
import sys
import json
import cv2
import dlib
import numpy as np
from datetime import datetime
import logging
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar calibradores específicos cuando estén listos
from core.fatigue.fatigue_calibration import FatigueCalibration
from core.behavior.behavior_calibration import BehaviorCalibration
from core.analysis.analysis_calibration import AnalysisCalibration
from core.face_recognition.face_recognition_calibration import FaceRecognitionCalibration
# from core.distraction.distraction_calibration import DistractionCalibration
# from core.yawn.yawn_calibration import YawnCalibration

class MasterCalibrationManager:
    def __init__(self, operators_dir="operators", model_path="assets/models/shape_predictor_68_face_landmarks.dat"):
        """
        Inicializa el gestor maestro de calibración.
        
        Args:
            operators_dir: Directorio base de operadores
            model_path: Ruta al modelo de landmarks
        """
        self.operators_dir = operators_dir
        self.baseline_dir = os.path.join(operators_dir, "baseline-json")
        self.model_path = model_path
        self.logger = logging.getLogger('MasterCalibrationManager')
        
        # Detectores para procesamiento de imágenes
        self.face_detector = None
        self.landmark_predictor = None
        
        # Calibradores específicos de cada módulo
        self.calibrators = {
            'fatigue': FatigueCalibration(self.baseline_dir),
            'behavior': BehaviorCalibration(self.baseline_dir),
            'analysis': AnalysisCalibration(self.baseline_dir),
            'face_recognition': FaceRecognitionCalibration(self.baseline_dir),
            # 'distraction': DistractionCalibration(self.baseline_dir),
            # 'yawn': YawnCalibration(self.baseline_dir)
        }
        
        # Buffer para métricas extraídas
        self.extracted_data = defaultdict(list)
        
        # Inicializar componentes
        self._initialize_detectors()
        
    def _initialize_detectors(self):
        """Inicializa detectores de rostro y landmarks"""
        try:
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(self.model_path)
            self.logger.info("Detectores maestros inicializados correctamente")
            return True
        except Exception as e:
            self.logger.error(f"Error inicializando detectores: {e}")
            return False
    
    def calibrate_all_operators(self, photos_base_path="server/operator-photo"):
        """
        Calibra todos los operadores encontrados en el directorio de fotos.
        
        Args:
            photos_base_path: Ruta base donde están las fotos de operadores
            
        Returns:
            dict: Resumen de calibraciones realizadas
        """
        self.logger.info("Iniciando calibración masiva de operadores")
        
        results = {
            'successful': [],
            'failed': [],
            'total_processed': 0
        }
        
        # Verificar que existe el directorio
        if not os.path.exists(photos_base_path):
            self.logger.error(f"No existe el directorio: {photos_base_path}")
            return results
        
        # Procesar cada carpeta de operador
        for operator_id in os.listdir(photos_base_path):
            operator_path = os.path.join(photos_base_path, operator_id)
            
            # Verificar que es un directorio
            if not os.path.isdir(operator_path):
                continue
            
            # Verificar que tiene info.txt
            info_file = os.path.join(operator_path, "info.txt")
            if not os.path.exists(info_file):
                self.logger.warning(f"Operador {operator_id} sin info.txt, omitiendo")
                continue
            
            # Leer nombre del operador
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    operator_name = f.readline().strip()
                    if operator_name.startswith("Operador:"):
                        operator_name = operator_name.replace("Operador:", "").strip()
            except:
                operator_name = "Desconocido"
            
            self.logger.info(f"Procesando operador: {operator_name} (ID: {operator_id})")
            
            # Calibrar este operador
            success = self.calibrate_operator(operator_id, operator_path, operator_name)
            
            if success:
                results['successful'].append(operator_id)
            else:
                results['failed'].append(operator_id)
            
            results['total_processed'] += 1
        
        # Resumen final
        self.logger.info(f"Calibración completa: {len(results['successful'])} exitosas, {len(results['failed'])} fallidas")
        
        return results
    
    def calibrate_operator(self, operator_id, photos_path, operator_name=None):
        """
        Calibra un operador específico procesando sus fotos.
        
        Args:
            operator_id: DNI/ID del operador
            photos_path: Ruta a las fotos del operador
            operator_name: Nombre del operador (opcional)
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        self.logger.info(f"Iniciando calibración para operador {operator_id}")
        
        # Resetear buffer de datos
        self.extracted_data.clear()
        
        # Procesar cada foto
        photos_processed = 0
        photo_files = [f for f in os.listdir(photos_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not photo_files:
            self.logger.error(f"No se encontraron fotos en {photos_path}")
            return False
        
        # Procesar cada foto
        for photo_file in photo_files:
            photo_path = os.path.join(photos_path, photo_file)
            
            try:
                # Cargar imagen
                image = cv2.imread(photo_path)
                if image is None:
                    self.logger.warning(f"No se pudo cargar: {photo_file}")
                    continue
                
                # Extraer todas las métricas necesarias
                success = self._extract_all_metrics(image)
                
                if success:
                    photos_processed += 1
                    self.logger.debug(f"Procesada foto {photo_file}")
                else:
                    self.logger.warning(f"No se detectó rostro en {photo_file}")
                    
            except Exception as e:
                self.logger.error(f"Error procesando {photo_file}: {e}")
                continue
        
        # Verificar que se procesaron suficientes fotos
        if photos_processed < 2:
            self.logger.error(f"Solo se procesaron {photos_processed} fotos, se necesitan al menos 2")
            return False
        
        # Generar calibración maestra
        master_calibration = self._generate_master_calibration(
            operator_id, 
            operator_name or "Desconocido",
            photos_processed
        )
        
        # Guardar calibración maestra
        if not self._save_master_calibration(operator_id, master_calibration):
            return False
        
        # Generar calibraciones específicas de cada módulo
        success_count = 0
        
        for module_name, calibrator in self.calibrators.items():
            try:
                self.logger.info(f"Generando calibración de {module_name}")
                
                # Cada calibrador recibe los datos ya extraídos
                module_success = self._calibrate_module(
                    module_name,
                    calibrator,
                    operator_id,
                    self.extracted_data,
                    photos_processed
                )
                
                if module_success:
                    success_count += 1
                    
            except Exception as e:
                self.logger.error(f"Error calibrando módulo {module_name}: {e}")
        
        # Éxito si al menos un módulo se calibró correctamente
        return success_count > 0
    
    def _extract_all_metrics(self, image):
        """
        Extrae TODAS las métricas necesarias de una imagen.
        
        Args:
            image: Imagen a procesar
            
        Returns:
            bool: True si se extrajeron métricas exitosamente
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_detector(gray, 0)
        
        if not faces:
            return False
        
        # Usar el primer rostro detectado
        face = faces[0]
        
        # Extraer landmarks
        landmarks = self.landmark_predictor(gray, face)
        
        # Convertir landmarks a array numpy para facilitar procesamiento
        landmarks_array = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # Extraer todas las métricas necesarias
        metrics = {}
        
        # 1. Métricas generales del rostro
        metrics['face_width'] = face.width()
        metrics['face_height'] = face.height()
        metrics['face_area'] = face.width() * face.height()
        
        # 2. Métricas de ojos (para fatiga)
        left_eye = landmarks_array[36:42]
        right_eye = landmarks_array[42:48]
        
        metrics['left_ear'] = self._calculate_ear(left_eye)
        metrics['right_ear'] = self._calculate_ear(right_eye)
        metrics['avg_ear'] = (metrics['left_ear'] + metrics['right_ear']) / 2
        
        # 3. Métricas de boca (para bostezos)
        mouth_points = landmarks_array[48:68]
        metrics['mar'] = self._calculate_mar(mouth_points)
        metrics['mouth_width'] = np.linalg.norm(mouth_points[0] - mouth_points[6])
        metrics['mouth_height'] = np.linalg.norm(mouth_points[3] - mouth_points[9])
        
        # 4. Métricas de orientación de cabeza (para distracciones)
        nose_tip = landmarks_array[30]
        chin = landmarks_array[8]
        left_eye_corner = landmarks_array[36]
        right_eye_corner = landmarks_array[45]
        
        metrics['head_tilt'] = self._calculate_head_tilt(nose_tip, chin)
        metrics['head_rotation'] = self._calculate_head_rotation(
            left_eye_corner, right_eye_corner, nose_tip
        )
        
        # 5. Métricas de distancias (para comportamiento)
        metrics['eye_distance'] = np.linalg.norm(left_eye_corner - right_eye_corner)
        metrics['nose_to_mouth'] = np.linalg.norm(nose_tip - mouth_points[3])
        
        # 6. Métricas de expresión facial
        left_eyebrow = landmarks_array[17:22]
        right_eyebrow = landmarks_array[22:27]
        
        metrics['eyebrow_distance'] = np.mean([
            np.linalg.norm(left_eyebrow[i] - left_eye[i]) 
            for i in range(min(len(left_eyebrow), len(left_eye)))
        ])
        
        # 7. Nivel de iluminación
        metrics['light_level'] = np.mean(gray)
        
        # 8. Guardar landmarks completos para referencia
        metrics['landmarks'] = landmarks_array.tolist()
        
        # Agregar al buffer
        self.extracted_data['metrics'].append(metrics)
        self.extracted_data['timestamps'].append(datetime.now().isoformat())
        
        return True
    
    def _generate_master_calibration(self, operator_id, operator_name, photos_processed):
        """
        Genera la calibración maestra con estadísticas generales.
        
        Args:
            operator_id: ID del operador
            operator_name: Nombre del operador
            photos_processed: Número de fotos procesadas
            
        Returns:
            dict: Calibración maestra
        """
        # Calcular estadísticas de todas las métricas
        all_metrics = self.extracted_data['metrics']
        
        # Crear estructura de calibración maestra
        master_calibration = {
            'operator_id': operator_id,
            'operator_name': operator_name,
            'calibration_info': {
                'created_at': datetime.now().isoformat(),
                'photos_processed': photos_processed,
                'version': '2.0',  # Nueva versión con calibración unificada
                'model_used': os.path.basename(self.model_path)
            },
            'face_metrics': {},
            'statistics': {}
        }
        
        # Calcular estadísticas para cada métrica
        metric_names = all_metrics[0].keys() if all_metrics else []
        
        for metric_name in metric_names:
            if metric_name == 'landmarks':  # Saltar landmarks completos
                continue
                
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            
            if values and all(isinstance(v, (int, float)) for v in values):
                master_calibration['statistics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'median': float(np.median(values))
                }
        
        # Guardar landmarks de referencia (promedio)
        if all_metrics and 'landmarks' in all_metrics[0]:
            all_landmarks = [np.array(m['landmarks']) for m in all_metrics]
            avg_landmarks = np.mean(all_landmarks, axis=0)
            master_calibration['reference_landmarks'] = avg_landmarks.tolist()
        
        # Información de calidad de calibración
        master_calibration['quality_metrics'] = {
            'consistency_score': self._calculate_consistency_score(all_metrics),
            'completeness': len(all_metrics) / 4.0,  # Asumiendo 4 fotos ideales
            'lighting_variation': np.std([m['light_level'] for m in all_metrics]) if all_metrics else 0
        }
        
        return master_calibration
    
    def _save_master_calibration(self, operator_id, calibration_data):
        """Guarda la calibración maestra"""
        try:
            # Crear directorio del operador
            operator_dir = os.path.join(self.baseline_dir, operator_id)
            os.makedirs(operator_dir, exist_ok=True)
            
            # Guardar archivo
            master_path = os.path.join(operator_dir, "master_baseline.json")
            with open(master_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Calibración maestra guardada en {master_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error guardando calibración maestra: {e}")
            return False
    
    def _calibrate_module(self, module_name, calibrator, operator_id, extracted_data, photos_processed):
        """
        Ejecuta la calibración específica de un módulo.
        
        Args:
            module_name: Nombre del módulo
            calibrator: Instancia del calibrador específico
            operator_id: ID del operador
            extracted_data: Datos ya extraídos
            photos_processed: Número de fotos procesadas
            
        Returns:
            bool: True si la calibración fue exitosa
        """
        try:
            # Preparar datos específicos para el módulo
            module_data = self._prepare_module_data(module_name, extracted_data)
            
            # Llamar al método de calibración específico del módulo
            # Cada módulo debe implementar: calibrate_from_extracted_data(operator_id, data, photos_count)
            if hasattr(calibrator, 'calibrate_from_extracted_data'):
                success = calibrator.calibrate_from_extracted_data(
                    operator_id,
                    module_data,
                    photos_processed
                )
                return success
            else:
                self.logger.warning(f"Calibrador {module_name} no implementa calibrate_from_extracted_data")
                return False
                
        except Exception as e:
            self.logger.error(f"Error en calibración de {module_name}: {e}")
            return False
    
    def _prepare_module_data(self, module_name, extracted_data):
        """Prepara datos específicos para cada módulo"""
        all_metrics = extracted_data['metrics']
        
        if module_name == 'fatigue':
            # Extraer solo métricas relevantes para fatiga
            return {
                'ear_values': [m['avg_ear'] for m in all_metrics],
                'left_ear_values': [m['left_ear'] for m in all_metrics],
                'right_ear_values': [m['right_ear'] for m in all_metrics],
                'light_levels': [m['light_level'] for m in all_metrics],
                'eye_distances': [m['eye_distance'] for m in all_metrics]
            }
        
        elif module_name == 'face_recognition':
            # Métricas para reconocimiento facial            
            # Calcular variabilidad de encodings faciales
            face_encodings_std = []
            if len(all_metrics) > 1:
                # Simular variabilidad basada en cambios en landmarks
                for i in range(1, len(all_metrics)):
                    if 'landmarks' in all_metrics[i] and 'landmarks' in all_metrics[i-1]:
                        # Calcular diferencia entre landmarks consecutivos
                        landmarks1 = np.array(all_metrics[i-1]['landmarks'])
                        landmarks2 = np.array(all_metrics[i]['landmarks'])
                        diff = np.mean(np.abs(landmarks1 - landmarks2))
                        face_encodings_std.append(diff / 100)  # Normalizar
            
            return {
                'face_areas': [m['face_area'] for m in all_metrics],
                'light_levels': [m['light_level'] for m in all_metrics],
                'face_encodings_std': face_encodings_std
            }
            
        elif module_name == 'yawn':
            # Métricas para bostezos
            return {
                'mar_values': [m['mar'] for m in all_metrics],
                'mouth_widths': [m['mouth_width'] for m in all_metrics],
                'mouth_heights': [m['mouth_height'] for m in all_metrics]
            }
            
        elif module_name == 'distraction':
            # Métricas para distracciones
            return {
                'head_tilts': [m['head_tilt'] for m in all_metrics],
                'head_rotations': [m['head_rotation'] for m in all_metrics],
                'face_widths': [m['face_width'] for m in all_metrics]
            }
            
        elif module_name == 'behavior':
            # Métricas para comportamiento
            return {
                'face_areas': [m['face_area'] for m in all_metrics],
                'nose_to_mouth': [m['nose_to_mouth'] for m in all_metrics],
                'reference_landmarks': extracted_data.get('reference_landmarks', [])
            }
        
        elif module_name == 'analysis':
            # Para análisis integrado, enviar métricas específicas procesadas
            return {
                # Métricas básicas
                'ear_values': [m['avg_ear'] for m in all_metrics],
                'left_ear_values': [m['left_ear'] for m in all_metrics],
                'right_ear_values': [m['right_ear'] for m in all_metrics],
                'mar_values': [m['mar'] for m in all_metrics],
                'eyebrow_distances': [m['eyebrow_distance'] for m in all_metrics],
                'face_widths': [m['face_width'] for m in all_metrics],
                'face_heights': [m['face_height'] for m in all_metrics],
                'light_levels': [m['light_level'] for m in all_metrics],
                
                # Métricas adicionales para análisis avanzado
                'nose_to_mouth': [m['nose_to_mouth'] for m in all_metrics],
                'eye_distances': [m['eye_distance'] for m in all_metrics],
                'mouth_widths': [m['mouth_width'] for m in all_metrics],
                'mouth_heights': [m['mouth_height'] for m in all_metrics],
                
                # Datos completos por si se necesitan
                'metrics': all_metrics,
                'timestamps': extracted_data.get('timestamps', [])
            }
        
        # Default: retornar todo
        return extracted_data
    
    # Métodos auxiliares para cálculos
    def _calculate_ear(self, eye_points):
        """Calcula Eye Aspect Ratio"""
        # Distancias verticales
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Distancia horizontal
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        return ear
    
    def _calculate_mar(self, mouth_points):
        """Calcula Mouth Aspect Ratio"""
        # Distancias verticales (múltiples para mayor precisión)
        A = np.linalg.norm(mouth_points[2] - mouth_points[10])  # Arriba-abajo izquierda
        B = np.linalg.norm(mouth_points[4] - mouth_points[8])   # Arriba-abajo derecha
        C = np.linalg.norm(mouth_points[3] - mouth_points[9])   # Arriba-abajo centro
        
        # Distancia horizontal
        D = np.linalg.norm(mouth_points[0] - mouth_points[6])   # Izquierda-derecha
        
        # MAR
        mar = (A + B + C) / (3.0 * D) if D > 0 else 0
        return mar
    
    def _calculate_head_tilt(self, nose_tip, chin):
        """Calcula inclinación de cabeza"""
        # Vector vertical ideal
        vertical = np.array([0, 1])
        
        # Vector nariz-mentón
        head_vector = chin - nose_tip
        head_vector_norm = head_vector / np.linalg.norm(head_vector)
        
        # Ángulo con vertical
        angle = np.arccos(np.clip(np.dot(head_vector_norm, vertical), -1.0, 1.0))
        return np.degrees(angle)
    
    def _calculate_head_rotation(self, left_eye, right_eye, nose):
        """Calcula rotación de cabeza (yaw)"""
        # Punto medio entre ojos
        eye_center = (left_eye + right_eye) / 2
        
        # Distancia de cada ojo a la nariz
        left_dist = np.linalg.norm(left_eye - nose)
        right_dist = np.linalg.norm(right_eye - nose)
        
        # Ratio de distancias indica rotación
        rotation_ratio = left_dist / right_dist if right_dist > 0 else 1.0
        
        return rotation_ratio
    
    def _calculate_consistency_score(self, all_metrics):
        """
        Calcula qué tan consistentes son las métricas entre fotos.
        
        Returns:
            float: Score de 0 a 1 (1 = muy consistente)
        """
        if len(all_metrics) < 2:
            return 0.0
        
        # Calcular variabilidad de métricas clave
        key_metrics = ['avg_ear', 'mar', 'face_width']
        variances = []
        
        for metric in key_metrics:
            values = [m.get(metric, 0) for m in all_metrics]
            if values:
                mean_val = np.mean(values)
                if mean_val > 0:
                    # Coeficiente de variación normalizado
                    cv = np.std(values) / mean_val
                    variances.append(cv)
        
        # Invertir para que menor variabilidad = mayor score
        avg_variance = np.mean(variances) if variances else 1.0
        consistency_score = max(0, 1 - avg_variance)
        
        return consistency_score


# Función auxiliar para ejecutar calibración desde línea de comandos
def calibrate_all_operators():
    """Función helper para calibrar todos los operadores"""
    manager = MasterCalibrationManager()
    results = manager.calibrate_all_operators()
    
    print("\n=== RESUMEN DE CALIBRACIÓN ===")
    print(f"Total procesados: {results['total_processed']}")
    print(f"Exitosos: {len(results['successful'])}")
    print(f"Fallidos: {len(results['failed'])}")
    
    if results['failed']:
        print(f"\nOperadores fallidos: {results['failed']}")
    
    return results


if __name__ == "__main__":
    # Ejecutar calibración de prueba
    calibrate_all_operators()