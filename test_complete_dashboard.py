"""
Test Completo con Reconocimiento Facial
=======================================
Prueba del dashboard completo con todos los módulos integrados.
"""

import cv2
import dlib
import time
import os
import sys
import numpy as np

# Agregar rutas al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.master_dashboard import MasterDashboard
from core.fatigue.integrated_fatigue_system import IntegratedFatigueSystem
from core.behavior.integrated_behavior_system import IntegratedBehaviorSystem
from core.face_recognition import IntegratedFaceSystem

class CompleteSystemTest:
    def __init__(self):
        print("=== TEST DEL SISTEMA COMPLETO ===")
        print("    Fatiga + Comportamiento + Reconocimiento Facial\n")
        
        # Configuración
        self.model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
        self.yolo_model_dir = "assets/models"
        self.audio_dir = "assets/audio"
        self.operators_dir = "operators"
        
        # Componentes
        self.master_dashboard = None
        self.fatigue_system = None
        self.behavior_system = None
        self.face_system = None
        self.face_detector = None
        self.landmark_predictor = None
        
        # Configuración de ventana
        self.window_width = 1280
        self.window_height = 720
        
        # Estadísticas
        self.frame_count = 0
        self.start_time = time.time()
        
    def initialize(self):
        """Inicializa todos los componentes"""
        print("1. Inicializando componentes...")
        
        try:
            # Dashboard maestro con análisis
            self.master_dashboard = MasterDashboard(
                width=300, 
                position='left',
                enable_analysis_dashboard=True
            )
            print("   ✓ Dashboard maestro inicializado")
            
            # Sistema de reconocimiento facial
            self.face_system = IntegratedFaceSystem(
                operators_dir=self.operators_dir,
                dashboard_position='none'  # No necesitamos su dashboard individual
            )
            self.face_system.enable_dashboard(False)  # Deshabilitamos su dashboard
            print("   ✓ Sistema de reconocimiento facial inicializado")
            
            # Sistema de fatiga
            self.fatigue_system = IntegratedFatigueSystem(
                operators_dir=self.operators_dir,
                model_path=self.model_path,
                headless=False
            )
            print("   ✓ Sistema de fatiga inicializado")
            
            # Sistema de comportamientos
            self.behavior_system = IntegratedBehaviorSystem(
                model_dir=self.yolo_model_dir,
                audio_dir=self.audio_dir,
                operators_dir=self.operators_dir
            )
            print("   ✓ Sistema de comportamientos inicializado")
            
            # Detectores faciales
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(self.model_path)
            print("   ✓ Detectores faciales inicializados")
            
            print(f"\n   Total operadores cargados: {len(self.face_system.recognizer.operators)}")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Ejecuta la prueba del sistema completo"""
        print("\n2. Iniciando prueba...")
        print("   Controles:")
        print("   - 'q': Salir")
        print("   - 's': Ver estadísticas")
        print("   - 'r': Forzar reporte de operador desconocido")
        print("   - 'n': Alternar modo nocturno")
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ✗ Error: No se puede acceder a la cámara")
            return False
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n   ✓ Sistema activo\n")
        
        # Crear ventana
        window_name = "Sistema Completo - Reconocimiento + Fatiga + Comportamiento"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.window_width, self.window_height)
        
        # Variables de control
        simulated_night = False
        last_operator_info = None
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Simular modo nocturno si está activado
            if simulated_night:
                frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
            
            # 1. RECONOCIMIENTO FACIAL PRIMERO
            face_result = self.face_system.identify_and_analyze(frame)
            
            # Obtener operador actual del reconocimiento facial
            current_operator = None
            if face_result and face_result.get('operator_info'):
                operator_info = face_result['operator_info']
                if operator_info.get('is_registered', False):
                    current_operator = {
                        'id': operator_info['id'],
                        'name': operator_info['name']
                    }
                    
                    # Si cambió el operador, actualizar otros sistemas
                    if (not last_operator_info or 
                        last_operator_info.get('id') != operator_info['id']):
                        print(f"\n   → Operador detectado: {current_operator['name']} (ID: {current_operator['id']})")
                        self.fatigue_system.set_operator(current_operator)
                        self.behavior_system.set_operator(current_operator)
                        last_operator_info = operator_info
                else:
                    # Operador no registrado
                    print(f"\r   ⚠️  Operador NO REGISTRADO detectado", end='')
            
            # 2. ANÁLISIS DE FATIGA (solo si hay rostro)
            fatigue_result = None
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray)
            
            if faces:
                face = faces[0]
                landmarks = self.landmark_predictor(gray, face)
                fatigue_result = self.fatigue_system.analyze_frame(frame, landmarks)
            
            # 3. ANÁLISIS DE COMPORTAMIENTO
            face_locations = [(face.top(), face.right(), face.bottom(), face.left())] if faces else None
            behavior_result = self.behavior_system.analyze_frame(frame, face_locations)
            
            # 4. PREPARAR DATOS DE ANÁLISIS (simulados)
            analysis_data = self._generate_analysis_data() if faces else None
            
            # 5. RENDERIZAR DASHBOARD COMPLETO
            # Agregar información del tiempo de operador desconocido
            if face_result and hasattr(self.face_system, 'unknown_operator_start_time'):
                if self.face_system.unknown_operator_start_time:
                    face_result['unknown_operator_time'] = time.time() - self.face_system.unknown_operator_start_time
            
            dashboard_frame = self.master_dashboard.render(
                frame,
                fatigue_result,
                behavior_result,
                face_result,
                analysis_data
            )
            
            # 6. MANEJAR ALERTAS CON FRAME COMPLETO
            if fatigue_result and fatigue_result.get('microsleep_detected'):
                self._handle_fatigue_alert(fatigue_result, dashboard_frame)
            
            if behavior_result and behavior_result.get('alerts'):
                self._handle_behavior_alerts(behavior_result, dashboard_frame)
            
            # Redimensionar para mostrar
            display_frame = cv2.resize(dashboard_frame, (self.window_width, self.window_height))
            
            # Mostrar FPS
            fps = self.frame_count / (time.time() - self.start_time)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", 
                       (10, self.window_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Mostrar
            cv2.imshow(window_name, display_frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nFinalizando...")
                break
            elif key == ord('s'):
                self._print_statistics()
            elif key == ord('r'):
                # Forzar reporte de operador desconocido
                if self.face_system.force_unknown_operator_report(dashboard_frame):
                    print("\n   📨 Reporte de operador desconocido generado")
                else:
                    print("\n   ⚠️  No hay operador desconocido activo")
            elif key == ord('n'):
                simulated_night = not simulated_night
                print(f"\n   Modo {'nocturno' if simulated_night else 'diurno'} activado")
            
            self.frame_count += 1
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        # Resumen final
        self._print_summary()
        
        return True
    
    def _generate_analysis_data(self):
        """Genera datos de prueba para el dashboard de análisis"""
        import random
        
        # Simular variación temporal
        time_factor = np.sin(time.time() * 0.1)
        
        return {
            'analysis': {
                'emotion': {
                    'dominant_emotion': 'neutral',
                    'wellbeing': int(60 + time_factor * 20),
                    'emotions': {
                        'neutral': 45 + random.randint(-5, 5),
                        'happy': 25 + random.randint(-5, 5),
                        'sad': 10 + random.randint(-3, 3),
                        'angry': 5 + random.randint(-2, 2),
                        'surprised': 8 + random.randint(-2, 2),
                        'fear': 4 + random.randint(-2, 2),
                        'disgust': 3 + random.randint(-1, 1)
                    }
                },
                'stress': {
                    'stress_level': int(40 + time_factor * 15)
                },
                'fatigue': {
                    'fatigue_percentage': int(30 + time_factor * 10)
                },
                'pulse': {
                    'is_valid': True,
                    'bpm': int(75 + time_factor * 10)
                },
                'anomaly': {
                    'indicators': {
                        'intoxication': {
                            'level': random.randint(5, 15),
                            'status': 'NORMAL'
                        },
                        'neurological': {
                            'level': random.randint(5, 15),
                            'status': 'NORMAL'
                        },
                        'erratic': {
                            'level': random.randint(10, 20),
                            'status': 'BAJO'
                        }
                    }
                }
            },
            'overall_assessment': {
                'status': 'CONDICION NORMAL',
                'score': 75,
                'color': (0, 255, 0)
            },
            'alerts': [],
            'recommendations': [
                'Mantener postura adecuada',
                'Realizar pausas activas cada hora'
            ]
        }
    
    def _handle_fatigue_alert(self, result, dashboard_frame):
        """Maneja alertas de fatiga con el frame completo"""
        if hasattr(self.fatigue_system, '_handle_microsleep_event'):
            self.fatigue_system._handle_microsleep_event(result, dashboard_frame)
    
    def _handle_behavior_alerts(self, result, dashboard_frame):
        """Maneja alertas de comportamiento con el frame completo"""
        for alert in result.get('alerts', []):
            if hasattr(self.behavior_system, '_handle_behavior_alert'):
                self.behavior_system._handle_behavior_alert(alert, dashboard_frame)
    
    def _print_statistics(self):
        """Imprime estadísticas del sistema"""
        print("\n" + "="*60)
        print("ESTADISTICAS DEL SISTEMA COMPLETO")
        print("="*60)
        
        # Tiempo de ejecución
        runtime = time.time() - self.start_time
        print(f"\nTiempo activo: {runtime:.1f} segundos")
        print(f"Frames procesados: {self.frame_count}")
        print(f"FPS promedio: {self.frame_count/runtime:.1f}")
        
        # Estadísticas de reconocimiento facial
        face_status = self.face_system.get_current_status()
        print(f"\nRECONOCIMIENTO FACIAL:")
        print(f"  - Total análisis: {face_status['session_stats']['total_recognitions']}")
        print(f"  - Exitosos: {face_status['session_stats']['successful_recognitions']}")
        print(f"  - Desconocidos: {face_status['session_stats']['unknown_detections']}")
        print(f"  - Operadores detectados: {len(face_status['session_stats']['operators_detected'])}")
        print(f"  - Reportes de desconocido: {face_status['session_stats']['unknown_operator_reports']}")
        
        # Estadísticas de fatiga
        fatigue_status = self.fatigue_system.get_current_status()
        print(f"\nFATIGA:")
        print(f"  - Microsueños: {fatigue_status['session_stats']['total_microsleeps']}")
        print(f"  - Alertas: {fatigue_status['session_stats']['total_alerts']}")
        
        # Estadísticas de comportamiento
        behavior_status = self.behavior_system.get_current_status()
        print(f"\nCOMPORTAMIENTO:")
        print(f"  - Alertas teléfono: {behavior_status['session_stats']['total_phone_alerts']}")
        print(f"  - Alertas cigarrillo: {behavior_status['session_stats']['total_smoking_alerts']}")
        
        print("="*60 + "\n")
    
    def _print_summary(self):
        """Imprime resumen final"""
        print("\n" + "="*70)
        print("RESUMEN DE LA SESION - SISTEMA COMPLETO")
        print("="*70)
        
        runtime = time.time() - self.start_time
        print(f"Duración total: {runtime:.1f} segundos")
        print(f"Frames totales: {self.frame_count}")
        print(f"FPS promedio: {self.frame_count/runtime:.1f}")
        
        # Generar reportes de sesión
        print("\nGenerando reportes finales...")
        
        # No generar reportes si no hubo actividad significativa
        if runtime > 10:
            fatigue_report = self.fatigue_system.generate_session_report()
            if fatigue_report:
                print(f"  ✓ Reporte de fatiga: {fatigue_report['id']}")
            
            behavior_report = self.behavior_system.generate_session_report()
            if behavior_report:
                print(f"  ✓ Reporte de comportamiento: {behavior_report['id']}")
        
        print("\n✓ Sesión completada exitosamente")
        print("="*70)


if __name__ == "__main__":
    tester = CompleteSystemTest()
    
    if tester.initialize():
        tester.run()
    else:
        print("\n✗ Error: No se pudo inicializar el sistema")