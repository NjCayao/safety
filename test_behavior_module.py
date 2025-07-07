"""
test_behavior_module.py
======================
Script de prueba para verificar el funcionamiento del módulo de comportamientos.
"""

import cv2
import time
import os
import sys
import numpy as np

# Agregar rutas al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.behavior.integrated_behavior_system import IntegratedBehaviorSystem
from core.behavior.behavior_dashboard import BehaviorDashboard
from operators.master_calibration_manager import MasterCalibrationManager

class BehaviorModuleTest:
    def __init__(self):
        print("=== TEST DEL MÓDULO DE COMPORTAMIENTOS ===\n")
        
        # Configuración
        self.model_dir = "assets/models"
        self.audio_dir = "assets/audio"
        self.operators_dir = "operators"
        
        # Detectar si estamos en modo headless
        self.headless = '--headless' in sys.argv
        
        # Componentes
        self.behavior_system = None
        self.dashboard = None
        
        # Operador de prueba
        self.test_operator = {
            'id': '47469940',  # Cambiar según tu operador de prueba
            'name': 'Operador Prueba'
        }
        
    def initialize(self):
        """Inicializa todos los componentes"""
        print("1. Inicializando componentes...")
        
        try:
            # Inicializar sistema de comportamientos
            self.behavior_system = IntegratedBehaviorSystem(
                model_dir=self.model_dir,
                audio_dir=self.audio_dir,
                operators_dir=self.operators_dir
            )
            print("   ✓ Sistema de comportamientos inicializado")
            
            # Inicializar dashboard si no es headless
            if not self.headless:
                self.dashboard = BehaviorDashboard(width=350, position='left')
                print("   ✓ Dashboard inicializado")
            
            return True
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return False
    
    def test_calibration(self):
        """Prueba la calibración del operador"""
        print("\n2. Probando calibración...")
        
        # Verificar si existe calibración
        baseline_path = os.path.join(
            self.operators_dir, 
            "baseline-json",
            self.test_operator['id'],
            "behavior_baseline.json"
        )
        
        if os.path.exists(baseline_path):
            print(f"   ✓ Calibración encontrada para {self.test_operator['name']}")
            
            # Cargar operador
            success = self.behavior_system.set_operator(self.test_operator)
            if success:
                print("   ✓ Operador cargado correctamente")
                
                # Mostrar configuración
                status = self.behavior_system.get_current_status()
                thresholds = status.get('thresholds', {})
                print(f"   - Umbral confianza: {thresholds.get('confidence_threshold', 'N/A')}")
                print(f"   - Umbral nocturno: {thresholds.get('night_confidence_threshold', 'N/A')}")
                print(f"   - Umbral teléfono 1: {thresholds.get('phone_alert_threshold_1', 'N/A')}s")
                print(f"   - Umbral teléfono 2: {thresholds.get('phone_alert_threshold_2', 'N/A')}s")
            else:
                print("   ✗ Error cargando operador")
                
        else:
            print(f"   ⚠ No hay calibración para {self.test_operator['name']}")
            print("   Usando valores por defecto")
        
        return True
    
    def test_detection(self):
        """Prueba la detección en tiempo real"""
        print("\n3. Iniciando prueba de detección...")
        print("   Controles:")
        print("   - 'q': Salir")
        print("   - 'n': Simular modo nocturno")
        print("   - 'd': Simular modo día")
        print("   - 's': Ver estadísticas")
        print("   - 'r': Generar reporte de sesión")
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ✗ Error: No se puede acceder a la cámara")
            return False
        
        # Configurar cámara
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n   ✓ Cámara inicializada - Iniciando detección...\n")
        
        # Variables para estadísticas
        frame_count = 0
        detection_times = []
        total_detections = 0
        total_alerts = 0
        
        # Modo simulado
        simulated_night_mode = False
        
        while True:
            start_time = time.time()
            
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("Error capturando frame")
                break
            
            # Simular modo nocturno si está activado
            if simulated_night_mode:
                # Oscurecer imagen
                frame = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
            
            # Analizar frame
            result = self.behavior_system.analyze_frame(frame)
            
            # Contar detecciones y alertas
            if result.get('detections'):
                total_detections += len(result['detections'])
            if result.get('alerts'):
                total_alerts += len(result['alerts'])
            
            # Obtener frame procesado
            processed_frame = result.get('frame', frame)
            
            # Log en modo headless
            if self.headless and frame_count % 30 == 0:
                self._print_headless_status(result)
            
            # Mostrar FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Renderizar dashboard si no es headless
            if not self.headless and self.dashboard:
                # Preparar datos para dashboard
                dashboard_data = {
                    'detections': result.get('detections', []),
                    'alerts': result.get('alerts', []),
                    'operator_name': result.get('operator_name', 'Unknown'),
                    'is_calibrated': result.get('is_calibrated', False),
                    'is_night_mode': result.get('is_night_mode', False),
                    'detector_info': {
                        'behavior_durations': self.behavior_system.detector.behavior_durations,
                        'cigarette_detections': len(self.behavior_system.detector.cigarette_detections),
                        'cigarette_pattern_threshold': self.behavior_system.detector.config['cigarette_pattern_threshold']
                    }
                }
                
                processed_frame = self.dashboard.render(processed_frame, dashboard_data)
            
            # Mostrar frame si no es headless
            if not self.headless:
                cv2.imshow("Test Módulo Comportamientos", processed_frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n   Finalizando prueba...")
                break
            elif key == ord('n'):
                simulated_night_mode = True
                print("   → Modo NOCTURNO simulado activado")
            elif key == ord('d'):
                simulated_night_mode = False
                print("   → Modo DIURNO activado")
            elif key == ord('s'):
                self._print_statistics()
            elif key == ord('r'):
                self._generate_session_report()
            
            # Actualizar estadísticas
            frame_count += 1
            detection_times.append(time.time() - start_time)
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar resumen
        self._print_summary(frame_count, detection_times, total_detections, total_alerts)
        
        return True
    
    def _print_headless_status(self, result):
        """Imprime estado en modo headless"""
        detections = result.get('detections', [])
        phone_detected = any(d[0] == 'cell phone' for d in detections)
        cigarette_detected = any(d[0] == 'cigarette' for d in detections)
        
        status_parts = []
        if phone_detected:
            status_parts.append("TELÉFONO")
        if cigarette_detected:
            status_parts.append("CIGARRILLO")
        if not status_parts:
            status_parts.append("Sin detecciones")
        
        mode = "NOCHE" if result.get('is_night_mode') else "DÍA"
        
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"Estado: {', '.join(status_parts)} | "
              f"Modo: {mode} | "
              f"Alertas: {len(result.get('alerts', []))}")
    
    def _print_statistics(self):
        """Imprime estadísticas actuales"""
        print("\n=== ESTADÍSTICAS ACTUALES ===")
        status = self.behavior_system.get_current_status()
        
        # Validar que status existe y tiene operador
        if status and status.get('operator'):
            print(f"Operador: {status['operator']['name']}")
            print(f"Calibrado: {'Sí' if status.get('is_calibrated', False) else 'No'}")
        else:
            print("Operador: No configurado")
            return  # Salir temprano si no hay operador
        
        # Resto de las estadísticas solo si hay datos
        session = status.get('session_stats', {})
        if session:
            print(f"\nEstadísticas de sesión:")
            print(f"  - Detecciones totales: {session.get('total_detections', 0)}")
            print(f"  - Alertas teléfono: {session.get('total_phone_alerts', 0)}")
            print(f"  - Alertas cigarrillo: {session.get('total_smoking_alerts', 0)}")
        
        detector = status.get('detector_config', {})
        if detector:
            print(f"\nConfiguración actual:")
            print(f"  - Procesamiento optimizado: {'Sí' if detector.get('enable_optimization') else 'No'}")
            print(f"  - Intervalo procesamiento: cada {detector.get('processing_interval', 1)} frames")
            print(f"  - ROI habilitado: {'Sí' if detector.get('roi_enabled') else 'No'}")
        
        report_stats = status.get('report_stats', {})
        if report_stats:
            print(f"\nReportes:")
            print(f"  - Generados: {report_stats.get('reports_generated', 0)}")
            print(f"  - Enviados: {report_stats.get('reports_sent', 0)}")
        
        print("=============================\n")
    
    def _generate_session_report(self):
        """Genera reporte de sesión"""
        print("\nGenerando reporte de sesión...")
        report = self.behavior_system.generate_session_report()
        
        if report:
            print(f"✓ Reporte generado: {report['id']}")
            print(f"  - Guardado en: {report.get('json_path', 'N/A')}")
        else:
            print("✗ Error generando reporte")
    
    def _print_summary(self, frame_count, detection_times, total_detections, total_alerts):
        """Imprime resumen final"""
        print("\n=== RESUMEN DE LA PRUEBA ===")
        print(f"Frames procesados: {frame_count}")
        
        if detection_times:
            avg_time = np.mean(detection_times) * 1000
            print(f"Tiempo promedio de procesamiento: {avg_time:.2f} ms")
            print(f"FPS promedio: {1000/avg_time:.1f}")
        
        print(f"Total detecciones: {total_detections}")
        print(f"Total alertas: {total_alerts}")
        
        # # Generar reporte final
        # report = self.behavior_system.generate_session_report()
        # if report:
        #     print(f"Reporte de sesión guardado: {report.get('id', 'N/A')}")
        
        print("=============================")
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        # Test 1: Inicialización
        if not self.initialize():
            print("\n✗ Fallo en la inicialización")
            return False
        
        # Test 2: Calibración
        if not self.test_calibration():
            print("\n✗ Fallo en la calibración")
            return False
        
        # Test 3: Detección
        if not self.test_detection():
            print("\n✗ Fallo en la detección")
            return False
        
        print("\n✓ Todas las pruebas completadas exitosamente")
        return True


def test_calibration_only():
    """Prueba solo la calibración de comportamientos"""
    print("=== PRUEBA DE CALIBRACIÓN DE COMPORTAMIENTOS ===\n")
    
    from core.behavior.behavior_calibration import BehaviorCalibration
    
    # Crear calibrador
    calibrator = BehaviorCalibration()
    
    # Operador de prueba
    operator_id = "47469940"  # Cambiar según tu operador
    
    # Datos de prueba simulados
    test_data = {
        'face_areas': [15000, 16000, 14500, 15500, 16500],
        'light_levels': [120, 125, 118, 130, 115],
        'nose_to_mouth': [45, 47, 44, 46, 45]
    }
    
    print(f"Calibrando comportamientos para operador {operator_id}...")
    
    # Ejecutar calibración
    success = calibrator.calibrate_from_extracted_data(
        operator_id,
        test_data,
        photos_count=5
    )
    
    if success:
        print("✓ Calibración exitosa")
        
        # Cargar y mostrar resultados
        profile = calibrator.get_operator_profile(operator_id)
        print(f"\nPerfil del operador:")
        print(f"  - Confianza calibración: {profile['calibration_confidence']*100:.1f}%")
        print(f"  - Umbral confianza: {profile['thresholds']['confidence_threshold']}")
        print(f"  - Factor proximidad: {profile['thresholds']['face_proximity_factor']}")
    else:
        print("✗ Calibración fallida")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test del módulo de comportamientos')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Ejecutar solo prueba de calibración')
    parser.add_argument('--headless', action='store_true',
                       help='Ejecutar en modo headless (sin GUI)')
    
    args = parser.parse_args()
    
    if args.calibrate:
        # Solo probar calibración
        test_calibration_only()
    else:
        # Prueba completa
        tester = BehaviorModuleTest()
        tester.run_all_tests()