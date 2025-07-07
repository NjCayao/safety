"""
test_fatigue_module.py
=====================
Script de prueba para verificar el funcionamiento del módulo de fatiga.
"""

import cv2
import dlib
import time
import os
import sys
import numpy as np

# Agregar rutas al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fatigue.integrated_fatigue_system import IntegratedFatigueSystem
from core.fatigue.fatigue_dashboard import FatigueDashboard
from operators.master_calibration_manager import MasterCalibrationManager

class FatigueModuleTest:
    def __init__(self):
        print("=== TEST DEL MÓDULO DE FATIGA ===\n")
        
        # Configuración
        self.model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
        self.operators_dir = "operators"
        
        # Detectar si estamos en modo headless
        self.headless = '--headless' in sys.argv
        
        # Componentes
        self.fatigue_system = None
        self.dashboard = None
        self.face_detector = None
        self.landmark_predictor = None
        
        # Operador de prueba
        self.test_operator = {
            'id': '47469940',  # Cambiar según tu operador de prueba
            'name': 'Nilson Cayao'
        }
        
    def initialize(self):
        """Inicializa todos los componentes"""
        print("1. Inicializando componentes...")
        
        try:
            # Inicializar sistema de fatiga
            self.fatigue_system = IntegratedFatigueSystem(
                operators_dir=self.operators_dir,
                model_path=self.model_path,
                headless=self.headless
            )
            print("   ✓ Sistema de fatiga inicializado")
            
            # Inicializar dashboard si no es headless
            if not self.headless:
                self.dashboard = FatigueDashboard(width=350, position='right')
                print("   ✓ Dashboard inicializado")
            
            # Inicializar detectores
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(self.model_path)
            print("   ✓ Detectores faciales inicializados")
            
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
            "fatigue_baseline.json"
        )
        
        if os.path.exists(baseline_path):
            print(f"   ✓ Calibración encontrada para {self.test_operator['name']}")
            
            # Cargar operador
            success = self.fatigue_system.set_operator(self.test_operator)
            if success:
                print("   ✓ Operador cargado correctamente")
                
                # Mostrar umbrales
                status = self.fatigue_system.get_current_status()
                thresholds = status.get('thresholds', {})
                print(f"   - Umbral EAR: {thresholds.get('ear_threshold', 'N/A')}")
                print(f"   - Ajuste nocturno: {thresholds.get('ear_night_adjustment', 'N/A')}")
                print(f"   - Confianza: {thresholds.get('calibration_confidence', 0)*100:.1f}%")
            else:
                print("   ✗ Error cargando operador")
                
        else:
            print(f"   ⚠ No hay calibración para {self.test_operator['name']}")
            print("   Ejecuta primero la calibración maestra")
            return False
        
        return True
    
    def test_detection(self):
        """Prueba la detección en tiempo real"""
        print("\n3. Iniciando prueba de detección...")
        print("   - Presiona 'q' para salir")
        print("   - Presiona 'n' para simular modo nocturno")
        print("   - Presiona 'd' para simular modo día")
        print("   - Presiona 's' para ver estadísticas")
        
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
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detectar rostros
            faces = self.face_detector(gray)
            
            if faces:
                # Usar el primer rostro
                face = faces[0]
                
                # Obtener landmarks
                landmarks = self.landmark_predictor(gray, face)
                
                # Analizar fatiga
                result = self.fatigue_system.analyze_frame(frame, landmarks)
                
                # Mostrar resultados
                self._display_results(frame, result, face)
                
                # Log en modo headless
                if self.headless and frame_count % 30 == 0:
                    self._print_headless_status(result)
            else:
                # Sin rostro detectado
                cv2.putText(frame, "No se detecta rostro", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Renderizar dashboard si no es headless
            if not self.headless and self.dashboard and faces:
                frame = self.dashboard.render(frame, result)
            
            # Mostrar frame si no es headless
            if not self.headless:
                cv2.imshow("Test Módulo Fatiga", frame)
            
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
            
            # Actualizar estadísticas
            frame_count += 1
            detection_times.append(time.time() - start_time)
        
        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        
        # Mostrar resumen
        self._print_summary(frame_count, detection_times)
        
        return True
    
    def _display_results(self, frame, result, face):
        """Muestra resultados en el frame"""
        if self.headless:
            return
        
        # Dibujar rectángulo del rostro
        cv2.rectangle(frame, 
                     (face.left(), face.top()), 
                     (face.right(), face.bottom()),
                     (0, 255, 0), 2)
        
        # Información básica
        y_offset = 30
        
        # Estado de fatiga
        fatigue_pct = result.get('fatigue_percentage', 0)
        color = (0, 0, 255) if fatigue_pct > 60 else (0, 165, 255) if fatigue_pct > 40 else (0, 255, 0)
        cv2.putText(frame, f"Fatiga: {fatigue_pct}%", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # EAR
        y_offset += 25
        ear = result.get('ear_value', 0)
        threshold = result.get('ear_threshold', 0.25)
        cv2.putText(frame, f"EAR: {ear:.3f} (Umbral: {threshold:.3f})", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Microsueños
        y_offset += 25
        microsleeps = result.get('microsleep_count', 0)
        color = (0, 0, 255) if microsleeps >= 3 else (0, 165, 255) if microsleeps >= 1 else (0, 255, 0)
        cv2.putText(frame, f"Microsuenos: {microsleeps}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Modo día/noche
        y_offset += 25
        mode = "NOCHE" if result.get('is_night_mode', False) else "DIA"
        cv2.putText(frame, f"Modo: {mode}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def _print_headless_status(self, result):
        """Imprime estado en modo headless"""
        print(f"[{time.strftime('%H:%M:%S')}] "
              f"Fatiga: {result.get('fatigue_percentage', 0)}% | "
              f"EAR: {result.get('ear_value', 0):.3f} | "
              f"Microsueños: {result.get('microsleep_count', 0)} | "
              f"Modo: {'NOCHE' if result.get('is_night_mode') else 'DÍA'}")
    
    def _print_statistics(self):
        """Imprime estadísticas actuales"""
        print("\n=== ESTADÍSTICAS ACTUALES ===")
        status = self.fatigue_system.get_current_status()
        
        print(f"Operador: {status['operator']['name']}")
        print(f"Calibrado: {'Sí' if status['is_calibrated'] else 'No'}")
        
        stats = status.get('detector_stats', {})
        print(f"Microsueños totales: {stats.get('total_microsleeps', 0)}")
        print(f"Parpadeos totales: {stats.get('total_blinks', 0)}")
        print(f"Tasa de parpadeo: {stats.get('average_blink_rate', 0):.1f}/min")
        
        session = status.get('session_stats', {})
        print(f"Frames analizados: {session.get('total_detections', 0)}")
        print(f"Alertas generadas: {session.get('total_alerts', 0)}")
        print("==============================\n")
    
    def _print_summary(self, frame_count, detection_times):
        """Imprime resumen final"""
        print("\n=== RESUMEN DE LA PRUEBA ===")
        print(f"Frames procesados: {frame_count}")
        
        if detection_times:
            avg_time = np.mean(detection_times) * 1000
            print(f"Tiempo promedio de procesamiento: {avg_time:.2f} ms")
            print(f"FPS promedio: {1000/avg_time:.1f}")
        
        # Obtener reporte final
        report = self.fatigue_system.get_report()
        session = report.get('session', {})
        
        print(f"Microsueños detectados: {session.get('total_microsleeps', 0)}")
        print(f"Alertas generadas: {session.get('total_alerts', 0)}")
        print("============================")
    
    def run_all_tests(self):
        """Ejecuta todas las pruebas"""
        # Test 1: Inicialización
        if not self.initialize():
            print("\n✗ Fallo en la inicialización")
            return False
        
        # Test 2: Calibración
        if not self.test_calibration():
            print("\n⚠ Advertencia: Sin calibración, usando valores por defecto")
        
        # Test 3: Detección
        if not self.test_detection():
            print("\n✗ Fallo en la detección")
            return False
        
        print("\n✓ Todas las pruebas completadas exitosamente")
        return True


def test_calibration_only():
    """Prueba solo la calibración"""
    print("=== PRUEBA DE CALIBRACIÓN ===\n")
    
    # Ejecutar calibración maestra
    manager = MasterCalibrationManager()
    
    # Calibrar operador específico
    operator_id = "47469940"  # Cambiar según tu operador
    photos_path = f"server/operator-photo/{operator_id}"
    
    if os.path.exists(photos_path):
        print(f"Calibrando operador {operator_id}...")
        success = manager.calibrate_operator(operator_id, photos_path, "Operador Prueba")
        
        if success:
            print("✓ Calibración exitosa")
            
            # Verificar archivos generados
            baseline_dir = f"operators/baseline-json/{operator_id}"
            files = ['master_baseline.json', 'fatigue_baseline.json']
            
            for file in files:
                path = os.path.join(baseline_dir, file)
                if os.path.exists(path):
                    print(f"   ✓ {file} generado")
                else:
                    print(f"   ✗ {file} NO encontrado")
        else:
            print("✗ Calibración fallida")
    else:
        print(f"✗ No se encontró la carpeta de fotos: {photos_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test del módulo de fatiga')
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
        tester = FatigueModuleTest()
        tester.run_all_tests()