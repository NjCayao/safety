"""
Script de Prueba del Sistema de Análisis Integrado
=================================================
Prueba visual completa con cámara web.
"""

import cv2
import numpy as np
import sys
import os
import time
import logging
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestAnalysis')

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import dlib
    import face_recognition
    print("✅ Librerías de detección facial cargadas")
except ImportError as e:
    print(f"❌ Error: {e}")
    print("Instala: pip install dlib face-recognition")
    sys.exit(1)

# Importar sistema de análisis
try:
    from core.analysis import IntegratedAnalysisSystem
    print("✅ Sistema de análisis importado")
except ImportError as e:
    print(f"❌ Error importando sistema de análisis: {e}")
    sys.exit(1)

class AnalysisSystemTester:
    def __init__(self):
        """Inicializa el sistema de prueba"""
        print("\n🚀 Inicializando Sistema de Prueba...")
        
        # Configuración
        self.operators_dir = "operators"
        self.show_gui = True
        self.test_mode = 'full'  # 'full', 'calibration', 'headless'
        
        # Crear directorio si no existe
        os.makedirs(self.operators_dir, exist_ok=True)
        
        # Inicializar cámara
        print("📷 Inicializando cámara...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            sys.exit(1)
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("✅ Cámara lista")
        
        # Inicializar detector facial
        print("🔍 Inicializando detector facial...")
        self.init_face_detector()
        
        # Inicializar sistema de análisis
        print("🧠 Inicializando sistema de análisis...")
        self.analysis_system = IntegratedAnalysisSystem(
            operators_dir=self.operators_dir,
            headless=(self.test_mode == 'headless')
        )
        print("✅ Sistema de análisis listo")
        
        # Operador de prueba
        self.test_operators = [
            {"id": "12345678", "name": "Juan Pérez"},
            {"id": "87654321", "name": "María García"},
            {"id": "11223344", "name": "Pedro López"}
        ]
        self.current_operator_idx = 0
        self.current_operator = self.test_operators[0]
        
        # Estado
        self.is_running = True
        self.show_debug = True
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        
        # Resultados del último análisis
        self.last_results = None
        
    def init_face_detector(self):
        """Inicializa el detector facial"""
        try:
            # Intentar con dlib
            model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
            if os.path.exists(model_path):
                self.face_detector = dlib.get_frontal_face_detector()
                self.landmark_predictor = dlib.shape_predictor(model_path)
                self.detection_method = 'dlib'
                print("✅ Detector facial dlib inicializado")
            else:
                print("⚠️ Modelo dlib no encontrado, usando face_recognition")
                self.detection_method = 'face_recognition'
        except Exception as e:
            print(f"⚠️ Error con dlib: {e}, usando face_recognition")
            self.detection_method = 'face_recognition'
    
    def detect_face_and_landmarks(self, frame):
        """Detecta rostro y landmarks"""
        if self.detection_method == 'dlib':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 0)
            
            if faces:
                face = faces[0]
                landmarks = self.landmark_predictor(gray, face)
                
                # Convertir a formato face_recognition
                landmarks_dict = self.dlib_to_face_recognition_format(landmarks)
                face_location = (face.top(), face.right(), face.bottom(), face.left())
                
                return landmarks_dict, face_location
        else:
            # Usar face_recognition
            face_locations = face_recognition.face_locations(frame)
            if face_locations:
                face_landmarks = face_recognition.face_landmarks(frame, face_locations)
                if face_landmarks:
                    return face_landmarks[0], face_locations[0]
        
        return None, None
    
    def dlib_to_face_recognition_format(self, landmarks):
        """Convierte landmarks de dlib al formato de face_recognition"""
        landmarks_dict = {
            'chin': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)],
            'left_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)],
            'right_eyebrow': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)],
            'nose_bridge': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)],
            'nose_tip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(31, 36)],
            'left_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
            'right_eye': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
            'top_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 55)] + 
                      [(landmarks.part(i).x, landmarks.part(i).y) for i in range(64, 59, -1)],
            'bottom_lip': [(landmarks.part(i).x, landmarks.part(i).y) for i in range(54, 60)] + 
                         [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 54)]
        }
        return landmarks_dict
    
    def run(self):
        """Ejecuta el sistema de prueba"""
        print("\n" + "="*60)
        print("SISTEMA DE ANÁLISIS INTEGRADO - MODO PRUEBA")
        print("="*60)
        print("\nControles:")
        print("  Q - Salir")
        print("  C - Cambiar operador")
        print("  R - Forzar recalibración")
        print("  D - Mostrar/Ocultar debug")
        print("  H - Cambiar a modo headless")
        print("  J - Exportar JSON de resultados")
        print("  ESPACIO - Pausar análisis")
        print("\n👤 Operador actual:", self.current_operator['name'])
        print("-"*60 + "\n")
        
        if self.show_gui:
            cv2.namedWindow("Test Sistema Análisis", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test Sistema Análisis", 1280, 720)
        
        paused = False
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Error leyendo frame")
                break
            
            # Voltear para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Actualizar FPS
            self.update_fps()
            
            # Detectar rostro y landmarks
            landmarks, face_location = self.detect_face_and_landmarks(frame)
            
            if landmarks and face_location and not paused:
                # Analizar con el sistema integrado
                frame, results = self.analysis_system.analyze_operator(
                    frame,
                    landmarks,
                    face_location,
                    self.current_operator
                )
                
                self.last_results = results
                
                # Mostrar estado de calibración
                if results.get('status') == 'calibrating':
                    progress = results.get('progress', 0)
                    print(f"\r🔄 Calibrando {self.current_operator['name']}: {progress}%", end='')
                    if progress == 100:
                        print("\n✅ Calibración completada!")
                
            else:
                if not landmarks:
                    cv2.putText(frame, "No se detecta rostro", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar información adicional
            # if self.show_debug:
            #     self.draw_debug_info(frame)
            
            # Mostrar frame
            if self.show_gui:
                cv2.imshow("Test Sistema Análisis", frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.change_operator()
            elif key == ord('r'):
                self.force_recalibration()
            elif key == ord('d'):
                self.show_debug = not self.show_debug
            elif key == ord('h'):
                self.toggle_headless_mode()
            elif key == ord('j'):
                self.export_json_results()
            elif key == ord(' '):
                paused = not paused
                print("⏸️ Análisis pausado" if paused else "▶️ Análisis reanudado")
            
            self.frame_count += 1
        
        self.cleanup()   
    
    def update_fps(self):
        """Actualiza el cálculo de FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time > 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def change_operator(self):
        """Cambia al siguiente operador de prueba"""
        self.current_operator_idx = (self.current_operator_idx + 1) % len(self.test_operators)
        self.current_operator = self.test_operators[self.current_operator_idx]
        print(f"\n🔄 Cambiado a operador: {self.current_operator['name']} ({self.current_operator['id']})")
    
    def force_recalibration(self):
        """Fuerza recalibración del operador actual"""
        if self.analysis_system.force_recalibration():
            print(f"\n🔄 Recalibrando a {self.current_operator['name']}...")
        else:
            print("\n⚠️ No se pudo iniciar recalibración")
    
    def toggle_headless_mode(self):
        """Cambia entre modo visual y headless"""
        self.analysis_system.headless = not self.analysis_system.headless
        mode = "Headless" if self.analysis_system.headless else "Visual"
        print(f"\n🔄 Cambiado a modo: {mode}")
    
    def export_json_results(self):
        """Exporta los resultados actuales a JSON"""
        if self.last_results:
            # Obtener reporte completo
            report = self.analysis_system.get_analysis_report()
            
            # Guardar a archivo
            filename = f"analysis_report_{int(time.time())}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\n📄 Reporte exportado a: {filename}")
            
            # Mostrar resumen en consola
            print("\n--- RESUMEN DEL ANÁLISIS ---")
            if 'current_analysis' in report:
                analysis = report['current_analysis']
                
                if 'fatigue' in analysis:
                    print(f"Fatiga: {analysis['fatigue']['fatigue_level']}%")
                
                if 'stress' in analysis:
                    print(f"Estrés: {analysis['stress']['stress_level']}%")
                
                if 'pulse' in analysis:
                    pulse_data = analysis['pulse']
                    if pulse_data.get('current_bpm'):
                        print(f"Pulso: {pulse_data['current_bpm']} BPM")
                
                if 'emotion' in analysis:
                    print(f"Emoción: {analysis['emotion'].get('current_emotion', 'N/A')}")
            print("-" * 30)
        else:
            print("\n⚠️ No hay resultados para exportar")
    
    def cleanup(self):
        """Limpia recursos"""
        print("\n🧹 Limpiando recursos...")
        self.cap.release()
        if self.show_gui:
            cv2.destroyAllWindows()
        print("✅ Test finalizado")

def main():
    """Función principal"""
    print("🧪 TEST DEL SISTEMA DE ANÁLISIS INTEGRADO")
    print("=========================================\n")
    
    # Verificar archivos necesarios
    required_files = [
        "core/analysis/__init__.py",
        "core/analysis/integrated_analysis_system.py",
        "core/analysis/calibration_manager.py",
        "core/analysis/fatigue_detector.py",
        "core/analysis/stress_analyzer.py",
        "core/analysis/emotion_analyzer.py",
        "core/analysis/pulse_estimator.py",
        "core/analysis/anomaly_detector.py",
        "core/analysis/analysis_dashboard.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Archivos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nAsegúrate de tener todos los archivos del sistema de análisis.")
        return
    
    try:
        tester = AnalysisSystemTester()
        tester.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()