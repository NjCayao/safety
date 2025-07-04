"""
Script de Prueba de Módulos de Análisis con Cámara Real
======================================================
Prueba los módulos usando detección facial real desde la cámara.
"""

import cv2
import numpy as np
import sys
import os
import time
import logging
import dlib
import face_recognition

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar los módulos a probar
try:
    from core.analysis import (
        EmotionAnalyzer,
        StressAnalyzer,
        PulseEstimator,
        AnomalyDetector,
        AnalysisDashboard,
        IntegratedAnalysisSystem,
        FatigueStressMonitor
    )
    print("✅ Todos los módulos de análisis importados correctamente")
except ImportError as e:
    print(f"❌ Error importando módulos de análisis: {e}")
    sys.exit(1)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TestWithCamera')


class CameraAnalysisTester:
    """Prueba los módulos de análisis con cámara en vivo"""
    
    def __init__(self):
        """Inicializa el sistema de prueba"""
        self.logger = logger
        
        # Inicializar cámara
        print("📷 Inicializando cámara...")
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("❌ Error: No se pudo abrir la cámara")
            sys.exit(1)
        
        # Configurar resolución
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✅ Cámara inicializada")
        
        # Inicializar detector facial
        print("🔍 Inicializando detector facial...")
        try:
            # Intentar cargar el predictor de landmarks
            model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(model_path):
                print(f"❌ Error: No se encuentra el modelo en {model_path}")
                print("📥 Descarga el modelo de: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
                sys.exit(1)
            
            self.face_detector = dlib.get_frontal_face_detector()
            self.landmark_predictor = dlib.shape_predictor(model_path)
            print("✅ Detector facial inicializado")
        except Exception as e:
            print(f"❌ Error inicializando detector facial: {e}")
            sys.exit(1)
        
        # Inicializar analizadores
        print("🧠 Inicializando analizadores...")
        self.emotion_analyzer = EmotionAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        self.fatigue_stress_monitor = FatigueStressMonitor()
        self.pulse_estimator = PulseEstimator()
        self.anomaly_detector = AnomalyDetector()
        self.dashboard = AnalysisDashboard(panel_width=400)
        self.integrated_system = IntegratedAnalysisSystem()
        print("✅ Analizadores inicializados")
        
        # Estado
        self.is_running = True
        self.show_landmarks = True
        self.current_mode = 'integrated'  # 'integrated', 'emotion', 'stress', 'fatigue_stress', 'pulse', 'anomaly'
        
        # FPS
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0
        
    def run(self):
        """Ejecuta el sistema de prueba con cámara"""
        print("\n" + "="*60)
        print("SISTEMA DE PRUEBA CON CÁMARA EN VIVO")
        print("="*60)
        print("\nControles:")
        print("  Q - Salir")
        print("  L - Mostrar/Ocultar landmarks")
        print("  1 - Modo Integrado (todos los análisis)")
        print("  2 - Solo Emociones")
        print("  3 - Solo Estrés")
        print("  4 - Monitor Fatiga/Estrés")
        print("  5 - Solo Pulso")
        print("  6 - Solo Anomalías")
        print("\n👤 Colócate frente a la cámara...")
        print("-"*60 + "\n")
        
        cv2.namedWindow("Prueba Análisis Facial", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Prueba Análisis Facial", 1280, 720)
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error leyendo frame de la cámara")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Procesar frame
            processed_frame = self.process_frame(frame)
            
            # Mostrar FPS
            self.update_fps()
            cv2.putText(processed_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Mostrar modo actual
            cv2.putText(processed_frame, f"Modo: {self.current_mode.upper()}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow("Prueba Análisis Facial", processed_frame)
            
            # Procesar teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('1'):
                self.current_mode = 'integrated'
                print("📊 Modo: Análisis Integrado")
            elif key == ord('2'):
                self.current_mode = 'emotion'
                print("😊 Modo: Solo Emociones")
            elif key == ord('3'):
                self.current_mode = 'stress'
                print("😰 Modo: Solo Estrés")
            elif key == ord('4'):
                self.current_mode = 'fatigue_stress'
                print("📊 Modo: Monitor Fatiga/Estrés")
            elif key == ord('5'):
                self.current_mode = 'pulse'
                print("❤️ Modo: Solo Pulso")
            elif key == ord('6'):
                self.current_mode = 'anomaly'
                print("⚠️ Modo: Solo Anomalías")
        
        self.cleanup()
        
    def process_frame(self, frame):
        """Procesa un frame con todos los análisis"""
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 0)
        
        if not faces:
            cv2.putText(frame, "No se detecta rostro", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame
        
        # Procesar el primer rostro detectado
        face = faces[0]
        
        # Obtener landmarks
        landmarks = self.landmark_predictor(gray, face)
        landmarks_dict = self.landmarks_to_dict(landmarks)
        
        # Dibujar rectángulo del rostro
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Dibujar landmarks si está habilitado
        if self.show_landmarks:
            self.draw_landmarks(frame, landmarks_dict)
        
        # Ejecutar análisis según el modo
        if self.current_mode == 'integrated':
            # Análisis completo integrado
            face_location = (y1, x2, y2, x1)  # formato: top, right, bottom, left
            analysis_result = self.integrated_system.analyze_operator(frame, landmarks_dict, face_location)
            
            # Renderizar dashboard completo
            frame = self.dashboard.render(frame, analysis_result)
            
        elif self.current_mode == 'emotion':
            # Solo emociones
            emotion_result = self.emotion_analyzer.analyze(frame, landmarks_dict)
            frame = self.emotion_analyzer.draw_emotion_panel(frame, emotion_result)
            
        elif self.current_mode == 'stress':
            # Solo estrés
            stress_result = self.stress_analyzer.analyze(frame, landmarks_dict)
            frame = self.stress_analyzer.draw_stress_info(frame, stress_result)
            
        elif self.current_mode == 'fatigue_stress':
            # Monitor de Fatiga/Estrés
            # Simular info de fatiga (en producción vendría del detector principal)
            fatigue_info = {
                'microsleep_count': 0  # Cambiar para simular diferentes niveles
            }
            
            # Actualizar monitor
            self.fatigue_stress_monitor.update(frame, landmarks_dict, fatigue_info)
            
            # Dibujar panel
            frame = self.fatigue_stress_monitor.draw_panel(frame)
            
            # Mostrar recomendaciones si hay alertas
            if self.fatigue_stress_monitor.should_alert():
                recommendations = self.fatigue_stress_monitor.get_recommendations()
                y = 50
                for rec in recommendations:
                    cv2.putText(frame, rec, (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    y += 30
            
        elif self.current_mode == 'pulse':
            # Solo pulso
            pulse_result = self.pulse_estimator.process_frame(frame, landmarks_dict)
            frame = self.pulse_estimator.draw_pulse_info(frame, pulse_result)
            
        elif self.current_mode == 'anomaly':
            # Solo anomalías
            # Necesitamos datos de emoción para mejor detección
            emotion_data = self.emotion_analyzer.analyze(frame, landmarks_dict)
            anomaly_result = self.anomaly_detector.analyze(frame, landmarks_dict, emotion_data)
            frame = self.anomaly_detector.draw_anomaly_info(frame, anomaly_result)
        
        return frame
    
    def landmarks_to_dict(self, landmarks):
        """Convierte landmarks de dlib a diccionario compatible con face_recognition"""
        landmarks_dict = {}
        
        # Mapeo de índices de dlib a características faciales
        landmarks_dict['chin'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)]
        landmarks_dict['left_eyebrow'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)]
        landmarks_dict['right_eyebrow'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)]
        landmarks_dict['nose_bridge'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(27, 31)]
        landmarks_dict['nose_tip'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(31, 36)]
        landmarks_dict['left_eye'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        landmarks_dict['right_eye'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
        landmarks_dict['top_lip'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 55)]
        landmarks_dict['top_lip'].extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(64, 59, -1)])
        landmarks_dict['bottom_lip'] = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(54, 60)]
        landmarks_dict['bottom_lip'].extend([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 60)])
        
        return landmarks_dict
    
    def draw_landmarks(self, frame, landmarks_dict):
        """Dibuja los landmarks faciales"""
        # Colores para diferentes partes
        colors = {
            'chin': (255, 255, 255),
            'left_eyebrow': (0, 255, 255),
            'right_eyebrow': (0, 255, 255),
            'nose_bridge': (255, 255, 0),
            'nose_tip': (255, 255, 0),
            'left_eye': (0, 255, 0),
            'right_eye': (0, 255, 0),
            'top_lip': (0, 0, 255),
            'bottom_lip': (0, 0, 255)
        }
        
        # Dibujar puntos
        for feature, points in landmarks_dict.items():
            color = colors.get(feature, (255, 255, 255))
            for point in points:
                cv2.circle(frame, point, 2, color, -1)
    
    def update_fps(self):
        """Actualiza el cálculo de FPS"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed > 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
    
    def cleanup(self):
        """Limpia recursos"""
        print("\n🧹 Limpiando recursos...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Recursos liberados")


def main():
    """Función principal"""
    print("🚀 Sistema de Prueba de Análisis Facial con Cámara Real")
    print("Versión 1.0")
    print()
    
    # Verificar dependencias
    try:
        import dlib
        import face_recognition
        print("✅ Dependencias verificadas")
    except ImportError as e:
        print(f"❌ Error: Falta instalar dependencias: {e}")
        print("\nInstala con:")
        print("  pip install dlib face_recognition")
        return
    
    # Crear y ejecutar tester
    try:
        tester = CameraAnalysisTester()
        tester.run()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()