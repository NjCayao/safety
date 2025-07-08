"""
Test del Sistema de Detección de Bostezos
"""

import cv2
import sys
import os
import dlib

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.yawn import IntegratedYawnSystem

def main():
    print("\n=== TEST SISTEMA DE DETECCIÓN DE BOSTEZOS ===")
    print("Inicializando...")
    
    # Verificar modelo de landmarks
    model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        return
    
    # Inicializar detectores
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(model_path)
    
    # Crear instancia del sistema
    yawn_system = IntegratedYawnSystem(
        operators_dir="operators",
        dashboard_position='right'  # Puedes cambiar a 'left'
    )
    
    print(f"✅ Sistema inicializado")
    
    # Configurar operador - CAMBIAR ESTE ID POR UNO REAL
    # Puedes usar el ID de un operador que tengas calibrado
    test_operator = {
        'id': '47469940',  # Cambia esto por el DNI de un operador real
        'name': 'Operador Real'  # Se actualizará con el nombre real si existe
    }
    
    # Intentar cargar información real del operador
    operator_path = os.path.join("operators", test_operator['id'])
    if os.path.exists(operator_path):
        info_file = os.path.join(operator_path, "info.txt")
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    test_operator['name'] = lines[1].strip()
    
    yawn_system.set_operator(test_operator)
    
    status = yawn_system.get_current_status()
    print(f"✅ Operador: {test_operator['name']}")
    print(f"✅ Calibración: {'PERSONALIZADA' if status['is_calibrated'] else 'POR DEFECTO'}")
    
    # Abrir cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    print("\n📷 Cámara activa")
    print("\nControles:")
    print("  • q: Salir")
    print("  • d: Cambiar posición del dashboard")
    print("  • h: Ocultar/mostrar dashboard")
    print("  • r: Reiniciar contador de bostezos")
    print("  • f: Forzar reporte de múltiples bostezos (testing)")
    print("\n")
    
    dashboard_visible = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        
        landmarks = None
        if faces:
            # Obtener landmarks del primer rostro
            face = faces[0]
            landmarks = landmark_predictor(gray, face)
        
        # Procesar frame con el sistema de bostezos
        result = yawn_system.analyze_frame(frame, landmarks)
        
        # Mostrar información en consola cada segundo (30 frames aprox)
        if frame_count % 30 == 0:
            detection = result.get('detection_result', {})
            
            if detection.get('mar_value', 0) > 0:
                if detection.get('is_yawning', False):
                    duration = detection.get('yawn_duration', 0)
                    print(f"\r🟡 BOSTEZANDO | "
                          f"Duración: {duration:.1f}s | "
                          f"MAR: {detection['mar_value']:.2f} | "
                          f"Bostezos: {result['yawn_count']}/{result['max_yawns']}    ", end='')
                else:
                    print(f"\r✅ Normal | "
                          f"MAR: {detection['mar_value']:.2f} | "
                          f"Umbral: {detection['mar_threshold']:.2f} | "
                          f"Bostezos: {result['yawn_count']}/{result['max_yawns']}    ", end='')
            else:
                print(f"\r⏸️  Sin detección de rostro                              ", end='')
        
        # Mostrar alerta especial si hay múltiples bostezos
        if result.get('multiple_yawns', False):
            if frame_count % 15 == 0:  # Parpadeo cada medio segundo
                print(f"\n⚠️  ALERTA FATIGA: {result['yawn_count']} bostezos detectados en {result['window_minutes']} minutos!")
        
        frame_count += 1
        
        # Mostrar frame con dashboard
        cv2.imshow('Test Detección de Bostezos', result['frame'])
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Cambiar posición del dashboard
            new_pos = 'left' if yawn_system.dashboard.position == 'right' else 'right'
            yawn_system.set_dashboard_position(new_pos)
            print(f"\n📊 Dashboard movido a: {new_pos}")
        elif key == ord('h'):
            # Ocultar/mostrar dashboard
            dashboard_visible = not dashboard_visible
            yawn_system.enable_dashboard(dashboard_visible)
            print(f"\n📊 Dashboard: {'visible' if dashboard_visible else 'oculto'}")
        elif key == ord('r'):
            # Reiniciar contador
            yawn_system.reset_yawn_counter()
            print("\n🔄 Contador de bostezos reiniciado")
        elif key == ord('f'):
            # Forzar reporte
            if yawn_system.force_yawn_report(result['frame']):
                print("\n📨 REPORTE FORZADO: Se generó reporte de múltiples bostezos")
                print("   Revisa la carpeta 'reports/yawn/'")
            else:
                print("\n⚠️  No hay bostezos registrados para reportar")
    
    # Al salir, mostrar resumen
    print("\n\n=== RESUMEN DE LA SESIÓN ===")
    
    status = yawn_system.get_current_status()
    stats = status['session_stats']
    
    print(f"Total de análisis: {stats['total_detections']}")
    print(f"Bostezos detectados: {stats['total_yawns']}")
    print(f"Eventos de múltiples bostezos: {stats['multiple_yawn_events']}")
    print(f"Reportes generados: {stats['reports_generated']}")
    
    # Información adicional
    if stats['total_yawns'] > 0:
        avg_time = stats['total_detections'] / stats['total_yawns']
        print(f"Promedio: 1 bostezo cada {avg_time:.0f} frames ({avg_time/30:.1f} segundos)")
    
    # Estado del detector
    detector_status = status['detector_status']
    print(f"\nEstado final:")
    print(f"  • Modo: {'NOCTURNO' if detector_status['is_night_mode'] else 'DIURNO'}")
    print(f"  • Nivel de luz: {detector_status['light_level']:.1f}")
    print(f"  • Umbral MAR: {detector_status['current_threshold']:.2f}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completado!")

if __name__ == "__main__":
    main()