"""
Test del Sistema de Detección de Distracciones
"""

import cv2
import sys
import os
import dlib

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distraction import IntegratedDistractionSystem

def main():
    print("\n=== TEST SISTEMA DE DETECCIÓN DE DISTRACCIONES ===")
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
    distraction_system = IntegratedDistractionSystem(
        operators_dir="operators",
        dashboard_position='right'
    )
    
    print(f"✅ Sistema inicializado")
    
    # Configurar operador
    test_operator = {
        'id': '47469940',  # Cambia esto por el DNI de un operador real
        'name': 'Operador Real'
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
    
    distraction_system.set_operator(test_operator)
    
    status = distraction_system.get_current_status()
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
    print("  • r: Reiniciar contador de distracciones")
    print("  • f: Forzar reporte de múltiples distracciones (testing)")
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
        
        # Procesar frame con el sistema de distracciones
        result = distraction_system.analyze_frame(frame, landmarks)
        
        # Mostrar información en consola cada segundo (30 frames aprox)
        if frame_count % 30 == 0:
            detector_status = result.get('detector_status', {})
            
            if detector_status:
                direction = detector_status.get('direction', 'CENTRO')
                alert_level = detector_status.get('current_alert_level', 0)
                distraction_time = detector_status.get('distraction_time', 0)
                
                if result.get('is_distracted', False):
                    level_str = f"Nivel {alert_level}" if alert_level > 0 else "Iniciando"
                    print(f"\r🟡 DISTRACCIÓN ({direction}) | "
                          f"{level_str} | "
                          f"Tiempo: {distraction_time:.1f}s | "
                          f"Total: {result['total_distractions']}/{result['max_distractions']}    ", end='')
                else:
                    print(f"\r✅ Atento (CENTRO) | "
                          f"Distracciones: {result['total_distractions']}/{result['max_distractions']}    ", end='')
            else:
                print(f"\r⏸️  Sin detección de rostro                              ", end='')
        
        # Mostrar alerta especial si hay múltiples distracciones
        if result.get('multiple_distractions', False):
            if frame_count % 15 == 0:  # Parpadeo cada medio segundo
                print(f"\n⚠️  ALERTA: {result['total_distractions']} distracciones detectadas en {result['window_minutes']} minutos!")
        
        frame_count += 1
        
        # Mostrar frame con dashboard
        cv2.imshow('Test Detección de Distracciones', result['frame'])
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Cambiar posición del dashboard
            new_pos = 'left' if distraction_system.dashboard.position == 'right' else 'right'
            distraction_system.set_dashboard_position(new_pos)
            print(f"\n📊 Dashboard movido a: {new_pos}")
        elif key == ord('h'):
            # Ocultar/mostrar dashboard
            dashboard_visible = not dashboard_visible
            distraction_system.enable_dashboard(dashboard_visible)
            print(f"\n📊 Dashboard: {'visible' if dashboard_visible else 'oculto'}")
        elif key == ord('r'):
            # Reiniciar contador
            distraction_system.reset_distraction_counter()
            print("\n🔄 Contador de distracciones reiniciado")
        elif key == ord('f'):
            # Forzar reporte
            if distraction_system.force_distraction_report(result['frame']):
                print("\n📨 REPORTE FORZADO: Se generó reporte de múltiples distracciones")
                print("   Revisa la carpeta 'reports/distraction/'")
            else:
                print("\n⚠️  No hay distracciones registradas para reportar")
    
    # Al salir, mostrar resumen
    print("\n\n=== RESUMEN DE LA SESIÓN ===")
    
    status = distraction_system.get_current_status()
    stats = status['session_stats']
    
    print(f"Total de análisis: {stats['total_detections']}")
    print(f"Distracciones detectadas: {stats['total_distractions']}")
    print(f"Eventos nivel 1: {stats['level1_events']}")
    print(f"Eventos nivel 2: {stats['level2_events']}")
    print(f"Eventos de múltiples distracciones: {stats['multiple_distraction_events']}")
    print(f"Reportes generados: {stats['reports_generated']}")
    
    # Información adicional
    if stats['total_distractions'] > 0:
        avg_time = stats['total_detections'] / stats['total_distractions']
        print(f"Promedio: 1 distracción cada {avg_time:.0f} frames ({avg_time/30:.1f} segundos)")
    
    # Estado del detector
    detector_status = status['detector_status']
    print(f"\nEstado final:")
    print(f"  • Modo: {'NOCTURNO' if detector_status['is_night_mode'] else 'DIURNO'}")
    print(f"  • Nivel de luz: {detector_status['light_level']:.1f}")
    print(f"  • Última dirección: {detector_status['direction']}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completado!")

if __name__ == "__main__":
    main()