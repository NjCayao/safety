"""
Test del Sistema de Reconocimiento Facial
"""

import cv2
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_recognition import IntegratedFaceSystem

def main():
    print("\n=== TEST SISTEMA DE RECONOCIMIENTO FACIAL ===")
    print("Inicializando...")
    
    # Crear instancia del sistema
    face_system = IntegratedFaceSystem(
        operators_dir="operators",
        dashboard_position='right'  # Puedes cambiar a 'left'
    )
    
    print(f"✅ Sistema inicializado")
    print(f"✅ Operadores cargados: {len(face_system.recognizer.operators)}")
    
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
    print("  • r: Forzar reporte de operador desconocido (testing)")
    print("\n")
    
    dashboard_visible = True
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        result = face_system.identify_and_analyze(frame)
        
        # Mostrar información en consola cada segundo (30 frames aprox)
        if frame_count % 30 == 0:
            status = face_system.get_current_status()
            
            if result['operator_info']:
                op = result['operator_info']
                if op.get('is_registered', False):
                    print(f"\r✅ Operador: {op['name']} | "
                          f"Calibrado: {'SÍ' if result['is_calibrated'] else 'NO'} | "
                          f"Confianza: {op['confidence']:.2f}    ", end='')
                else:
                    unknown_time = status.get('unknown_operator_time', 0)
                    minutes = int(unknown_time / 60)
                    seconds = int(unknown_time % 60)
                    print(f"\r⚠️  OPERADOR NO REGISTRADO | "
                          f"Tiempo: {minutes}:{seconds:02d} | "
                          f"Reporte en: {max(0, 15-minutes)} min    ", end='')
            else:
                print(f"\r⏸️  Sin detección de rostro                              ", end='')
        
        frame_count += 1
        
        # Mostrar frame con dashboard
        cv2.imshow('Test Reconocimiento Facial', result['frame'])
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            # Cambiar posición del dashboard
            new_pos = 'left' if face_system.dashboard.position == 'right' else 'right'
            face_system.set_dashboard_position(new_pos)
            print(f"\n📊 Dashboard movido a: {new_pos}")
        elif key == ord('h'):
            # Ocultar/mostrar dashboard
            dashboard_visible = not dashboard_visible
            face_system.enable_dashboard(dashboard_visible)
            print(f"\n📊 Dashboard: {'visible' if dashboard_visible else 'oculto'}")
        elif key == ord('r'):
            # Forzar reporte de operador desconocido
            if face_system.force_unknown_operator_report(result['frame']):
                print("\n📨 REPORTE FORZADO: Se generó reporte de operador desconocido")
                print("   Revisa la carpeta 'reports/face_recognition/'")
            else:
                print("\n⚠️  No hay operador desconocido activo para reportar")
    
    # Al salir, mostrar resumen
    print("\n\n=== RESUMEN DE LA SESIÓN ===")
    
    status = face_system.get_current_status()
    stats = status['session_stats']
    
    print(f"Total de análisis: {stats['total_recognitions']}")
    print(f"Reconocimientos exitosos: {stats['successful_recognitions']}")
    print(f"Detecciones desconocidas: {stats['unknown_detections']}")
    print(f"Operadores detectados: {len(stats['operators_detected'])}")
    print(f"Reportes de desconocido: {stats['unknown_operator_reports']}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completado!")

if __name__ == "__main__":
    main()