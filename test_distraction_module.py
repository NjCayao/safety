"""
Test Mejorado del Sistema de Detección de Distracciones
Con diagnóstico de problemas comunes
"""

import cv2
import sys
import os
import dlib
import time

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.distraction import IntegratedDistractionSystem

def test_camera():
    """Prueba la cámara antes de iniciar el sistema"""
    print("\n🎥 Probando cámara...")
    
    # Probar diferentes índices de cámara
    for index in [0, 1, 2]:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"✅ Cámara {index} funciona: {w}x{h}")
                cap.release()
                return index
            cap.release()
    
    print("❌ No se encontró ninguna cámara funcional")
    return None

def main():
    print("\n=== TEST SISTEMA DE DETECCIÓN DE DISTRACCIONES ===")
    print("Versión mejorada con diagnóstico")
    
    # 1. Verificar cámara primero
    camera_index = test_camera()
    if camera_index is None:
        print("\n❌ No se puede continuar sin cámara")
        print("Posibles soluciones:")
        print("  • Verifica que tu cámara esté conectada")
        print("  • Cierra otras aplicaciones que usen la cámara")
        print("  • Verifica los permisos de la cámara")
        return
    
    # 2. Verificar modelo de landmarks
    print("\n📁 Verificando modelo de landmarks...")
    model_path = "assets/models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(model_path):
        print(f"❌ Error: No se encuentra el modelo en {model_path}")
        print("Descárgalo de: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    else:
        print("✅ Modelo encontrado")
    
    # 3. Inicializar detectores
    print("\n🔧 Inicializando detectores...")
    try:
        face_detector = dlib.get_frontal_face_detector()
        landmark_predictor = dlib.shape_predictor(model_path)
        print("✅ Detectores inicializados")
    except Exception as e:
        print(f"❌ Error al inicializar detectores: {e}")
        return
    
    # 4. Crear sistema de distracciones
    print("\n🚀 Inicializando sistema de distracciones...")
    try:
        distraction_system = IntegratedDistractionSystem(
            operators_dir="operators",
            dashboard_position='right'
        )
        print("✅ Sistema inicializado")
    except Exception as e:
        print(f"❌ Error al inicializar sistema: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. Configurar operador de prueba
    test_operator = {
        'id': 'test_operator',
        'name': 'Operador de Prueba'
    }
    
    # Buscar operador real si existe
    if os.path.exists("operators"):
        operators = [d for d in os.listdir("operators") if os.path.isdir(os.path.join("operators", d))]
        if operators:
            print(f"\n📋 Operadores disponibles: {operators}")
            # Usar el primer operador encontrado
            test_operator['id'] = operators[0]
            operator_path = os.path.join("operators", test_operator['id'])
            info_file = os.path.join(operator_path, "info.txt")
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        test_operator['name'] = lines[1].strip()
    
    distraction_system.set_operator(test_operator)
    
    status = distraction_system.get_current_status()
    print(f"\n👤 Operador: {test_operator['name']} (ID: {test_operator['id']})")
    print(f"📊 Calibración: {'PERSONALIZADA' if status['is_calibrated'] else 'POR DEFECTO'}")
    
    # 6. Configurar cámara con parámetros optimizados
    print(f"\n📷 Abriendo cámara {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verificar que la cámara funciona
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print("❌ Error: La cámara no está capturando frames")
        cap.release()
        return
    
    print("✅ Cámara funcionando correctamente")
    print(f"   Resolución: {test_frame.shape[1]}x{test_frame.shape[0]}")
    
    print("\n" + "="*50)
    print("CONTROLES:")
    print("  • q: Salir")
    print("  • d: Cambiar posición del dashboard")
    print("  • h: Ocultar/mostrar dashboard")
    print("  • r: Reiniciar contador de distracciones")
    print("  • f: Forzar reporte (testing)")
    print("  • ESPACIO: Pausar/reanudar")
    print("="*50 + "\n")
    
    dashboard_visible = True
    frame_count = 0
    fps_time = time.time()
    fps = 0
    paused = False
    last_frame = None
    
    print("🎯 Sistema listo. Mueve tu cabeza hacia los lados para probar.\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("\n❌ Error al capturar frame")
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            last_frame = frame.copy()
        else:
            if last_frame is not None:
                frame = last_frame.copy()
            else:
                continue
        
        # Calcular FPS
        if frame_count % 30 == 0:
            current_time = time.time()
            if current_time - fps_time > 0:
                fps = 30 / (current_time - fps_time)
                fps_time = current_time
        
        # Detectar rostros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        
        landmarks = None
        if faces:
            # Dibujar rectángulo del rostro
            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Obtener landmarks
            landmarks = landmark_predictor(gray, face)
            
            # Dibujar algunos puntos clave para verificar
            for i in [30, 8, 36, 45]:  # Nariz, mentón, ojos
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        
        # Procesar con el sistema de distracciones
        result = distraction_system.analyze_frame(frame, landmarks)
        
        # Agregar información de debug en la imagen
        debug_frame = result['frame']
        cv2.putText(debug_frame, f"FPS: {fps:.1f}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if paused:
            cv2.putText(debug_frame, "PAUSADO", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Mostrar estado en consola
        if frame_count % 30 == 0 and not paused:
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
                          f"Total: {result['total_distractions']}/{result['max_distractions']} | "
                          f"FPS: {fps:.1f}    ", end='', flush=True)
                else:
                    print(f"\r✅ Atento (CENTRO) | "
                          f"Distracciones: {result['total_distractions']}/{result['max_distractions']} | "
                          f"FPS: {fps:.1f}    ", end='', flush=True)
            else:
                print(f"\r⏸️  Sin detección de rostro | FPS: {fps:.1f}                    ", end='', flush=True)
        
        # Alertas especiales
        if result.get('multiple_distractions', False):
            if frame_count % 15 == 0:
                print(f"\n⚠️  ALERTA: {result['total_distractions']} distracciones en {result['window_minutes']} minutos!")
        
        frame_count += 1
        
        # Mostrar frame
        cv2.imshow('Test Detección de Distracciones', debug_frame)
        
        # Procesar teclas
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('d'):
            new_pos = 'left' if distraction_system.dashboard.position == 'right' else 'right'
            distraction_system.set_dashboard_position(new_pos)
            print(f"\n📊 Dashboard movido a: {new_pos}")
        elif key == ord('h'):
            dashboard_visible = not dashboard_visible
            distraction_system.enable_dashboard(dashboard_visible)
            print(f"\n📊 Dashboard: {'visible' if dashboard_visible else 'oculto'}")
        elif key == ord('r'):
            distraction_system.reset_distraction_counter()
            print("\n🔄 Contador reiniciado")
        elif key == ord('f'):
            if distraction_system.force_distraction_report(debug_frame):
                print("\n📨 Reporte generado en 'reports/distraction/'")
        elif key == ord(' '):
            paused = not paused
            print(f"\n⏸️  {'PAUSADO' if paused else 'REANUDADO'}")
    
    # Resumen final
    print("\n\n" + "="*50)
    print("RESUMEN DE LA SESIÓN")
    print("="*50)
    
    status = distraction_system.get_current_status()
    stats = status['session_stats']
    
    print(f"Total frames procesados: {frame_count}")
    print(f"Distracciones detectadas: {stats['total_distractions']}")
    print(f"Reportes generados: {stats['reports_generated']}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completado!")

if __name__ == "__main__":
    main()