import cv2
import time

def test_camera():
    print("Iniciando prueba de cámara...")
    
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)  # 0 para la cámara principal
    
    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara")
        return False
    
    # Configurar la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    print("Cámara inicializada correctamente")
    print("Mostrando vista previa durante 10 segundos...")
    
    # Tiempo de inicio
    start_time = time.time()
    
    while time.time() - start_time < 10:  # Ejecutar durante 10 segundos
        # Capturar un frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo capturar frame")
            break
        
        # Mostrar el frame
        cv2.imshow('Prueba de Cámara', frame)
        
        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    print("Prueba de cámara finalizada")
    return True

if __name__ == "__main__":
    test_camera()