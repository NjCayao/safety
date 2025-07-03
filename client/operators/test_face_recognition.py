import sys
print("Python versión:", sys.version)

try:
    import face_recognition
    print("La biblioteca face_recognition está instalada correctamente")
    print("Versión:", face_recognition.__version__ if hasattr(face_recognition, '__version__') else "Desconocida")
except ImportError as e:
    print("ERROR: No se pudo importar face_recognition")
    print("Detalles:", str(e))
    print("
Por favor, instale la biblioteca con: pip install face_recognition")
    
    try:
        # Intentar importar otras bibliotecas comunes para verificar entorno
        print("
Comprobando otras bibliotecas:")
        import numpy
        print("- numpy: OK")
    except ImportError:
        print("- numpy: NO INSTALADO")
        
    try:
        import cv2
        print("- cv2 (OpenCV): OK")
    except ImportError:
        print("- cv2 (OpenCV): NO INSTALADO")
        
    try:
        import dlib
        print("- dlib: OK")
    except ImportError:
        print("- dlib: NO INSTALADO (requerido por face_recognition)")
except Exception as e:
    print("Error desconocido:", str(e))