"""
Diagnóstico de Importaciones
===========================
Identifica problemas de importación circular.
"""

import sys
import os

# Agregar el directorio al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🔍 Diagnóstico de importaciones del sistema de análisis\n")

# Verificar archivos
files_to_check = [
    "core/analysis/__init__.py",
    "core/analysis/calibration_manager.py",
    "core/analysis/fatigue_detector.py",
    "core/analysis/stress_analyzer.py",
    "core/analysis/pulse_estimator.py",
    "core/analysis/emotion_analyzer.py",
    "core/analysis/anomaly_detector.py",
    "core/analysis/analysis_dashboard.py",
    "core/analysis/integrated_analysis_system.py"
]

print("1. Verificando existencia de archivos:")
for file in files_to_check:
    exists = os.path.exists(file)
    print(f"   {'✅' if exists else '❌'} {file}")

print("\n2. Intentando importar módulos individualmente:")

# Probar cada módulo por separado
modules_to_test = [
    ("CalibrationManager", "core.analysis.calibration_manager"),
    ("FatigueDetector", "core.analysis.fatigue_detector"),
    ("StressAnalyzer", "core.analysis.stress_analyzer"),
    ("PulseEstimator", "core.analysis.pulse_estimator"),
    ("EmotionAnalyzer", "core.analysis.emotion_analyzer"),
    ("AnomalyDetector", "core.analysis.anomaly_detector"),
    ("AnalysisDashboard", "core.analysis.analysis_dashboard"),
    ("IntegratedAnalysisSystem", "core.analysis.integrated_analysis_system"),
]

successful_imports = []
failed_imports = []

for class_name, module_path in modules_to_test:
    try:
        # Intentar importar el módulo
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        successful_imports.append(class_name)
        print(f"   ✅ {class_name} desde {module_path}")
    except Exception as e:
        failed_imports.append((class_name, str(e)))
        print(f"   ❌ {class_name}: {e}")

print(f"\n3. Resumen:")
print(f"   ✅ Importaciones exitosas: {len(successful_imports)}")
print(f"   ❌ Importaciones fallidas: {len(failed_imports)}")

if failed_imports:
    print("\n4. Detalles de errores:")
    for class_name, error in failed_imports:
        print(f"\n   {class_name}:")
        print(f"   {error}")

# Probar importación completa del módulo
print("\n5. Intentando importar el módulo completo:")
try:
    import core.analysis
    print("   ✅ Módulo core.analysis importado correctamente")
    
    # Verificar qué está disponible
    print("\n6. Clases disponibles en core.analysis:")
    for attr in dir(core.analysis):
        if not attr.startswith('_'):
            print(f"   - {attr}")
            
except Exception as e:
    print(f"   ❌ Error importando core.analysis: {e}")

# Verificar dependencias específicas
print("\n7. Verificando dependencias externas:")
dependencies = [
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("dlib", "dlib"),
    ("face_recognition", "face-recognition")
]

for module_name, package_name in dependencies:
    try:
        __import__(module_name)
        print(f"   ✅ {module_name}")
    except ImportError:
        print(f"   ❌ {module_name} - Instalar con: pip install {package_name}")

print("\n8. Revisando imports en anomaly_detector.py:")
try:
    with open("core/analysis/anomaly_detector.py", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        imports = [line.strip() for line in lines if line.strip().startswith(('import', 'from'))]
        
    print("   Imports encontrados:")
    for imp in imports[:10]:  # Primeros 10 imports
        print(f"   - {imp}")
        
except Exception as e:
    print(f"   ❌ Error leyendo archivo: {e}")

print("\n✨ Diagnóstico completado")