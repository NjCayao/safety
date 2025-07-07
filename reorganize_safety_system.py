#!/usr/bin/env python3
"""
Script de Reorganización para Safety System
==========================================
Este script reorganiza la estructura del proyecto de forma segura,
manteniendo toda la funcionalidad intacta.

IMPORTANTE: Hace backup automático antes de cualquier cambio.
"""

import os
import shutil
import datetime
import json
from pathlib import Path

class SafetySystemReorganizer:
    def __init__(self):
        self.root_dir = os.getcwd()
        self.backup_dir = None
        self.moves_log = []
        self.errors = []
        
        # Definir la nueva estructura
        self.new_structure = {
            'core': [
                'camera_module.py',
                'face_recognition_module.py',
                'fatigue_detection.py',
                'bostezo_detection.py',
                'distraction_detection.py',
                'behavior_detection_module.py',
                'alarm_module.py',
                'report_generator.py'
            ],
            'sync': [
                'config_sync_client.py',
                'device_auth.py',
                'heartbeat_sender.py'
            ],
            'sync/wrappers': [
                'behavior_detection_wrapper.py',
                'fatigue_detection_wrapper.py',
                'face_recognition_wrapper.py'
            ],
            'scripts': [
                'register_operator.py',
                'process_photos.py'
            ],
            'assets/audio': 'audio',  # mover carpeta completa
            'assets/models': 'models',  # mover carpeta completa
            'assets/operators': 'operators',  # mover carpeta completa
            'output/reports': 'reports',  # mover carpeta completa
            'output/logs': 'logs'  # mover carpeta completa
        }
        
        # Archivos a eliminar (duplicados y obsoletos)
        self.files_to_delete = [
            'fatigue_adapter.py',
            'behavior_adapter.py',
            'face_recognition_adapter.py',
            'sync_integrator.py',
            'main_with_sync.py',
            'cp_errordocument.shtml',
            'SYNC_INTEGRATION_GUIDE.md',
            'setup_sync.sh'
        ]
        
        # Carpetas vacías a eliminar
        self.folders_to_delete = [
            'expresiones_faciales'  # Ya confirmaste que la eliminaste
        ]

    def create_backup(self):
        """Crear backup completo del proyecto"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = os.path.join(os.path.dirname(self.root_dir), f"safety_system_backup_{timestamp}")
        
        print(f"🔄 Creando backup en: {self.backup_dir}")
        
        try:
            # Ignorar archivos innecesarios en el backup
            ignore_patterns = shutil.ignore_patterns(
                '*.pyc', '__pycache__', '.git', 'venv', '*.log', '.DS_Store'
            )
            
            shutil.copytree(self.root_dir, self.backup_dir, ignore=ignore_patterns)
            print("✅ Backup creado exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error creando backup: {e}")
            return False

    def create_directories(self):
        """Crear la nueva estructura de directorios"""
        print("\n📁 Creando nueva estructura de directorios...")
        
        directories = [
            'core',
            'sync',
            'sync/wrappers',
            'scripts',
            'assets',
            'assets/audio',
            'assets/models',
            'assets/operators',
            'output',
            'output/reports',
            'output/logs'
        ]
        
        for dir_path in directories:
            full_path = os.path.join(self.root_dir, dir_path)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"  ✅ Creado: {dir_path}/")

    def move_files(self):
        """Mover archivos a su nueva ubicación"""
        print("\n📦 Moviendo archivos...")
        
        for dest_dir, files in self.new_structure.items():
            dest_path = os.path.join(self.root_dir, dest_dir)
            
            if isinstance(files, list):
                # Mover archivos individuales
                for file_name in files:
                    src = os.path.join(self.root_dir, file_name)
                    dst = os.path.join(dest_path, file_name)
                    
                    if os.path.exists(src) and not os.path.exists(dst):
                        try:
                            shutil.move(src, dst)
                            self.moves_log.append(f"{file_name} -> {dest_dir}/")
                            print(f"  ✅ {file_name} -> {dest_dir}/")
                        except Exception as e:
                            self.errors.append(f"Error moviendo {file_name}: {e}")
                            print(f"  ❌ Error moviendo {file_name}: {e}")
            
            elif isinstance(files, str):
                # Mover carpeta completa
                src = os.path.join(self.root_dir, files)
                if os.path.exists(src) and os.path.isdir(src):
                    # Si la carpeta destino ya existe, mover contenido
                    if os.path.exists(dest_path):
                        for item in os.listdir(src):
                            src_item = os.path.join(src, item)
                            dst_item = os.path.join(dest_path, item)
                            if not os.path.exists(dst_item):
                                shutil.move(src_item, dst_item)
                        # Eliminar carpeta origen vacía
                        if not os.listdir(src):
                            os.rmdir(src)
                    else:
                        shutil.move(src, dest_path)
                    
                    self.moves_log.append(f"{files}/ -> {dest_dir}/")
                    print(f"  ✅ {files}/ -> {dest_dir}/")

    def update_imports(self):
        """Actualizar imports en archivos principales"""
        print("\n🔧 Actualizando imports...")
        
        # Definir las actualizaciones de imports
        import_updates = {
            'main_system.py': [
                ('from camera_module import', 'from core.camera_module import'),
                ('from face_recognition_module import', 'from core.face_recognition_module import'),
                ('from fatigue_detection import', 'from core.fatigue_detection import'),
                ('from bostezo_detection import', 'from core.bostezo_detection import'),
                ('from distraction_detection import', 'from core.distraction_detection import'),
                ('from alarm_module import', 'from core.alarm_module import'),
                ('from behavior_detection_module import', 'from core.behavior_detection_module import'),
            ],
            'main_system_wrapper.py': [
                ('from behavior_detection_wrapper import', 'from sync.wrappers.behavior_detection_wrapper import'),
                ('from fatigue_detection_wrapper import', 'from sync.wrappers.fatigue_detection_wrapper import'),
                ('from face_recognition_wrapper import', 'from sync.wrappers.face_recognition_wrapper import'),
            ],
            'sync/wrappers/behavior_detection_wrapper.py': [
                ('from behavior_detection_module import', 'from core.behavior_detection_module import'),
            ],
            'sync/wrappers/fatigue_detection_wrapper.py': [
                ('from fatigue_detection import', 'from core.fatigue_detection import'),
            ],
            'sync/wrappers/face_recognition_wrapper.py': [
                ('from face_recognition import', 'from core.face_recognition_module import'),
            ]
        }
        
        for file_path, replacements in import_updates.items():
            full_path = os.path.join(self.root_dir, file_path)
            
            if os.path.exists(full_path):
                try:
                    # Leer archivo
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Aplicar reemplazos
                    original_content = content
                    for old_import, new_import in replacements:
                        content = content.replace(old_import, new_import)
                    
                    # Guardar si hubo cambios
                    if content != original_content:
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"  ✅ Actualizado: {file_path}")
                    
                except Exception as e:
                    self.errors.append(f"Error actualizando {file_path}: {e}")
                    print(f"  ❌ Error actualizando {file_path}: {e}")

    def update_path_references(self):
        """Actualizar referencias a rutas en el código"""
        print("\n🔧 Actualizando referencias de rutas...")
        
        # Archivos que pueden contener rutas
        files_to_check = [
            'main_system.py',
            'core/face_recognition_module.py',
            'core/behavior_detection_module.py',
            'scripts/register_operator.py',
            'scripts/process_photos.py'
        ]
        
        # Mapeo de rutas antiguas a nuevas
        path_updates = {
            '"operators"': '"assets/operators"',
            "'operators'": "'assets/operators'",
            '"models"': '"assets/models"',
            "'models'": "'assets/models'",
            '"audio"': '"assets/audio"',
            "'audio'": "'assets/audio'",
            '"reports"': '"output/reports"',
            "'reports'": "'output/reports'",
            '"logs"': '"output/logs"',
            "'logs'": "'output/logs'",
            'OPERATORS_DIR = os.path.join(BASE_DIR, "operators")': 'OPERATORS_DIR = os.path.join(BASE_DIR, "assets/operators")',
            'MODEL_DIR = os.path.join(BASE_DIR, "models")': 'MODEL_DIR = os.path.join(BASE_DIR, "assets/models")',
            'AUDIO_DIR = os.path.join(BASE_DIR, "audio")': 'AUDIO_DIR = os.path.join(BASE_DIR, "assets/audio")',
            'REPORTS_DIR = os.path.join(BASE_DIR, "reports")': 'REPORTS_DIR = os.path.join(BASE_DIR, "output/reports")',
            'LOGS_DIR = os.path.join(BASE_DIR, "logs")': 'LOGS_DIR = os.path.join(BASE_DIR, "output/logs")',
        }
        
        for file_path in files_to_check:
            full_path = os.path.join(self.root_dir, file_path)
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    for old_path, new_path in path_updates.items():
                        content = content.replace(old_path, new_path)
                    
                    if content != original_content:
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        print(f"  ✅ Rutas actualizadas en: {file_path}")
                
                except Exception as e:
                    self.errors.append(f"Error actualizando rutas en {file_path}: {e}")

    def delete_obsolete_files(self):
        """Eliminar archivos obsoletos y duplicados"""
        print("\n🗑️  Eliminando archivos obsoletos...")
        
        for file_name in self.files_to_delete:
            file_path = os.path.join(self.root_dir, file_name)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"  ✅ Eliminado: {file_name}")
                except Exception as e:
                    self.errors.append(f"Error eliminando {file_name}: {e}")
                    print(f"  ❌ Error eliminando {file_name}: {e}")

    def create_init_files(self):
        """Crear archivos __init__.py en los nuevos paquetes"""
        print("\n📝 Creando archivos __init__.py...")
        
        packages = ['core', 'sync', 'sync/wrappers', 'scripts', 'assets']
        
        for package in packages:
            init_path = os.path.join(self.root_dir, package, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write('# -*- coding: utf-8 -*-\n')
                print(f"  ✅ Creado: {package}/__init__.py")

    def generate_report(self):
        """Generar reporte de la reorganización"""
        report_path = os.path.join(self.root_dir, 'reorganization_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== REPORTE DE REORGANIZACIÓN ===\n")
            f.write(f"Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backup creado en: {self.backup_dir}\n\n")
            
            f.write("MOVIMIENTOS REALIZADOS:\n")
            for move in self.moves_log:
                f.write(f"  - {move}\n")
            
            if self.errors:
                f.write("\nERRORES ENCONTRADOS:\n")
                for error in self.errors:
                    f.write(f"  - {error}\n")
            
            f.write("\nNUEVA ESTRUCTURA:\n")
            f.write("""
safety_system/
├── core/                    # Módulos principales
├── sync/                    # Sistema de sincronización
│   └── wrappers/           # Wrappers para sincronización
├── config/                  # Configuración (sin cambios)
├── client/                  # Cliente de sincronización (sin cambios)
├── server/                  # Dashboard PHP (sin cambios)
├── scripts/                 # Scripts auxiliares
├── assets/                  # Recursos
│   ├── audio/
│   ├── models/
│   └── operators/
├── output/                  # Salidas del sistema
│   ├── reports/
│   └── logs/
├── main_system.py          # Sistema principal
└── main_system_wrapper.py  # Sistema con sincronización
""")
        
        print(f"\n📄 Reporte guardado en: {report_path}")

    def run(self):
        """Ejecutar la reorganización completa"""
        print("🚀 INICIANDO REORGANIZACIÓN DEL SAFETY SYSTEM")
        print("=" * 50)
        
        # Paso 1: Crear backup
        if not self.create_backup():
            print("❌ No se pudo crear el backup. Abortando operación.")
            return False
        
        # Paso 2: Crear directorios
        self.create_directories()
        
        # Paso 3: Mover archivos
        self.move_files()
        
        # Paso 4: Actualizar imports
        self.update_imports()
        
        # Paso 5: Actualizar referencias de rutas
        self.update_path_references()
        
        # Paso 6: Eliminar archivos obsoletos
        self.delete_obsolete_files()
        
        # Paso 7: Crear archivos __init__.py
        self.create_init_files()
        
        # Paso 8: Generar reporte
        self.generate_report()
        
        print("\n" + "=" * 50)
        if self.errors:
            print(f"⚠️  Reorganización completada con {len(self.errors)} errores.")
            print("   Revisa el reporte para más detalles.")
        else:
            print("✅ ¡Reorganización completada exitosamente!")
        
        print(f"\n💾 Backup guardado en: {self.backup_dir}")
        print("📌 Recomendación: Prueba el sistema antes de eliminar el backup.")
        
        return True

def main():
    """Función principal"""
    print("╔═══════════════════════════════════════════════════╗")
    print("║     SAFETY SYSTEM - SCRIPT DE REORGANIZACIÓN      ║")
    print("╚═══════════════════════════════════════════════════╝")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('main_system.py'):
        print("\n❌ Error: Este script debe ejecutarse desde la raíz del proyecto Safety System.")
        print("   No se encontró 'main_system.py' en el directorio actual.")
        return
    
    # Confirmar con el usuario
    print("\nEste script:")
    print("  1. Creará un backup completo del proyecto")
    print("  2. Reorganizará la estructura de archivos")
    print("  3. Actualizará los imports automáticamente")
    print("  4. Eliminará archivos duplicados y obsoletos")
    print("  5. Generará un reporte detallado")
    
    response = input("\n¿Deseas continuar? (s/n): ").lower()
    
    if response == 's':
        reorganizer = SafetySystemReorganizer()
        reorganizer.run()
    else:
        print("Operación cancelada.")

if __name__ == "__main__":
    main()