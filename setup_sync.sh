#!/bin/bash

# Script para configurar el sistema de sincronización

echo "Configurando sistema de sincronización online/offline..."

# Crear directorios necesarios
mkdir -p client/logs
mkdir -p client/images/fatigue
mkdir -p client/images/cellphone
mkdir -p client/images/smoking
mkdir -p client/images/unrecognized_operator

# Instalar dependencias
pip install -r client/requirements.txt

# Configurar permisos
chmod -R 755 client/images

echo "Configuración completada."
echo "Para iniciar el sistema con sincronización, ejecute: python main_system_wrapper.py"