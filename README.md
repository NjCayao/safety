# Safety System
⚠️ PROYECTO EN DESARROLLO - NO USAR EN PRODUCCIÓN
© 2025 www.DevCayao.com - Todos los derechos reservados
Código educacional - No autorizado para uso comercial.

# INSTALAR PYTHON 3.9 
INSTLAR CMAKE     https://cmake.org/download/
Crear un entorno virtual de Python para el desarrollo: 
python -m venv venv
# En Windows:
venv\Scripts\activate

# Instalar dependencias de desarrollo: 
pip install opencv-python dlib face_recognition numpy imutils
pip install tensorflow pillow fpdf requests paho-mqtt
pip install pytest flask # Para pruebas y desarrollo del API

pip install requests
pip install pytz


# estructura

safety_system/
├── audio/
├── docs/
├── logs/
├── models/
├── operator-photo/
├── operators/
├── reports/
├── config/                     # 🆕 NUEVO
│   ├── config_manager.py       # Sistema de configuración inteligente
│   ├── default.yaml           # Configuración por defecto (segura)
│   ├── development.yaml       # Para tu Windows (con GUI)
│   └── production.yaml        # Para Raspberry Pi (sin GUI)
├── client/
    ├── config/
    │   ├── config.ini        # Configuración general
    │   └── logging.ini       # Configuración de logging
    ├── db/
    │   └── local_storage.py  # Base de datos SQLite para almacenamiento local
    ├── api/
    │   ├── api_client.py     # Cliente de API para comunicación con el servidor
    │   ├── auth.py           # Manejo de autenticación
    │   └── sync.py           # Sincronización de datos
    ├── utils/
    │   ├── connection.py     # Manejo de estado de conexión
    │   ├── file_manager.py   # Gestión de archivos locales
    │   ├── event_manager.py   # 
    │   └── logger.py         # Sistema de logging
    ├── main.py               # Punto de entrada principal
    └── requirements.txt      # Dependencias
    ├── models/              # Modelos de ML para detección
    ├── utils/               # Funciones auxiliares
    ├── config/              # Configuraciones
    ├── audio/               # Archivos de audio para alertas
│   ├── sync/                            # 🆕 NUEVO - Comunicación Pi ↔ Server
│   │   ├── config_sync_client.py        #  Cliente sincronización
│   │   ├── device_auth.py               #  Autenticación de dispositivo
│   │   └── heartbeat_sender.py          #  Envío de heartbeats
│   
│
├── server/
    ├── assets/            # Archivos CSS, JS, imágenes de AdminLTE -- modulo echo
    |   ├── js/
            └── device-config.js
            ├── realtime-alerts.js
    ├── config/            # Archivos de configuración
    │   ├── database.php   # Conexión a la base de datos  -- modulo echo
    │   └── config.php     # Configuraciones generales -- modulo echo
    ├── includes/          # Código PHP reutilizable
        └── charts/
            ├── alerts_by_type.php # Gráfico de alertas por tipo
            ├── alerts_by_operator.php # Gráfico de alertas por operador
            └── alerts_trend.php   # Gráfico de tendencias de alertas
        ├── siderbar.php    # modulo echo
        ├── auth.php   
        ├── alerts.php   # mensajes de éxito/error
    │   ├── header.php     # Encabezado común de AdminLTE -- modulo echo
    │   ├── sidebar.php    # Barra lateral de AdminLTE -- modulo echo
    │   ├── footer.php     # Pie de página común -- modulo echo
    │   └── functions.php  # Funciones auxiliares -- modulo echo
    │   ├── content.php     # Pie de página común -- modulo echo
        ├── device_config.php        # ✅ PASO 2 - Funciones backend
    │   └── photo_functions.php  # Funciones auxiliares -- modulo echo
    ├── models/            # Lógica para interactuar con la base de datos
    │   ├── Operator.php   # Clase para gestionar operadores
    │   ├── Alert.php      # Clase para gestionar alertas
    │   └── Report.php     # Clase para gestionar reportes
    ├── operador-photos/   # modulo fotos de cada operador y sub carpeta con su dni
    ├── api/                              # Endpoints para la Raspberry Pi
    │   ├── v1/                           # Versionar la API es una buena práctica
    │   │   ├── auth/
    │   │   │   ├── authenticate.php      # Para autenticar dispositivos
    │   │   │   └── verify.php            # Para verificar tokens
    │   │   ├── alerts/
    │   │   │   ├── recent.php      
    │   │   ├── dashboard/
    │   │   │   ├── stats.php                    # Datos estadísticos generales
    │   │   │   ├── chart_data.php    # Datos para gráficos                   
    │   │   │   └── recent_alerts.php            # Datos para gráficos
    │   │   ├── events/
    │   │   │   ├── create.php            # Para recibir eventos nuevos
    │   │   │   └── index.php             # Para listar eventos
    │   │   │   └── upload_image.php             # 
    │   │   ├── operators/
    │   │   │   └── sync.php              # Para sincronizar operadores
    │   │   ├── sync/
    │   │   │   ├── batch.php             # Para recibir lotes de eventos pendientes
    │   │   │   ├── confirm.php           # Para confirmar sincronización
    │   │   │   └── status.php            # Para verificar estado de sincronización
    │   │   └── devices/
    │   │       ├── register.php          # Para registrar dispositivos nuevos
    │   │       └── status.php            # Para actualizar/verificar estado
                ├── heartbeat.php         # 
                |-- config.php               # ✅ PASO 3 - API Pi ↔ Server
    │   ├── config/
    │   │   ├── core.php                  # Configuración central de la API
    │   │   ├── database.php              # Configuración de base de datos
    │   │   ├── api_config.php            # 
    │   │   └── headers.php               # Configuración de cabeceras HTTP
    │   ├── models/
    │   │   ├── Device.php                # Modelo para dispositivos
    │   │   ├── Event.php                 # Modelo para eventos
    │   │   ├── User.php                 # 
    │   │   ├── Permission.php                 # 
    │   │   ├──                 # 
    │   │   ├── Operator.php              # Modelo para operadores
    │   │   └── SyncStatus.php            # Modelo para estado de sincronización
    │   ├── utils/
    │   │   ├── password.php          # 
    │   │   ├── 
    │   │   ├── 
    │   │   └── 
    │   └── index.php                     # Punto de entrada (opcional)
    │   ├── authenticate.php -- modulo echo
    │   ├── operators.php -- modulo echo
    │   ├── events.php -- 
    │   ├── uploads.php -- 
    │   └── alerts.php -- modulo echo   
    ├── scripts/             # Páginas del dashboard
    |   ├── monitor_reports.php  
        ├── monitor_devices.php  
    ├── pages/             # Páginas del dashboard
    |   ├── alerts/             # 
    │   |    ├── index.php     
    │   |    ├── view.php 
    |   ├── dashboard/             # estadisticas
    │   |    ├── index.php     
    │   |    ├── create.php
    │   |    ├── edit.php     
    │   |    ├── delete.php 
    │   |    ├── permissions.php 
    |   ├── devices/             # dispositivos
    │   |    ├── index.php     
    │   |    ├── create.php
    │   |    ├── edit.php     
    │   |    ├── delete.php 
    │   |    ├── view.php
            ├── config.php               # ✅ PASO 4 - Lista dispositivos
│   │   │   ├── configure.php            # ✅ PASO 5 - Configuración individual
│   │   │   ├── get_config_history.php   # ✅ PASO 6.1 - Historial
│   │   │   ├── retry_config.php         # ✅ PASO 6.2 - Reintentar
│   │   │   ├── load_default_config.php  # ✅ PASO 6.3 - Config por defecto
│   │   │   ├── test_connection.php      # ✅ PASO 6.4 - Test conexión
│   │   │   ├── export_config.php        # ✅ PASO 6.5 - Exportar
│   │   │   ├── apply_profile.php        # ✅ PASO 6.6 - Aplicar perfil
│   │   │   ├── duplicate_config.php     # ✅ PASO 6.7 - Duplicar
│   │   │   ├── reset_config.php         # ✅ PASO 6.8 - Reset
│   │   │   ├── get_config_details.php   #  PASO 6.9 - Detalles config
│   │   │   └── rollback_config.php      #  PASO 6.10 - Rollback
    |   ├── users/             # 
    │   |    ├── index.php     
    │   |    ├── stats.php 
    |   ├── machines/             # Páginas del dashboard
    │   |    ├── create.php     
    │   |    ├── delete.php 
    │   |    ├── edit.php
    │   |    ├── index.php
    │   |    ├── unassign.php       
    │   |    └── view.php 
    |   ├── operators/             # Páginas del dashboard
    │   |    ├── assign.php     -- modulo echo
    │   |    ├── check_dni.php  -- modulo echo
    │   |    ├── create.php  -- modulo echo
    │   |    ├── delete.php -- modulo echo
    │   |    ├── edith.php      -- modulo echo 
    │   |    └── index.php  -- modulo echo
    |        └── view.php   -- modulo echo
    ├── index.php          # Página principal del dashboard -- modulo echo
    ├── database.sql       # Estructura inicial de la BD -- modulo echo
    ├── login.php -- modulo echo
    └──logut.php -- modulo echo
    ├── uploads/    
    |    ├── events/
│   |    ├──         
    
│
├── tests/                   # Pruebas unitarias y de integración
├── venv/
├── .env.example             # Ejemplo de variables de entorno
├── requirements.txt         # Dependencias Python para el cliente
└── README.md                # Documentación general
│
├── alarm_module.py  # modulo echo
├── behavior_detection_module.py  # modulo comportamiento 
├── bostezo_detection.py  # modulo echo
├── camera_module.py  # modulo echo
├── distraction_detection.py  # modulo distraccion
├── face_recognition_module.py  # reconocimiento_facial
├── fatigue_detection.py  # Deteccion de Factiga 
|__ behavior_detection_wrapper.py  
├── main_system.py   # 
├── process_photos.py  # 
├── register_operator.py  # 
├── report_generator.py  # 
├── fatigue_adapter.py #
├── face_recognition_wrapper.py
├── fatigue_detection_wrapper.py
├── main_system_wrapper.py
├── fatigue_adapter.py
|__ behavior_adapter.py
|__ face_recognition_adapter.py
|__ sync_integrator.py
|__ main_with_sync.py
|__ SYNC_INTEGRATION_GUIDE.md

# RASPBERRY 
nombre del Pi raspberrypi
Usuario: SafetySystem
Contraseña: Thenilfer1414
# para conectarse con ssh
ssh SafetySystem@raspberrypi.local
# para subir o el proyecto remotamente -> desde cmd o powercell
scp "C:\xampp\htdocs\safety_system\proyecto.zip" SafetySystem@raspberrypi.local:/home/SafetySystem/

# en el Pi
mkdir ~/safety_system = crear una carpeta para extrael el proyecto 
cd ~/safety_system  = entrar a la carpeta
mv ~/proyecto.zip . = mover el zip a la carpeta 
unzip proyecto.zip  = extraer el proyecto 
rm proyecto.zip = eliminar el zip 
# ✅ Para correr el sistema en entorno virtual:
source ~/safety_env/bin/activate = crear el entorno virtual
cd ~/safety_system = entrar al proyecto
pip install -r requirements.txt  = instalar todos los requerimientos.
pip install pygame = instalar la dependencia

# CONECTAR EL PI CON EL PANEL
cd ~/safety_system/client/config
nano config.ini = editar el codigo directo desde el py
Ctrl + X = para salir
Y = para confirmar guardar
Enter = para confirmar el nombre
_________________
luego :
cd ~/safety_system = regresar a la carpeta del proyecto
python3 -c "from configparser import ConfigParser; config = ConfigParser(); config.read('client/config/config.ini'); print('URL:', config.get('SERVER', 'api_url')); print('Device ID:', config.get('DEVICE', 'device_id')); print('API Key:', config.get('DEVICE', 'api_key')[:10] + '...')"
 - copiar todo en una sola linea y obtener el resultado:
 URL: http://192.168.1.100/safety_system/server/api/v1
Device ID: RPI50001
API Key: fe02131d43...

# CARGAR LAS DEPENDENCIAS DEL PI
cd ~/safety_system
pip3 install opencv-python numpy requests pygame mediapipe

- Para face-recognition en Pi, es mejor usar apt primero:
sudo apt update
sudo apt install python3-opencv python3-numpy python3-requests

- Luego intentar face-recognition:
pip3 install face-recognition


# ejecutar
python3 main_system.py

# COPIAR SOLO MAIN
scp C:\xampp\htdocs\safety_system\main_system.py SafetySystem@raspberrypi.local:/home/SafetySystem/safety_system/





# IMPORTANTE Reemplaza "tu_usuario" y "tu_contraseña" con tus credenciales reales de MySQL. -> SERVER / CONFIG / DATABASE.PHP

# Asegurar permisos correctos para el directorio de uploads
chmod -R 755 server/uploads # -R: aplica de forma recursiva (a todo lo que esté dentro).

755: permisos típicos para que:

Propietario pueda leer, escribir y ejecutar.

Otros usuarios puedan leer y ejecutar, pero no escribir.

# git bash
./setup_sync.sh
python main_system_wrapper.py ---Para usar el sistema con sincronización, simplemente ejecuta:

# cron para ejecutar el monitor
# Monitorear el estado de los dispositivos cada minuto
* * * * * cd /path/to/your/safety_system/server/scripts && php monitor_devices.php >> /path/to/your/safety_system/server/logs/device_monitor.log 2>&1

# Monitorear reportes (ya existente)
* * * * * cd /path/to/your/safety_system/server/scripts && php monitor_reports.php >> /path/to/your/safety_system/server/logs/monitor_reports.log 2>&1

* * * * * Manual mente desde Xampp 
 C:\xampp\htdocs\safety_system\server\scripts\monitor_devices.php


# API Key generada
fe02131d43d2cd8099f09ad78ab13bce4f26d2cd2e2a1e7611986775e1636905
ID Dispositivo: RPI50001
Nombre: Raspberry Piloto01
# CONECTAR EL PI CON EL PANEL
1. Registrar el dispositivo en el dashboard:

Ir a tu panel web (server/pages/devices/create.php)
Registrar el Pi 5 con un ID único (ej: "RPI_EXCAVADORA_01")
El sistema generará una API Key

2. Configurar la API Key en el código Python:

Buscar en tu código donde se configura la conexión al servidor
Añadir la API Key que generó el dashboard
Configurar la URL de tu servidor

3. Configurar la IP/URL del servidor:

El Pi necesita saber dónde está tu servidor web
Puede ser IP local o dominio

4. Probar la conexión:

Ejecutar el sistema en el Pi
Verificar que aparezca como "online" en el dashboard
Comprobar que lleguen los reportes