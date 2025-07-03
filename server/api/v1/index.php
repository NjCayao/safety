<?php
/**
 * API v1 - Endpoint principal
 * server/api/v1/index.php
 */

// Configurar cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Manejar solicitudes OPTIONS (preflight)
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    header("HTTP/1.1 200 OK");
    exit();
}

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';

// Función para enviar respuesta JSON
function sendResponse($data, $status_code = 200) {
    http_response_code($status_code);
    echo json_encode($data);
    exit();
}

// Verificar conexión a la base de datos
$database = new Database();
$pdo = $database->getConnection();

if ($pdo === null) {
    sendResponse([
        'status' => 'error',
        'message' => 'Error de conexión a la base de datos',
        'timestamp' => date('Y-m-d H:i:s')
    ], 500);
}

// Información de la API
$api_info = [
    'status' => 'success',
    'message' => 'Safety System API v1',
    'version' => '1.0.0',
    'timestamp' => date('Y-m-d H:i:s'),
    'endpoints' => [
        'authentication' => [
            'POST /api/v1/auth/authenticate' => 'Autenticación de dispositivos',
            'POST /api/v1/auth/verify' => 'Verificar token de acceso'
        ],
        'devices' => [
            'POST /api/v1/devices/register' => 'Registrar nuevo dispositivo',
            'POST /api/v1/devices/heartbeat' => 'Enviar heartbeat',
            'GET /api/v1/devices/status' => 'Obtener estado del dispositivo',
            'GET /api/v1/devices/config' => 'Obtener configuración'
        ],
        'sync' => [
            'POST /api/v1/sync/batch' => 'Sincronizar lote de eventos',
            'POST /api/v1/sync/confirm' => 'Confirmar sincronización',
            'GET /api/v1/sync/status' => 'Estado de sincronización'
        ],
        'operators' => [
            'GET /api/v1/operators/sync' => 'Sincronizar operadores'
        ]
    ]
];

sendResponse($api_info);
?>