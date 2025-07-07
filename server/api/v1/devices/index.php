<?php
/**
 * API v1/devices - Endpoint principal para dispositivos
 * server/api/v1/devices/index.php
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
require_once '../../../config/config.php';
require_once '../../../config/database.php';

// Función para enviar respuesta JSON
function sendResponse($data, $status_code = 200) {
    http_response_code($status_code);
    echo json_encode($data);
    exit();
}

// Información del endpoint de dispositivos
$devices_info = [
    'status' => 'success',
    'message' => 'Safety System API v1 - Devices Endpoint',
    'version' => '1.0.0',
    'timestamp' => date('Y-m-d H:i:s'),
    'available_endpoints' => [
        'POST /api/v1/devices/register' => 'Registrar nuevo dispositivo',
        'POST /api/v1/devices/heartbeat' => 'Enviar heartbeat de estado',
        'GET /api/v1/devices/status' => 'Obtener estado del dispositivo',
        'GET /api/v1/devices/config' => 'Obtener configuración del dispositivo',
        'PUT /api/v1/devices/config' => 'Actualizar configuración del dispositivo'
    ],
    'authentication' => 'Requerido: Bearer token en header Authorization',
    'example_urls' => [
        'https://safetysystem.devcayao.com/server/api/v1/devices/register',
        'https://safetysystem.devcayao.com/server/api/v1/devices/heartbeat',
        'https://safetysystem.devcayao.com/server/api/v1/devices/status',
        'https://safetysystem.devcayao.com/server/api/v1/devices/config'
    ]
];

sendResponse($devices_info);
?>