<?php
// api/v1/devices/status.php

// Permitir solicitudes desde cualquier origen (CORS)
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: GET, POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/JwtHandler.php';
include_once '../../utils/Response.php';

// Verificar método permitido
if (!in_array($_SERVER['REQUEST_METHOD'], ['GET', 'POST'])) {
    echo Response::error('Método no permitido', 405);
    exit();
}

// Verificar token JWT
$headers = apache_request_headers();
$authHeader = isset($headers['Authorization']) ? $headers['Authorization'] : '';

if (empty($authHeader)) {
    echo Response::error('Token no proporcionado', 401);
    exit();
}

// Extraer token del header
$token = '';
if (preg_match('/Bearer\s(\S+)/', $authHeader, $matches)) {
    $token = $matches[1];
}

// Verificar token
$jwt = new JwtHandler();
$decoded = $jwt->verifyToken($token);

if (!$decoded) {
    echo Response::error('Token inválido o expirado', 401);
    exit();
}

// Obtener información del dispositivo
$device = db_fetch_one(
    "SELECT d.*, m.name as machine_name, m.status as machine_status
     FROM devices d
     LEFT JOIN machines m ON d.machine_id = m.id
     WHERE d.device_id = ?",
    [$decoded->device_id]
);

if (!$device) {
    echo Response::error('Dispositivo no encontrado', 404);
    exit();
}

// Calcular tiempo sin conexión
$last_access = $device['last_access'] ? new DateTime($device['last_access']) : null;
$now = new DateTime();
$offline_minutes = $last_access ? $now->diff($last_access)->i + ($now->diff($last_access)->h * 60) + ($now->diff($last_access)->days * 1440) : null;

// Si ha pasado más de 5 minutos sin heartbeat, marcar como offline
if ($offline_minutes && $offline_minutes > 5 && $device['status'] !== 'offline') {
    db_query(
        "UPDATE devices SET status = 'offline' WHERE device_id = ?",
        [$device['device_id']]
    );
    $device['status'] = 'offline';
}

// Obtener estadísticas del dispositivo
$stats = [
    'alerts_today' => db_fetch_one(
        "SELECT COUNT(*) as count FROM alerts 
         WHERE device_id = ? AND DATE(timestamp) = CURDATE()",
        [$device['device_id']]
    )['count'],
    
    'events_pending' => db_fetch_one(
        "SELECT COUNT(*) as count FROM events 
         WHERE device_id = ? AND is_synced = 0",
        [$device['device_id']]
    )['count'],
    
    'last_alert' => db_fetch_one(
        "SELECT timestamp FROM alerts 
         WHERE device_id = ? 
         ORDER BY timestamp DESC LIMIT 1",
        [$device['device_id']]
    )['timestamp'] ?? null
];

// Preparar respuesta
$response = [
    'device_info' => [
        'device_id' => $device['device_id'],
        'name' => $device['name'],
        'type' => $device['device_type'],
        'status' => $device['status'],
        'last_access' => $device['last_access'],
        'ip_address' => $device['ip_address'],
        'location' => $device['location']
    ],
    'machine_info' => [
        'machine_id' => $device['machine_id'],
        'machine_name' => $device['machine_name'],
        'machine_status' => $device['machine_status']
    ],
    'statistics' => $stats,
    'connection_info' => [
        'offline_minutes' => $offline_minutes,
        'is_online' => $device['status'] === 'online'
    ]
];

echo Response::success($response, 'Estado del dispositivo');
?>