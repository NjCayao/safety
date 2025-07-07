<?php
// api/v1/devices/register.php

// Permitir solicitudes desde cualquier origen (CORS)
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/Response.php';

// Solo permitir método POST
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo Response::error('Método no permitido', 405);
    exit();
}

// Obtener datos enviados
$data = json_decode(file_get_contents("php://input"));

// Validar datos requeridos
if (!isset($data->device_id) || !isset($data->device_type) || !isset($data->registration_key)) {
    echo Response::error('Datos incompletos', 400);
    exit();
}

// Verificar clave de registro (podría ser una clave maestra o temporal)
$MASTER_REGISTRATION_KEY = 'your-secure-registration-key-here'; // Cambiar esto en producción

if ($data->registration_key !== $MASTER_REGISTRATION_KEY) {
    echo Response::error('Clave de registro inválida', 401);
    exit();
}

// Verificar si el dispositivo ya existe
$existing = db_fetch_one(
    "SELECT id FROM devices WHERE device_id = ?", 
    [$data->device_id]
);

if ($existing) {
    echo Response::error('El dispositivo ya está registrado', 409);
    exit();
}

// Generar API key para el dispositivo
$api_key = bin2hex(random_bytes(32));
$api_key_hash = password_hash($api_key, PASSWORD_DEFAULT);

// Obtener IP del cliente
$client_ip = $_SERVER['REMOTE_ADDR'];
if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
    $client_ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
}

// Registrar dispositivo
$deviceData = [
    'device_id' => $data->device_id,
    'api_key' => $api_key_hash,
    'name' => $data->name ?? null,
    'device_type' => $data->device_type,
    'machine_id' => $data->machine_id ?? null,
    'location' => $data->location ?? null,
    'status' => 'online',
    'ip_address' => $client_ip,
    'last_access' => date('Y-m-d H:i:s'),
    'created_at' => date('Y-m-d H:i:s')
];

$device_id = db_insert('devices', $deviceData);

if ($device_id) {
    // Registrar en los logs
    db_insert('system_logs', [
        'log_type' => 'info',
        'message' => "Nuevo dispositivo registrado: {$data->device_id}",
        'details' => json_encode([
            'device_type' => $data->device_type,
            'ip' => $client_ip,
            'machine_id' => $data->machine_id ?? null
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);
    
    echo Response::success([
        'device_id' => $data->device_id,
        'api_key' => $api_key, // Solo se muestra una vez
        'status' => 'registered',
        'message' => 'Dispositivo registrado exitosamente. Guarde su API key, no se mostrará nuevamente.'
    ], 'Registro exitoso');
} else {
    echo Response::error('Error al registrar el dispositivo', 500);
}
?>