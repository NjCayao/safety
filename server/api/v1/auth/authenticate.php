<?php
// server/api/v1/auth/authenticate.php

// Permitir solicitudes desde cualquier origen (CORS)
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/JwtHandler.php';
include_once '../../utils/Response.php';
include_once '../../models/Device.php';

// Solo permitir método POST
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo Response::error('Método no permitido', 405);
    exit();
}

// Obtener conexión a la base de datos
$database = new Database();
$db = $database->getConnection();

// Preparar objeto Device
$device = new Device($db);

// Obtener datos enviados
$data = json_decode(file_get_contents("php://input"));

// Verificar datos requeridos
if (!isset($data->device_id) || !isset($data->api_key)) {
    echo Response::error('Datos incompletos', 400);
    exit();
}

// Verificar credenciales del dispositivo
if ($device->verifyCredentials($data->device_id, $data->api_key)) {
    // Credenciales válidas, generar token
    $jwt = new JwtHandler();
    
    $token = $jwt->generateToken([
        'device_id' => $data->device_id,
        'device_type' => $device->device_type,
        'machine_id' => $device->machine_id
    ]);
    
    // Actualizar último acceso
    $device->updateLastAccess();
    
    echo Response::success([
        'token' => $token,
        'expires_in' => 3600 * 12, // 12 horas en segundos
        'device_id' => $data->device_id,
        'machine_id' => $device->machine_id
    ], 'Autenticación exitosa');
} else {
    // Credenciales inválidas
    echo Response::error('Credenciales inválidas', 401);
}
?>