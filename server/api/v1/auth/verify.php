<?php
// server/api/v1/auth/verify.php

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

// Obtener token del header de autorización
$headers = apache_request_headers();
$authHeader = isset($headers['Authorization']) ? $headers['Authorization'] : '';

if (empty($authHeader)) {
    echo Response::error('Token no proporcionado', 401);
    exit();
}

// Extraer token del header (Bearer token)
$token = '';
if (preg_match('/Bearer\s(\S+)/', $authHeader, $matches)) {
    $token = $matches[1];
}

if (empty($token)) {
    echo Response::error('Token inválido', 401);
    exit();
}

// Verificar token
$jwt = new JwtHandler();
$decoded = $jwt->verifyToken($token);

if ($decoded) {
    // Token válido
    echo Response::success([
        'valid' => true,
        'device_id' => $decoded->device_id,
        'device_type' => $decoded->device_type,
        'machine_id' => $decoded->machine_id,
        'expires_at' => date('Y-m-d H:i:s', $decoded->exp)
    ], 'Token válido');
} else {
    // Token inválido o expirado
    echo Response::error('Token inválido o expirado', 401);
}
?>