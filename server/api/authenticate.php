<?php
/**
 * API para autenticación de clientes (Raspberry Pi)
 */

// Encabezados para API
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Access-Control-Allow-Headers, Content-Type, Access-Control-Allow-Methods, Authorization, X-Requested-With');

// Incluir archivos necesarios
require_once '../config/database.php';
require_once '../includes/functions.php';

// Verificar método de solicitud
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode([
        'success' => false,
        'message' => 'Método no permitido'
    ]);
    exit;
}

// Obtener datos enviados
$data = json_decode(file_get_contents('php://input'), true);

// Si no hay datos JSON, intentar con POST
if (!$data) {
    $data = $_POST;
}

// Verificar datos recibidos
if (!isset($data['device_id']) || !isset($data['api_key'])) {
    echo json_encode([
        'success' => false,
        'message' => 'Faltan parámetros requeridos'
    ]);
    exit;
}

// Verificar credenciales (esto es un ejemplo, deberías tener una tabla de dispositivos)
$deviceId = $data['device_id'];
$apiKey = $data['api_key'];

// En un escenario real, verificarías contra la base de datos
// Aquí usamos valores de ejemplo
$validDeviceId = 'raspberry_01';
$validApiKey = 'your_secure_api_key_123';

if ($deviceId === $validDeviceId && $apiKey === $validApiKey) {
    // Generar token de acceso (en un sistema real, usarías JWT o similar)
    $accessToken = bin2hex(random_bytes(32));
    $expiresAt = time() + 3600; // 1 hora
    
    // En un escenario real, guardarías este token en la base de datos
    
    echo json_encode([
        'success' => true,
        'message' => 'Autenticación exitosa',
        'access_token' => $accessToken,
        'expires_at' => $expiresAt
    ]);
} else {
    echo json_encode([
        'success' => false,
        'message' => 'Credenciales inválidas'
    ]);
}