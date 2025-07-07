<?php
// api/v1/devices/heartbeat.php

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

// Verificar token (JWT o API Key)
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

$device_id = null;
$machine_id = null;

// ✅ NUEVO: Intentar primero como API Key
try {
    $database = new Database();
    $db = $database->getConnection();
    
    // Buscar dispositivo por API Key
    $stmt = $db->prepare("SELECT device_id, machine_id, api_key FROM devices WHERE api_key = ?");
    $stmt->execute([password_hash($token, PASSWORD_DEFAULT)]);
    
    if ($stmt->rowCount() == 0) {
        // Probar verificando hash
        $stmt = $db->prepare("SELECT device_id, machine_id, api_key FROM devices");
        $stmt->execute();
        $devices = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        foreach ($devices as $device) {
            if (password_verify($token, $device['api_key'])) {
                $device_id = $device['device_id'];
                $machine_id = $device['machine_id'];
                break;
            }
        }
    } else {
        $device_data = $stmt->fetch(PDO::FETCH_ASSOC);
        $device_id = $device_data['device_id'];
        $machine_id = $device_data['machine_id'];
    }
    
} catch (Exception $e) {
    // Si falla la verificación de API Key, intentar JWT
}

// ✅ Si no se encontró por API Key, intentar JWT
if (!$device_id) {
    $jwt = new JwtHandler();
    $decoded = $jwt->verifyToken($token);
    
    if (!$decoded) {
        echo Response::error('Token inválido o expirado', 401);
        exit();
    }
    
    $device_id = $decoded->device_id;
    $machine_id = $decoded->machine_id ?? null;
}

// Obtener datos enviados
$data = json_decode(file_get_contents("php://input"));

// Obtener IP del cliente
$client_ip = $_SERVER['REMOTE_ADDR'];
if (!empty($_SERVER['HTTP_X_FORWARDED_FOR'])) {
    $client_ip = $_SERVER['HTTP_X_FORWARDED_FOR'];
}

// ✅ NUEVO: Actualizar dispositivo por device_id
try {
    $database = new Database();
    $db = $database->getConnection();
    
    $result = $db->prepare(
        "UPDATE devices 
         SET last_access = NOW(), 
             status = ?, 
             ip_address = ?
         WHERE device_id = ?"
    );
    
    $success = $result->execute([
        $data->status ?? 'online',
        $client_ip,
        $device_id
    ]);

    if ($success && $result->rowCount() > 0) {
        // Registrar heartbeat en los logs
        try {
            $log_stmt = $db->prepare(
                "INSERT INTO system_logs (log_type, machine_id, message, details, timestamp) 
                 VALUES (?, ?, ?, ?, ?)"
            );
            $log_stmt->execute([
                'info',
                $machine_id,
                "Heartbeat recibido del dispositivo {$device_id}",
                json_encode([
                    'ip' => $client_ip,
                    'status' => $data->status ?? 'online',
                    'additional_info' => $data->info ?? null
                ]),
                date('Y-m-d H:i:s')
            ]);
        } catch (Exception $e) {
            // Log error but don't fail the heartbeat
        }
        
        echo Response::success([
            'device_id' => $device_id,
            'timestamp' => date('Y-m-d H:i:s'),
            'status' => 'acknowledged'
        ], 'Heartbeat recibido');
        
    } else {
        echo Response::error('Dispositivo no encontrado o no se pudo actualizar', 404);
    }
    
} catch (Exception $e) {
    echo Response::error('Error al actualizar estado del dispositivo: ' . $e->getMessage(), 500);
}
?>