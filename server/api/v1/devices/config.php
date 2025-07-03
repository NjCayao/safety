<?php
/**
 * API Endpoint: Gestión de configuración de dispositivos
 * server/api/v1/devices/config.php
 */

// Configurar cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: GET, POST, PUT");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../../includes/device_config.php';
include_once '../../utils/authenticate.php';

// Manejar solicitudes OPTIONS (preflight)
if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    header("HTTP/1.1 200 OK");
    exit();
}

// Función para enviar respuesta JSON
function sendResponse($data, $status_code = 200) {
    http_response_code($status_code);
    echo json_encode($data);
    exit();
}

// Obtener método HTTP
$method = $_SERVER['REQUEST_METHOD'];
$request_uri = $_SERVER['REQUEST_URI'];

// Extraer device_id de la URL si está presente
$path_parts = explode('/', trim(parse_url($request_uri, PHP_URL_PATH), '/'));
$device_id = null;

// Buscar device_id en los parámetros de la URL
if (isset($_GET['device_id'])) {
    $device_id = $_GET['device_id'];
} elseif (count($path_parts) >= 6) {
    // Asumiendo estructura /api/v1/devices/config/{device_id}
    $device_id = $path_parts[5];
}

try {
    switch ($method) {
        case 'GET':
            handleGetRequest($device_id);
            break;
            
        case 'POST':
            handlePostRequest();
            break;
            
        case 'PUT':
            handlePutRequest($device_id);
            break;
            
        default:
            sendResponse(['error' => 'Método no permitido'], 405);
    }
} catch (Exception $e) {
    sendResponse(['error' => 'Error interno del servidor: ' . $e->getMessage()], 500);
}

/**
 * Maneja solicitudes GET - Raspberry Pi consulta configuración
 */
function handleGetRequest($device_id) {
    if (!$device_id) {
        sendResponse(['error' => 'device_id requerido'], 400);
    }

    // Autenticar dispositivo
    $device_data = authenticate();
    if (!$device_data || $device_data['device_id'] !== $device_id) {
        sendResponse(['error' => 'No autorizado'], 401);
    }

    // Actualizar último check de configuración
    db_update(
        'devices', 
        ['last_config_check' => date('Y-m-d H:i:s')], 
        'device_id = ?', 
        [$device_id]
    );

    // Obtener configuración actual
    $config_data = DeviceConfigManager::getDeviceConfig($device_id);
    
    if (!$config_data) {
        sendResponse(['error' => 'Dispositivo no encontrado'], 404);
    }

    // Respuesta para la Raspberry Pi
    $response = [
        'device_id' => $device_id,
        'config_version' => $config_data['version'],
        'config_pending' => $config_data['pending'],
        'last_applied' => $config_data['last_applied'],
        'config' => $config_data['config'],
        'timestamp' => date('Y-m-d H:i:s')
    ];

    sendResponse($response);
}

/**
 * Maneja solicitudes POST - Confirmaciones y reportes de error
 */
function handlePostRequest() {
    $input = json_decode(file_get_contents('php://input'), true);
    
    if (!$input || !isset($input['action'])) {
        sendResponse(['error' => 'Acción requerida'], 400);
    }

    $action = $input['action'];
    $device_id = $input['device_id'] ?? null;

    if (!$device_id) {
        sendResponse(['error' => 'device_id requerido'], 400);
    }

    // Autenticar dispositivo
    $device_data = authenticate();
    if (!$device_data || $device_data['device_id'] !== $device_id) {
        sendResponse(['error' => 'No autorizado'], 401);
    }

    switch ($action) {
        case 'config_applied':
            handleConfigApplied($input);
            break;
            
        case 'config_error':
            handleConfigError($input);
            break;
            
        case 'heartbeat':
            handleHeartbeat($input);
            break;
            
        default:
            sendResponse(['error' => 'Acción no válida'], 400);
    }
}

/**
 * Maneja confirmación de configuración aplicada
 */
function handleConfigApplied($input) {
    $device_id = $input['device_id'];
    $config_version = $input['config_version'] ?? null;
    $history_id = $input['history_id'] ?? null;

    if (!$config_version) {
        sendResponse(['error' => 'config_version requerida'], 400);
    }

    $result = DeviceConfigManager::confirmConfigApplied($device_id, $config_version, $history_id);
    
    if ($result) {
        // Registrar en log
        log_system_message(
            'info',
            "Configuración aplicada exitosamente en dispositivo {$device_id}",
            null,
            "Versión: {$config_version}"
        );

        sendResponse([
            'success' => true,
            'message' => 'Configuración confirmada como aplicada',
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    } else {
        sendResponse(['error' => 'Error al confirmar aplicación de configuración'], 500);
    }
}

/**
 * Maneja reporte de error en configuración
 */
function handleConfigError($input) {
    $device_id = $input['device_id'];
    $error_message = $input['error_message'] ?? 'Error desconocido';
    $history_id = $input['history_id'] ?? null;
    $config_version = $input['config_version'] ?? null;

    $result = DeviceConfigManager::reportConfigError($device_id, $error_message, $history_id);
    
    if ($result) {
        sendResponse([
            'success' => true,
            'message' => 'Error reportado correctamente',
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    } else {
        sendResponse(['error' => 'Error al reportar problema de configuración'], 500);
    }
}

/**
 * Maneja heartbeat del dispositivo
 */
function handleHeartbeat($input) {
    $device_id = $input['device_id'];
    $status = $input['status'] ?? 'online';
    $system_info = $input['system_info'] ?? null;

    // Actualizar estado del dispositivo
    $update_data = [
        'last_access' => date('Y-m-d H:i:s'),
        'status' => $status,
        'last_config_check' => date('Y-m-d H:i:s')
    ];

    // Si hay información del sistema, agregarla a los detalles
    if ($system_info) {
        // Aquí podrías agregar una columna system_info si la necesitas
    }

    $result = db_update('devices', $update_data, 'device_id = ?', [$device_id]);

    if ($result !== false) {
        // Verificar si hay configuración pendiente
        $config_data = DeviceConfigManager::getDeviceConfig($device_id);
        
        sendResponse([
            'success' => true,
            'message' => 'Heartbeat recibido',
            'config_pending' => $config_data['pending'],
            'config_version' => $config_data['version'],
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    } else {
        sendResponse(['error' => 'Error al actualizar heartbeat'], 500);
    }
}

/**
 * Maneja solicitudes PUT - Actualización manual de configuración (desde dashboard)
 */
function handlePutRequest($device_id) {
    if (!$device_id) {
        sendResponse(['error' => 'device_id requerido'], 400);
    }

    // Para actualizaciones desde dashboard, necesitamos autenticación de usuario web
    // Aquí deberías implementar autenticación de sesión web
    session_start();
    if (!isset($_SESSION['user_id'])) {
        sendResponse(['error' => 'Sesión requerida'], 401);
    }

    // Verificar permisos (admin o supervisor)
    if (!isset($_SESSION['user_role']) || !in_array($_SESSION['user_role'], ['admin', 'supervisor'])) {
        sendResponse(['error' => 'Permisos insuficientes'], 403);
    }

    $input = json_decode(file_get_contents('php://input'), true);
    
    if (!$input || !isset($input['config'])) {
        sendResponse(['error' => 'Configuración requerida'], 400);
    }

    $new_config = $input['config'];
    $change_summary = $input['change_summary'] ?? null;
    $user_id = $_SESSION['user_id'];

    $result = DeviceConfigManager::updateDeviceConfig($device_id, $new_config, $user_id, $change_summary);

    if ($result['success']) {
        sendResponse([
            'success' => true,
            'message' => $result['message'],
            'history_id' => $result['history_id'],
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    } else {
        sendResponse(['error' => $result['error']], 400);
    }
}

// Función de autenticación específica para este endpoint
function authenticate() {
    // Verificar token JWT del header Authorization
    $headers = getallheaders();
    $authHeader = $headers['Authorization'] ?? '';
    
    if (!$authHeader || !preg_match('/Bearer\s(\S+)/', $authHeader, $matches)) {
        return false;
    }

    $jwt = $matches[1];
    
    // Aquí deberías usar tu implementación de JWT
    // Por simplicidad, asumimos que tienes una función de verificación
    include_once '../../utils/JwtHandler.php';
    
    $jwtHandler = new JwtHandler();
    $payload = $jwtHandler->decode($jwt);
    
    if (!$payload) {
        return false;
    }
    
    return $payload['data'] ?? false;
}
?>