<?php
/**
 * Script auxiliar: Aplicar perfil de configuración
 * server/pages/devices/apply_profile.php
 */

header('Content-Type: application/json');

// Verificar sesión
session_start();
if (!isset($_SESSION['user_id'])) {
    http_response_code(401);
    echo json_encode(['success' => false, 'message' => 'No autorizado']);
    exit;
}

// Verificar permisos
if (!isset($_SESSION['user_role']) || !in_array($_SESSION['user_role'], ['admin', 'supervisor'])) {
    http_response_code(403);
    echo json_encode(['success' => false, 'message' => 'Sin permisos']);
    exit;
}

// Incluir dependencias
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/device_config.php';

// Verificar método POST
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Método no permitido']);
    exit;
}

// Obtener datos
$device_id = $_POST['device_id'] ?? null;
$profile_id = $_POST['profile_id'] ?? null;

if (!$device_id || !$profile_id) {
    echo json_encode(['success' => false, 'message' => 'ID de dispositivo y perfil requeridos']);
    exit;
}

try {
    // Verificar que el dispositivo existe
    $device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$device_id]);
    if (!$device) {
        echo json_encode(['success' => false, 'message' => 'Dispositivo no encontrado']);
        exit;
    }

    // Verificar que el perfil existe
    $profile = db_fetch_one("SELECT * FROM device_config_profiles WHERE id = ?", [$profile_id]);
    if (!$profile) {
        echo json_encode(['success' => false, 'message' => 'Perfil no encontrado']);
        exit;
    }

    // Verificar compatibilidad de tipo de dispositivo
    if ($profile['device_type'] && $profile['device_type'] !== $device['device_type']) {
        echo json_encode([
            'success' => false, 
            'message' => "El perfil '{$profile['name']}' no es compatible con dispositivos de tipo '{$device['device_type']}'"
        ]);
        exit;
    }

    // Obtener configuración del perfil
    $profile_config = json_decode($profile['config_json'], true);
    if (!$profile_config) {
        throw new Exception('Configuración del perfil inválida');
    }

    // Aplicar el perfil usando el manager
    $result = DeviceConfigManager::applyConfigProfile($device_id, $profile_id, $_SESSION['user_id']);

    if (!$result['success']) {
        throw new Exception($result['error']);
    }

    // Registrar aplicación de perfil en logs
    db_insert('system_logs', [
        'log_type' => 'info',
        'machine_id' => $device['machine_id'],
        'message' => "Perfil '{$profile['name']}' aplicado al dispositivo {$device_id}",
        'details' => json_encode([
            'profile_id' => $profile_id,
            'profile_name' => $profile['name'],
            'user' => $_SESSION['username'] ?? 'N/A',
            'config_version' => $result['config_version'] ?? null
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);

    echo json_encode([
        'success' => true,
        'message' => "Perfil '{$profile['name']}' aplicado correctamente",
        'profile_name' => $profile['name'],
        'profile_description' => $profile['description'],
        'config_version' => $result['config_version'] ?? null,
        'history_id' => $result['history_id'] ?? null
    ]);

} catch (Exception $e) {
    error_log("Error en apply_profile.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al aplicar perfil: ' . $e->getMessage()
    ]);
}
?>