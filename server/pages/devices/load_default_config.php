<?php
/**
 * Script auxiliar: Cargar configuración por defecto
 * server/pages/devices/load_default_config.php
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

if (!$device_id) {
    echo json_encode(['success' => false, 'message' => 'ID de dispositivo requerido']);
    exit;
}

try {
    // Verificar que el dispositivo existe
    $device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$device_id]);
    if (!$device) {
        echo json_encode(['success' => false, 'message' => 'Dispositivo no encontrado']);
        exit;
    }

    // Obtener configuración por defecto según el tipo de dispositivo
    $default_profile = db_fetch_one(
        "SELECT * FROM device_config_profiles WHERE device_type = ? AND is_default = 1 LIMIT 1",
        [$device['device_type']]
    );

    if (!$default_profile) {
        // Si no hay perfil por defecto específico, usar configuración hardcodeada
        $default_config = getHardcodedDefaultConfig();
    } else {
        $default_config = json_decode($default_profile['config_json'], true);
    }

    if (!$default_config) {
        throw new Exception('No se pudo obtener configuración por defecto');
    }

    // Obtener configuración actual para backup
    $current_config = $device['config_json'] ? json_decode($device['config_json'], true) : null;

    // Aplicar configuración por defecto usando el manager
    $result = DeviceConfigManager::updateDeviceConfig(
        $device_id,
        $default_config,
        $_SESSION['user_id'],
        'Configuración restaurada por defecto'
    );

    if (!$result['success']) {
        throw new Exception($result['error']);
    }

    echo json_encode([
        'success' => true,
        'message' => 'Configuración por defecto cargada correctamente',
        'config' => $default_config,
        'history_id' => $result['history_id']
    ]);

} catch (Exception $e) {
    error_log("Error en load_default_config.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al cargar configuración por defecto: ' . $e->getMessage()
    ]);
}

/**
 * Configuración por defecto hardcodeada como fallback
 */
function getHardcodedDefaultConfig() {
    return [
        'camera' => [
            'fps' => 15,
            'width' => 640,
            'height' => 480,
            'brightness' => 0,
            'contrast' => 0,
            'saturation' => 0,
            'buffer_size' => 1,
            'use_threading' => true,
            'warmup_time' => 2
        ],
        'fatigue' => [
            'eye_closed_threshold' => 1.5,
            'ear_threshold' => 0.25,
            'ear_night_adjustment' => 0.03,
            'frames_to_confirm' => 2,
            'calibration_period' => 30,
            'alarm_cooldown' => 5,
            'multiple_fatigue_threshold' => 3,
            'night_mode_threshold' => 50,
            'enable_night_mode' => true
        ],
        'yawn' => [
            'mouth_threshold' => 0.7,
            'duration_threshold' => 2.5,
            'frames_to_confirm' => 3,
            'alert_cooldown' => 5.0,
            'max_yawns_before_alert' => 3,
            'report_delay' => 2.0,
            'enable_auto_calibration' => true,
            'calibration_frames' => 60,
            'calibration_factor' => 0.4,
            'enable_sounds' => true
        ],
        'distraction' => [
            'rotation_threshold_day' => 2.6,
            'rotation_threshold_night' => 2.8,
            'level1_time' => 3,
            'level2_time' => 5,
            'confidence_threshold' => 0.7,
            'audio_enabled' => true,
            'level1_volume' => 0.8,
            'level2_volume' => 1.0
        ],
        'behavior' => [
            'confidence_threshold' => 0.4,
            'night_confidence_threshold' => 0.35,
            'phone_alert_threshold_1' => 3,
            'phone_alert_threshold_2' => 7,
            'cigarette_continuous_threshold' => 7,
            'audio_enabled' => true
        ],
        'audio' => [
            'enabled' => true,
            'volume' => 1.0,
            'frequency' => 44100,
            'channels' => 2,
            'buffer' => 2048
        ],
        'system' => [
            'enable_gui' => false,
            'log_level' => 'INFO',
            'debug_mode' => false,
            'performance_monitoring' => true,
            'auto_optimization' => true
        ],
        'sync' => [
            'enabled' => true,
            'auto_sync_interval' => 300,
            'batch_size' => 50,
            'connection_timeout' => 10,
            'max_retries' => 3
        ]
    ];
}
?>