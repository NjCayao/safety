<?php
/**
 * Script auxiliar: Resetear configuración a valores por defecto
 * server/pages/devices/reset_config.php
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

    // Obtener configuración actual para backup
    $current_config = null;
    if ($device['config_json']) {
        $current_config = json_decode($device['config_json'], true);
    }

    // Buscar perfil por defecto para este tipo de dispositivo
    $default_profile = db_fetch_one(
        "SELECT * FROM device_config_profiles WHERE device_type = ? AND is_default = 1 LIMIT 1",
        [$device['device_type']]
    );

    if ($default_profile) {
        // Usar configuración del perfil por defecto
        $default_config = json_decode($default_profile['config_json'], true);
        $config_source = "perfil por defecto '{$default_profile['name']}'";
    } else {
        // Usar configuración hardcodeada
        $default_config = getSystemDefaultConfig();
        $config_source = "configuración del sistema";
    }

    if (!$default_config) {
        throw new Exception('No se pudo obtener configuración por defecto');
    }

    // Crear resumen de cambios
    $change_summary = "Configuración reseteada a valores por defecto ({$config_source}) por " . ($_SESSION['username'] ?? 'usuario');

    // Aplicar configuración por defecto
    $result = DeviceConfigManager::updateDeviceConfig(
        $device_id,
        $default_config,
        $_SESSION['user_id'],
        $change_summary
    );

    if (!$result['success']) {
        throw new Exception($result['error']);
    }

    // Crear backup de la configuración anterior si existía
    if ($current_config) {
        createConfigBackup($device_id, $current_config, $_SESSION['user_id']);
    }

    // Registrar en logs del sistema
    db_insert('system_logs', [
        'log_type' => 'info',
        'machine_id' => $device['machine_id'],
        'message' => "Configuración de {$device_id} reseteada a valores por defecto",
        'details' => json_encode([
            'config_source' => $config_source,
            'user' => $_SESSION['username'] ?? 'N/A',
            'had_previous_config' => !empty($current_config),
            'profile_used' => $default_profile['name'] ?? null
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);

    echo json_encode([
        'success' => true,
        'message' => 'Configuración reseteada a valores por defecto correctamente',
        'config_source' => $config_source,
        'config_version' => $result['config_version'] ?? null,
        'history_id' => $result['history_id'] ?? null,
        'backup_created' => !empty($current_config)
    ]);

} catch (Exception $e) {
    error_log("Error en reset_config.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al resetear configuración: ' . $e->getMessage()
    ]);
}

/**
 * Crear backup de configuración anterior
 */
function createConfigBackup($device_id, $config, $user_id) {
    try {
        $backup_data = [
            'device_id' => $device_id,
            'changed_by' => $user_id,
            'change_type' => 'rollback',
            'config_before' => null,
            'config_after' => json_encode($config),
            'changes_summary' => 'Backup antes de reset a configuración por defecto',
            'applied_successfully' => true,
            'applied_at' => date('Y-m-d H:i:s'),
            'created_at' => date('Y-m-d H:i:s')
        ];

        db_insert('device_config_history', $backup_data);
        return true;
    } catch (Exception $e) {
        error_log("Error creando backup de configuración: " . $e->getMessage());
        return false;
    }
}

/**
 * Configuración por defecto del sistema
 */
function getSystemDefaultConfig() {
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