<?php
/**
 * Script auxiliar: Realizar rollback a configuración anterior
 * server/pages/devices/rollback_config.php
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
    echo json_encode(['success' => false, 'message' => 'Sin permisos para realizar rollback']);
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
$history_id = $_POST['history_id'] ?? null;
$device_id = $_POST['device_id'] ?? null;

if (!$history_id) {
    echo json_encode(['success' => false, 'message' => 'ID de historial requerido']);
    exit;
}

try {
    // Obtener registro de historial específico
    $history_record = db_fetch_one("
        SELECT dch.*, d.device_id, d.name as device_name, d.device_type
        FROM device_config_history dch
        LEFT JOIN devices d ON dch.device_id = d.device_id
        WHERE dch.id = ?
    ", [$history_id]);

    if (!$history_record) {
        echo json_encode(['success' => false, 'message' => 'Registro de historial no encontrado']);
        exit;
    }

    // Verificar device_id si se proporcionó
    if ($device_id && $device_id !== $history_record['device_id']) {
        echo json_encode(['success' => false, 'message' => 'El historial no corresponde al dispositivo especificado']);
        exit;
    }

    // Verificar que el registro se aplicó exitosamente
    if (!$history_record['applied_successfully']) {
        echo json_encode([
            'success' => false, 
            'message' => 'No se puede hacer rollback a una configuración que no se aplicó exitosamente'
        ]);
        exit;
    }

    // Obtener configuración para rollback
    $rollback_config = json_decode($history_record['config_after'], true);
    if (!$rollback_config) {
        echo json_encode(['success' => false, 'message' => 'Configuración de rollback inválida']);
        exit;
    }

    // Obtener configuración actual para backup
    $current_config_data = DeviceConfigManager::getDeviceConfig($history_record['device_id']);
    $current_config = $current_config_data['config'];

    // Crear resumen de cambios para rollback
    $rollback_summary = generateRollbackSummary($history_record);

    // Aplicar configuración de rollback
    $result = DeviceConfigManager::updateDeviceConfig(
        $history_record['device_id'],
        $rollback_config,
        $_SESSION['user_id'],
        $rollback_summary
    );

    if (!$result['success']) {
        throw new Exception($result['error']);
    }

    // Registrar el rollback en el historial
    $rollback_history_data = [
        'device_id' => $history_record['device_id'],
        'changed_by' => $_SESSION['user_id'],
        'change_type' => 'rollback',
        'config_before' => json_encode($current_config),
        'config_after' => $history_record['config_after'],
        'changes_summary' => $rollback_summary,
        'applied_successfully' => null,
        'rollback_from_history_id' => $history_id,
        'created_at' => date('Y-m-d H:i:s')
    ];

    $rollback_history_id = db_insert('device_config_history', $rollback_history_data);

    // Registrar en logs del sistema
    db_insert('system_logs', [
        'log_type' => 'info',
        'machine_id' => getDeviceMachineId($history_record['device_id']),
        'message' => "Rollback realizado en dispositivo {$history_record['device_id']}",
        'details' => json_encode([
            'rollback_to_date' => $history_record['created_at'],
            'rollback_by_user' => $_SESSION['username'] ?? 'N/A',
            'original_history_id' => $history_id,
            'new_history_id' => $rollback_history_id,
            'change_type' => $history_record['change_type']
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);

    // Enviar notificación si el dispositivo está online
    sendRollbackNotification($history_record['device_id'], $rollback_summary);

    echo json_encode([
        'success' => true,
        'message' => 'Rollback realizado exitosamente',
        'rollback_details' => [
            'device_id' => $history_record['device_id'],
            'device_name' => $history_record['device_name'],
            'rollback_to_date' => date('d/m/Y H:i', strtotime($history_record['created_at'])),
            'rollback_summary' => $rollback_summary,
            'new_history_id' => $rollback_history_id,
            'original_change_type' => $history_record['change_type']
        ]
    ]);

} catch (Exception $e) {
    error_log("Error en rollback_config.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error realizando rollback: ' . $e->getMessage()
    ]);
}

/**
 * Genera resumen descriptivo del rollback
 */
function generateRollbackSummary($history_record) {
    $date = date('d/m/Y H:i', strtotime($history_record['created_at']));
    $change_type = $history_record['change_type'];
    $user = $_SESSION['username'] ?? 'usuario';
    
    $change_type_names = [
        'manual' => 'cambio manual',
        'profile' => 'aplicación de perfil',
        'reset' => 'reset a configuración por defecto',
        'rollback' => 'rollback previo'
    ];
    
    $change_name = $change_type_names[$change_type] ?? $change_type;
    
    $summary = "Rollback por {$user} a configuración del {$date} ({$change_name})";
    
    if ($history_record['changes_summary']) {
        $summary .= " - Original: " . $history_record['changes_summary'];
    }
    
    return $summary;
}

/**
 * Obtiene machine_id de un dispositivo
 */
function getDeviceMachineId($device_id) {
    $device = db_fetch_one("SELECT machine_id FROM devices WHERE device_id = ?", [$device_id]);
    return $device['machine_id'] ?? null;
}

/**
 * Envía notificación de rollback al dispositivo (si está online)
 */
function sendRollbackNotification($device_id, $summary) {
    try {
        // Verificar si el dispositivo está online
        $device = db_fetch_one("
            SELECT status, last_access 
            FROM devices 
            WHERE device_id = ?
        ", [$device_id]);

        if (!$device || $device['status'] !== 'online') {
            return; // Dispositivo offline, se sincronizará automáticamente
        }

        // Verificar si la última conexión fue reciente (menos de 5 minutos)
        if ($device['last_access']) {
            $last_access_time = strtotime($device['last_access']);
            $now = time();
            $minutes_since_last_access = ($now - $last_access_time) / 60;

            if ($minutes_since_last_access > 5) {
                return; // Dispositivo no está realmente activo
            }
        }

        // Aquí podrías implementar una notificación en tiempo real
        // Por ejemplo, usando WebSockets, Server-Sent Events, o un sistema de colas
        
        // Por ahora, simplemente marcamos que hay configuración pendiente
        // El dispositivo la obtendrá en su próximo heartbeat/check
        
        // Log de la notificación
        error_log("Notificación de rollback enviada a dispositivo {$device_id}: {$summary}");

    } catch (Exception $e) {
        error_log("Error enviando notificación de rollback: " . $e->getMessage());
    }
}

/**
 * Función auxiliar para logging de sistema (si no existe en includes)
 */
function log_system_message($level, $message, $machine_id = null, $details = null) {
    try {
        db_insert('system_logs', [
            'log_type' => $level,
            'machine_id' => $machine_id,
            'message' => $message,
            'details' => $details,
            'timestamp' => date('Y-m-d H:i:s')
        ]);
    } catch (Exception $e) {
        error_log("Error logging system message: " . $e->getMessage());
    }
}
?>