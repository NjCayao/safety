<?php
/**
 * Script auxiliar: Reintentar configuración desde historial
 * server/pages/devices/retry_config_history.php
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

    // Verificar que el registro tuvo error de aplicación
    if ($history_record['applied_successfully'] !== false) {
        echo json_encode([
            'success' => false, 
            'message' => 'Solo se pueden reintentar configuraciones que fallaron en la aplicación'
        ]);
        exit;
    }

    // Obtener configuración del historial
    $config_to_retry = json_decode($history_record['config_after'], true);
    if (!$config_to_retry) {
        echo json_encode(['success' => false, 'message' => 'Configuración de historial inválida']);
        exit;
    }

    // Crear resumen para el reintento
    $retry_summary = generateRetrySummary($history_record);

    // Aplicar configuración nuevamente usando el manager
    $result = DeviceConfigManager::updateDeviceConfig(
        $history_record['device_id'],
        $config_to_retry,
        $_SESSION['user_id'],
        $retry_summary
    );

    if (!$result['success']) {
        throw new Exception($result['error']);
    }

    // Marcar el registro original como reintentado
    db_update('device_config_history', [
        'retry_attempted' => true,
        'retry_at' => date('Y-m-d H:i:s'),
        'retry_by' => $_SESSION['user_id'],
        'retry_history_id' => $result['history_id']
    ], 'id = ?', [$history_id]);

    // Registrar en logs del sistema
    db_insert('system_logs', [
        'log_type' => 'info',
        'machine_id' => getDeviceMachineId($history_record['device_id']),
        'message' => "Configuración reintentada en dispositivo {$history_record['device_id']}",
        'details' => json_encode([
            'original_history_id' => $history_id,
            'new_history_id' => $result['history_id'],
            'retry_by_user' => $_SESSION['username'] ?? 'N/A',
            'original_error' => $history_record['error_message'],
            'original_date' => $history_record['created_at']
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);

    echo json_encode([
        'success' => true,
        'message' => 'Configuración reenviada correctamente para reintento',
        'retry_details' => [
            'device_id' => $history_record['device_id'],
            'device_name' => $history_record['device_name'],
            'original_date' => date('d/m/Y H:i', strtotime($history_record['created_at'])),
            'retry_summary' => $retry_summary,
            'new_history_id' => $result['history_id'],
            'original_error' => $history_record['error_message']
        ]
    ]);

} catch (Exception $e) {
    error_log("Error en retry_config_history.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al reintentar configuración: ' . $e->getMessage()
    ]);
}

/**
 * Genera resumen descriptivo del reintento
 */
function generateRetrySummary($history_record) {
    $date = date('d/m/Y H:i', strtotime($history_record['created_at']));
    $change_type = $history_record['change_type'];
    $user = $_SESSION['username'] ?? 'usuario';
    
    $change_type_names = [
        'manual' => 'cambio manual',
        'profile' => 'aplicación de perfil',
        'reset' => 'reset a configuración por defecto',
        'rollback' => 'rollback'
    ];
    
    $change_name = $change_type_names[$change_type] ?? $change_type;
    
    $summary = "Reintento por {$user} de configuración fallida del {$date} ({$change_name})";
    
    if ($history_record['changes_summary']) {
        $summary .= " - Original: " . $history_record['changes_summary'];
    }
    
    if ($history_record['error_message']) {
        $error_short = substr($history_record['error_message'], 0, 100);
        if (strlen($history_record['error_message']) > 100) {
            $error_short .= '...';
        }
        $summary .= " - Error previo: " . $error_short;
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
 * Función auxiliar para agregar columnas de retry si no existen
 */
function ensureRetryColumns() {
    try {
        // Verificar si las columnas de retry existen
        $columns = db_fetch_all("SHOW COLUMNS FROM device_config_history LIKE 'retry_%'");
        
        if (count($columns) < 4) {
            // Agregar columnas de retry si no existen
            $alter_queries = [
                "ALTER TABLE device_config_history ADD COLUMN retry_attempted TINYINT(1) DEFAULT 0 COMMENT 'Si se intentó reenviar esta configuración'",
                "ALTER TABLE device_config_history ADD COLUMN retry_at DATETIME NULL COMMENT 'Cuándo se reintentó'",
                "ALTER TABLE device_config_history ADD COLUMN retry_by INT(11) NULL COMMENT 'Usuario que reintentó'",
                "ALTER TABLE device_config_history ADD COLUMN retry_history_id INT(11) NULL COMMENT 'ID del nuevo registro de historial creado por el reintento'"
            ];
            
            foreach ($alter_queries as $query) {
                try {
                    db_query($query);
                } catch (Exception $e) {
                    // Columna puede ya existir, continuar
                }
            }
            
            // Agregar foreign keys si no existen
            try {
                db_query("ALTER TABLE device_config_history ADD CONSTRAINT fk_retry_by FOREIGN KEY (retry_by) REFERENCES users(id) ON DELETE SET NULL");
            } catch (Exception $e) {
                // FK puede ya existir
            }
        }
    } catch (Exception $e) {
        error_log("Error asegurando columnas de retry: " . $e->getMessage());
    }
}

// Asegurar que las columnas de retry existen
ensureRetryColumns();
?>