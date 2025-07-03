<?php
/**
 * Script auxiliar: Reintentar configuración fallida
 * server/pages/devices/retry_config.php
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

    // Verificar que hay configuración pendiente o con error
    if (!$device['config_pending']) {
        echo json_encode(['success' => false, 'message' => 'No hay configuración pendiente para este dispositivo']);
        exit;
    }

    // Marcar como pendiente nuevamente (en caso de que estuviera en error)
    $result = db_update(
        'devices',
        [
            'config_pending' => 1,
            'status' => 'online'  // Cambiar de error a online para que lo intente
        ],
        'device_id = ?',
        [$device_id]
    );

    if ($result === false) {
        throw new Exception('Error al actualizar estado del dispositivo');
    }

    // Registrar en historial que se reintentó
    $history_data = [
        'device_id' => $device_id,
        'changed_by' => $_SESSION['user_id'],
        'change_type' => 'manual',
        'config_before' => null,
        'config_after' => $device['config_json'],
        'changes_summary' => 'Configuración reenviada manualmente por ' . ($_SESSION['username'] ?? 'usuario'),
        'applied_successfully' => null,
        'created_at' => date('Y-m-d H:i:s')
    ];

    $history_id = db_insert('device_config_history', $history_data);

    // Registrar en logs del sistema
    db_insert('system_logs', [
        'log_type' => 'info',
        'machine_id' => $device['machine_id'],
        'message' => "Configuración reenviada para dispositivo {$device_id}",
        'details' => "Usuario: " . ($_SESSION['username'] ?? 'N/A') . ", Versión: {$device['config_version']}",
        'timestamp' => date('Y-m-d H:i:s')
    ]);

    echo json_encode([
        'success' => true,
        'message' => 'Configuración reenviada correctamente. El dispositivo intentará aplicarla en la próxima conexión.',
        'history_id' => $history_id
    ]);

} catch (Exception $e) {
    error_log("Error en retry_config.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error interno del servidor: ' . $e->getMessage()
    ]);
}
?>