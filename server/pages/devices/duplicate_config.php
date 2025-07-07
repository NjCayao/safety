<?php
/**
 * Script auxiliar: Duplicar configuración entre dispositivos
 * server/pages/devices/duplicate_config.php
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
$source_device_id = $_POST['source_device_id'] ?? null;
$target_device_ids = $_POST['target_device_ids'] ?? [];

if (!$source_device_id || empty($target_device_ids)) {
    echo json_encode(['success' => false, 'message' => 'Dispositivo origen y destino requeridos']);
    exit;
}

// Asegurar que target_device_ids es un array
if (!is_array($target_device_ids)) {
    $target_device_ids = [$target_device_ids];
}

try {
    // Verificar que el dispositivo origen existe
    $source_device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$source_device_id]);
    if (!$source_device) {
        echo json_encode(['success' => false, 'message' => 'Dispositivo origen no encontrado']);
        exit;
    }

    // Obtener configuración del dispositivo origen
    $source_config_data = DeviceConfigManager::getDeviceConfig($source_device_id);
    $source_config = $source_config_data['config'];

    if (empty($source_config)) {
        echo json_encode(['success' => false, 'message' => 'El dispositivo origen no tiene configuración']);
        exit;
    }

    $results = [];
    $success_count = 0;
    $error_count = 0;

    // Aplicar configuración a cada dispositivo destino
    foreach ($target_device_ids as $target_device_id) {
        try {
            // Verificar que el dispositivo destino existe
            $target_device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$target_device_id]);
            if (!$target_device) {
                $results[] = [
                    'device_id' => $target_device_id,
                    'success' => false,
                    'message' => 'Dispositivo no encontrado'
                ];
                $error_count++;
                continue;
            }

            // Verificar compatibilidad de tipos
            if ($source_device['device_type'] !== $target_device['device_type']) {
                $results[] = [
                    'device_id' => $target_device_id,
                    'success' => false,
                    'message' => 'Tipo de dispositivo incompatible'
                ];
                $error_count++;
                continue;
            }

            // Aplicar configuración
            $change_summary = "Configuración duplicada desde {$source_device_id} por " . ($_SESSION['username'] ?? 'usuario');
            
            $result = DeviceConfigManager::updateDeviceConfig(
                $target_device_id,
                $source_config,
                $_SESSION['user_id'],
                $change_summary
            );

            if ($result['success']) {
                $results[] = [
                    'device_id' => $target_device_id,
                    'device_name' => $target_device['name'],
                    'success' => true,
                    'message' => 'Configuración aplicada correctamente',
                    'history_id' => $result['history_id']
                ];
                $success_count++;

                // Registrar en logs
                db_insert('system_logs', [
                    'log_type' => 'info',
                    'machine_id' => $target_device['machine_id'],
                    'message' => "Configuración duplicada desde {$source_device_id} a {$target_device_id}",
                    'details' => json_encode([
                        'source_device' => $source_device_id,
                        'target_device' => $target_device_id,
                        'user' => $_SESSION['username'] ?? 'N/A'
                    ]),
                    'timestamp' => date('Y-m-d H:i:s')
                ]);

            } else {
                $results[] = [
                    'device_id' => $target_device_id,
                    'success' => false,
                    'message' => $result['error']
                ];
                $error_count++;
            }

        } catch (Exception $e) {
            $results[] = [
                'device_id' => $target_device_id,
                'success' => false,
                'message' => 'Error: ' . $e->getMessage()
            ];
            $error_count++;
        }
    }

    // Preparar respuesta final
    $total_devices = count($target_device_ids);
    
    if ($success_count === $total_devices) {
        $response = [
            'success' => true,
            'message' => "Configuración duplicada exitosamente a {$success_count} dispositivos",
            'summary' => [
                'total' => $total_devices,
                'success' => $success_count,
                'errors' => $error_count
            ],
            'results' => $results
        ];
    } elseif ($success_count > 0) {
        $response = [
            'success' => true,
            'message' => "Configuración aplicada parcialmente: {$success_count} exitosos, {$error_count} errores",
            'summary' => [
                'total' => $total_devices,
                'success' => $success_count,
                'errors' => $error_count
            ],
            'results' => $results
        ];
    } else {
        $response = [
            'success' => false,
            'message' => "No se pudo aplicar la configuración a ningún dispositivo",
            'summary' => [
                'total' => $total_devices,
                'success' => $success_count,
                'errors' => $error_count
            ],
            'results' => $results
        ];
    }

    echo json_encode($response);

} catch (Exception $e) {
    error_log("Error en duplicate_config.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al duplicar configuración: ' . $e->getMessage()
    ]);
}
?>