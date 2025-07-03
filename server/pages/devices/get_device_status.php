<?php
/**
 * Script auxiliar: Obtener estado actualizado de dispositivos
 * server/pages/devices/get_device_status.php
 */

header('Content-Type: application/json');

// Verificar sesión
session_start();
if (!isset($_SESSION['user_id'])) {
    http_response_code(401);
    echo json_encode(['success' => false, 'message' => 'No autorizado']);
    exit;
}

// Incluir dependencias
require_once '../../config/config.php';
require_once '../../config/database.php';

try {
    // Obtener dispositivos con información de estado actualizada
    $devices = db_fetch_all("
        SELECT d.*, 
               m.name as machine_name,
               m.location as machine_location,
               CASE 
                   WHEN d.last_access IS NULL THEN 'never_connected'
                   WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 2 MINUTE) AND d.status = 'online' THEN 'stale_online'
                   WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 5 MINUTE) THEN 'offline'
                   WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 1 HOUR) THEN 'long_offline'
                   ELSE d.status
               END as computed_status,
               TIMESTAMPDIFF(MINUTE, d.last_access, NOW()) as minutes_since_last_access,
               (SELECT COUNT(*) FROM alerts a WHERE a.device_id = d.device_id AND a.acknowledged = 0 AND a.timestamp > DATE_SUB(NOW(), INTERVAL 24 HOUR)) as pending_alerts,
               (SELECT COUNT(*) FROM events e WHERE e.device_id = d.device_id AND e.is_synced = 0) as pending_events
        FROM devices d
        LEFT JOIN machines m ON d.machine_id = m.id
        ORDER BY d.last_access DESC, d.device_id ASC
    ");

    // Procesar cada dispositivo para agregar información de estado
    $processed_devices = [];
    foreach ($devices as $device) {
        $status_info = getStatusInfo($device['computed_status'], $device['minutes_since_last_access']);
        $last_seen_text = getLastSeenText($device['last_access'], $device['minutes_since_last_access']);
        
        $processed_devices[] = array_merge($device, [
            'status_info' => $status_info,
            'last_seen_text' => $last_seen_text,
            'last_access_formatted' => $device['last_access'] ? 
                date('d/m/Y H:i', strtotime($device['last_access'])) : null
        ]);
    }

    // Calcular estadísticas
    $stats = [
        'total' => count($devices),
        'online' => 0,
        'offline' => 0,
        'error' => 0,
        'never_connected' => 0,
        'pending_alerts' => 0,
        'pending_events' => 0
    ];

    foreach ($devices as $device) {
        switch ($device['computed_status']) {
            case 'online':
                $stats['online']++;
                break;
            case 'offline':
            case 'long_offline':
            case 'stale_online':
                $stats['offline']++;
                break;
            case 'error':
                $stats['error']++;
                break;
            case 'never_connected':
                $stats['never_connected']++;
                break;
        }
        $stats['pending_alerts'] += $device['pending_alerts'];
        $stats['pending_events'] += $device['pending_events'];
    }

    echo json_encode([
        'success' => true,
        'devices' => $processed_devices,
        'stats' => $stats,
        'timestamp' => date('Y-m-d H:i:s')
    ]);

} catch (Exception $e) {
    error_log("Error en get_device_status.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al obtener estado de dispositivos: ' . $e->getMessage()
    ]);
}

/**
 * Obtener información de estado según el estado computado
 */
function getStatusInfo($computed_status, $minutes_since_last_access) {
    switch ($computed_status) {
        case 'online':
            return [
                'class' => 'success',
                'icon' => 'fa-circle',
                'text' => 'En línea',
                'description' => 'Conectado y funcionando'
            ];
        case 'stale_online':
            return [
                'class' => 'warning',
                'icon' => 'fa-circle',
                'text' => 'Conexión irregular',
                'description' => 'Marcado como online pero sin actividad reciente'
            ];
        case 'offline':
            $time_desc = $minutes_since_last_access < 60 ? 
                $minutes_since_last_access . ' min' : 
                round($minutes_since_last_access / 60, 1) . ' hrs';
            return [
                'class' => 'danger',
                'icon' => 'fa-circle',
                'text' => 'Fuera de línea',
                'description' => "Última conexión hace $time_desc"
            ];
        case 'long_offline':
            $hours = round($minutes_since_last_access / 60);
            $time_desc = $hours > 24 ? round($hours / 24, 1) . ' días' : $hours . ' hrs';
            return [
                'class' => 'secondary',
                'icon' => 'fa-circle',
                'text' => 'Offline prolongado',
                'description' => "Sin conexión hace $time_desc"
            ];
        case 'syncing':
            return [
                'class' => 'info',
                'icon' => 'fa-sync fa-spin',
                'text' => 'Sincronizando',
                'description' => 'Transfiriendo datos'
            ];
        case 'error':
            return [
                'class' => 'danger',
                'icon' => 'fa-exclamation-triangle',
                'text' => 'Error',
                'description' => 'Error de comunicación'
            ];
        case 'never_connected':
            return [
                'class' => 'dark',
                'icon' => 'fa-question-circle',
                'text' => 'Nunca conectado',
                'description' => 'Dispositivo no se ha conectado'
            ];
        default:
            return [
                'class' => 'secondary',
                'icon' => 'fa-question',
                'text' => 'Desconocido',
                'description' => 'Estado no determinado'
            ];
    }
}

/**
 * Obtener texto de última conexión
 */
function getLastSeenText($last_access, $minutes_since_last_access) {
    if (!$last_access) return 'Nunca';
    
    if ($minutes_since_last_access < 1) return 'Ahora mismo';
    if ($minutes_since_last_access < 60) return "Hace {$minutes_since_last_access} min";
    
    $hours = round($minutes_since_last_access / 60);
    if ($hours < 24) return "Hace {$hours} hrs";
    
    $days = round($hours / 24, 1);
    return "Hace {$days} días";
}
?>