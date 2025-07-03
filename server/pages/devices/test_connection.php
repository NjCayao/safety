<?php
/**
 * Script auxiliar: Probar conexión con dispositivo
 * server/pages/devices/test_connection.php
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

    // Calcular tiempo desde última conexión
    $last_seen = null;
    $status_info = [];
    
    if ($device['last_access']) {
        $last_access_time = strtotime($device['last_access']);
        $now = time();
        $diff_minutes = ($now - $last_access_time) / 60;
        
        $last_seen = date('d/m/Y H:i:s', $last_access_time);
        
        if ($diff_minutes < 2) {
            $status_info['connection'] = 'online';
            $status_info['message'] = 'Dispositivo en línea';
        } elseif ($diff_minutes < 10) {
            $status_info['connection'] = 'recent';
            $status_info['message'] = 'Última conexión reciente (' . round($diff_minutes) . ' min)';
        } elseif ($diff_minutes < 60) {
            $status_info['connection'] = 'warning';
            $status_info['message'] = 'Sin conexión reciente (' . round($diff_minutes) . ' min)';
        } else {
            $status_info['connection'] = 'offline';
            $hours = round($diff_minutes / 60, 1);
            if ($hours > 24) {
                $days = round($hours / 24, 1);
                $status_info['message'] = "Dispositivo offline (${days} días)";
            } else {
                $status_info['message'] = "Dispositivo offline (${hours} horas)";
            }
        }
    } else {
        $status_info['connection'] = 'never';
        $status_info['message'] = 'Dispositivo nunca se ha conectado';
    }

    // Verificar conectividad de red si tenemos IP
    $ping_result = null;
    if ($device['ip_address']) {
        $ping_result = testNetworkConnectivity($device['ip_address']);
    }

    // Obtener información adicional del dispositivo
    $device_info = [
        'status' => $device['status'],
        'last_access' => $last_seen,
        'last_sync' => $device['last_sync'] ? date('d/m/Y H:i:s', strtotime($device['last_sync'])) : 'Nunca',
        'ip_address' => $device['ip_address'],
        'config_version' => $device['config_version'],
        'config_pending' => (bool)$device['config_pending']
    ];

    // Respuesta según el estado
    if ($status_info['connection'] === 'online' || $status_info['connection'] === 'recent') {
        echo json_encode([
            'success' => true,
            'message' => $status_info['message'],
            'last_seen' => $last_seen,
            'connection_status' => $status_info['connection'],
            'device_info' => $device_info,
            'ping_result' => $ping_result
        ]);
    } else {
        echo json_encode([
            'success' => false,
            'message' => $status_info['message'],
            'last_seen' => $last_seen,
            'connection_status' => $status_info['connection'],
            'device_info' => $device_info,
            'ping_result' => $ping_result,
            'recommendations' => getConnectionRecommendations($status_info['connection'], $ping_result)
        ]);
    }

} catch (Exception $e) {
    error_log("Error en test_connection.php: " . $e->getMessage());
    
    echo json_encode([
        'success' => false,
        'message' => 'Error al probar conexión: ' . $e->getMessage()
    ]);
}

/**
 * Probar conectividad de red con ping
 */
function testNetworkConnectivity($ip_address) {
    // Validar IP
    if (!filter_var($ip_address, FILTER_VALIDATE_IP)) {
        return ['success' => false, 'message' => 'IP inválida'];
    }

    // Ejecutar ping (solo en sistemas Unix/Linux)
    if (strtoupper(substr(PHP_OS, 0, 3)) === 'WIN') {
        $command = "ping -n 3 " . escapeshellarg($ip_address);
    } else {
        $command = "ping -c 3 -W 3 " . escapeshellarg($ip_address);
    }

    $output = [];
    $return_code = 0;
    
    exec($command, $output, $return_code);
    
    if ($return_code === 0) {
        // Ping exitoso
        $ping_info = [
            'success' => true,
            'message' => 'Dispositivo responde a ping',
            'ip' => $ip_address
        ];
        
        // Extraer tiempo de respuesta si es posible
        foreach ($output as $line) {
            if (preg_match('/time[<=](\d+\.?\d*)/', $line, $matches)) {
                $ping_info['response_time'] = $matches[1] . 'ms';
                break;
            }
        }
        
        return $ping_info;
    } else {
        return [
            'success' => false,
            'message' => 'Dispositivo no responde a ping',
            'ip' => $ip_address
        ];
    }
}

/**
 * Obtener recomendaciones según el estado de conexión
 */
function getConnectionRecommendations($connection_status, $ping_result) {
    $recommendations = [];
    
    switch ($connection_status) {
        case 'never':
            $recommendations[] = 'Verificar configuración inicial del dispositivo';
            $recommendations[] = 'Comprobar que la API key esté configurada correctamente';
            $recommendations[] = 'Verificar conectividad de red del dispositivo';
            break;
            
        case 'offline':
            $recommendations[] = 'Verificar que el dispositivo esté encendido';
            $recommendations[] = 'Comprobar conexión de red (WiFi/Ethernet)';
            $recommendations[] = 'Verificar que el servicio esté ejecutándose';
            break;
            
        case 'warning':
            $recommendations[] = 'Verificar estabilidad de la conexión de red';
            $recommendations[] = 'Comprobar logs del dispositivo por errores';
            break;
    }
    
    // Recomendaciones basadas en ping
    if ($ping_result && !$ping_result['success']) {
        $recommendations[] = 'Dispositivo no responde a ping - verificar red';
        $recommendations[] = 'Comprobar configuración de firewall';
    }
    
    // Recomendaciones generales
    $recommendations[] = 'Contactar al técnico de campo si persiste el problema';
    
    return $recommendations;
}
?>