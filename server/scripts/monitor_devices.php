<?php
// scripts/monitor_devices.php

require_once '../config/database.php';

// Configuración
$OFFLINE_THRESHOLD_MINUTES = 5; // Minutos sin heartbeat para marcar como offline
$ERROR_THRESHOLD_MINUTES = 15;  // Minutos sin heartbeat para generar alerta

// Registrar inicio del monitoreo
db_insert('system_logs', [
    'log_type' => 'info',
    'message' => 'Iniciando monitoreo de dispositivos',
    'details' => 'Script: monitor_devices.php',
    'timestamp' => date('Y-m-d H:i:s')
]);

// Obtener todos los dispositivos activos
$devices = db_fetch_all("
    SELECT d.*, m.name as machine_name
    FROM devices d
    LEFT JOIN machines m ON d.machine_id = m.id
    WHERE d.status != 'inactive'
");

$devices_processed = 0;
$alerts_created = 0;

foreach ($devices as $device) {
    if ($device['last_access']) {
        $last_access = new DateTime($device['last_access']);
        $now = new DateTime();
        $diff_minutes = ($now->getTimestamp() - $last_access->getTimestamp()) / 60;
        
        // Verificar si debe marcarse como offline
        if ($diff_minutes > $OFFLINE_THRESHOLD_MINUTES && $device['status'] !== 'offline') {
            db_query(
                "UPDATE devices SET status = 'offline' WHERE id = ?",
                [$device['id']]
            );
            
            db_insert('system_logs', [
                'log_type' => 'warning',
                'machine_id' => $device['machine_id'],
                'message' => "Dispositivo {$device['device_id']} marcado como offline",
                'details' => "Sin heartbeat por {$diff_minutes} minutos",
                'timestamp' => date('Y-m-d H:i:s')
            ]);
        }
        
        // Verificar si debe generar alerta técnica
        if ($diff_minutes > $ERROR_THRESHOLD_MINUTES) {
            // Verificar si ya existe una alerta reciente sin atender
            $recent_alert = db_fetch_one(
                "SELECT id FROM alerts 
                 WHERE device_id = ? 
                 AND alert_type = 'other' 
                 AND details LIKE '%desconexión%'
                 AND acknowledged = 0 
                 AND timestamp > DATE_SUB(NOW(), INTERVAL 1 HOUR)",
                [$device['device_id']]
            );
            
            if (!$recent_alert) {
                // Crear alerta técnica
                $alert_data = [
                    'device_id' => $device['device_id'],
                    'machine_id' => $device['machine_id'],
                    'alert_type' => 'other',
                    'timestamp' => date('Y-m-d H:i:s'),
                    'details' => "Dispositivo sin conexión por más de {$ERROR_THRESHOLD_MINUTES} minutos. Última conexión: " . $device['last_access'],
                    'acknowledged' => 0
                ];
                
                $alert_id = db_insert('alerts', $alert_data);
                
                if ($alert_id) {
                    $alerts_created++;
                    
                    db_insert('system_logs', [
                        'log_type' => 'error',
                        'machine_id' => $device['machine_id'],
                        'message' => "Alerta creada por dispositivo desconectado: {$device['device_id']}",
                        'details' => "Sin conexión por {$diff_minutes} minutos",
                        'timestamp' => date('Y-m-d H:i:s')
                    ]);
                }
            }
        }
    } else {
        // Dispositivo nunca se ha conectado
        if ($device['status'] !== 'offline') {
            db_query(
                "UPDATE devices SET status = 'offline' WHERE id = ?",
                [$device['id']]
            );
        }
    }
    
    $devices_processed++;
}

// Actualizar dispositivos que han vuelto a estar online
db_query(
    "UPDATE devices 
     SET status = 'online' 
     WHERE status = 'offline' 
     AND last_access > DATE_SUB(NOW(), INTERVAL ? MINUTE)",
    [$OFFLINE_THRESHOLD_MINUTES]
);

// Registrar fin del monitoreo
db_insert('system_logs', [
    'log_type' => 'info',
    'message' => "Monitoreo de dispositivos finalizado. Procesados: {$devices_processed}. Alertas creadas: {$alerts_created}",
    'details' => 'Script: monitor_devices.php',
    'timestamp' => date('Y-m-d H:i:s')
]);

echo "Monitoreo completado. Dispositivos procesados: {$devices_processed}, Alertas creadas: {$alerts_created}\n";
?>