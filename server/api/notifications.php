<?php
/**
 * API para obtener notificaciones en tiempo real
 */

// Encabezados para API
header('Content-Type: application/json');

// Incluir archivos necesarios
require_once '../config/database.php';
require_once '../includes/functions.php';

// En un sistema real, verificarías la autenticación aquí
// ...

// Obtener alertas no reconocidas
$alerts = db_fetch_all(
    "SELECT a.id, a.type, a.description, a.alert_time, o.name as operator_name 
     FROM alerts a 
     JOIN operators o ON a.operator_id = o.id 
     WHERE a.acknowledged = 0 
     ORDER BY a.alert_time DESC 
     LIMIT 10"
);

// Formatear notificaciones
$notifications = [];
foreach ($alerts as $alert) {
    $type = $alert['type'];
    $operatorName = $alert['operator_name'];
    
    // Mensaje según el tipo de alerta
    switch ($type) {
        case 'fatigue':
            $message = "Fatiga detectada en $operatorName";
            break;
        case 'distraction':
            $message = "Distracción detectada en $operatorName";
            break;
        case 'yawn':
            $message = "Bostezo detectado en $operatorName";
            break;
        case 'behavior':
            $message = "Comportamiento anormal de $operatorName";
            break;
        default:
            $message = "Alerta para $operatorName";
    }
    
    $notifications[] = [
        'id' => $alert['id'],
        'type' => $type,
        'message' => $message,
        'time_ago' => timeAgo($alert['alert_time'])
    ];
}

echo json_encode([
    'success' => true,
    'notifications' => $notifications
]);