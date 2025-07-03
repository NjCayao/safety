<?php
// api/v1/dashboard/recent_alerts.php

// Configurar cabeceras para API
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET');

// Incluir configuración y modelos
require_once '../../../config/config.php';
require_once '../../../models/Alert.php';

// Verificar autenticación (implementar según tu sistema)
// require_once '../../../utils/authenticate.php';

// Obtener parámetros
$limit = isset($_GET['limit']) ? intval($_GET['limit']) : 10;

// Crear respuesta
$response = [
    'success' => false,
    'message' => '',
    'data' => null
];

try {
    $alertModel = new Alert();
    $recentAlerts = $alertModel->getAlerts([], $limit, 0);

    // Mapeo de tipos de alerta para mostrar nombre legible
    $alertTypes = [
        'fatigue' => 'Fatiga',
        'drowsiness' => 'Somnolencia',
        'distraction' => 'Distracción',
        'phone' => 'Uso de celular',
        'smoking' => 'Fumando',
        'no_operator' => 'Sin operador',
        'unknown_operator' => 'Operador desconocido'
    ];

    // Formatear datos para la respuesta
    $formattedAlerts = [];
    foreach ($recentAlerts as $alert) {
        $type = $alert['alert_type'];
        $formattedAlerts[] = [
            'id' => $alert['id'],
            'created_at' => $alert['timestamp'],
            'created_at_formatted' => date('d/m/Y H:i', strtotime($alert['timestamp'])),
            'alert_type' => $type,
            'alert_type_name' => isset($alertTypes[$type]) ? $alertTypes[$type] : ucfirst($type),
            'operator_id' => $alert['operator_id'],
            'operator_name' => $alert['operator_name'],
            'machine_id' => $alert['machine_id'],
            'machine_name' => $alert['machine_name'],
            'acknowledged' => $alert['acknowledged'],
            'status_name' => $alert['acknowledged'] == 1 ? 'Revisado' : 'Sin revisar'
        ];
    }

    $response['success'] = true;
    $response['message'] = 'Alertas recientes obtenidas correctamente';
    $response['data'] = $formattedAlerts;
} catch (Exception $e) {
    $response['message'] = 'Error al obtener alertas recientes: ' . $e->getMessage();
}

// Devolver respuesta JSON
echo json_encode($response);
exit;
