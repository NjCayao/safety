<?php
// Verificar si la sesión ya está iniciada
if (session_status() == PHP_SESSION_NONE) {
    session_start();
}

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    // Si se accede por AJAX, devolver código de error
    if (!empty($_SERVER['HTTP_X_REQUESTED_WITH']) && strtolower($_SERVER['HTTP_X_REQUESTED_WITH']) == 'xmlhttprequest') {
        http_response_code(401);
        echo json_encode(['error' => 'No autorizado']);
        exit;
    }
    // Si no, redirigir al login
    header('Location: ../../login.php');
    exit;
}

// Incluir modelos y dependencias necesarias
$base_dir = dirname(dirname(dirname(__FILE__)));
require_once $base_dir . '/config/database.php';
require_once $base_dir . '/models/Alert.php';

// Crear instancia del modelo Alert
$alertModel = new Alert();

// Obtener parámetros de fecha
$dateFrom = isset($_GET['date_from']) ? $_GET['date_from'] : null;
$dateTo = isset($_GET['date_to']) ? $_GET['date_to'] : null;

// Obtener estadísticas de alertas por tipo
$stats = $alertModel->getAlertsByType($dateFrom, $dateTo);

// Definir etiquetas para los tipos de alerta
$alertTypeLabels = [
    'fatigue' => 'Fatiga',
    'yawn' => 'Bostezo',
    'phone' => 'Uso de Teléfono',
    'smoking' => 'Fumando',
    'distraction' => 'Distracción',
    'unauthorized' => 'Operador No Autorizado',
    'behavior' => 'Comportamiento Inadecuado',
    'other' => 'Otro'
];

// Formatear los datos para Chart.js
$data = [];
foreach ($stats as $stat) {
    $data[] = [
        'label' => $alertTypeLabels[$stat['alert_type']] ?? $stat['alert_type'],
        'count' => (int)$stat['count']
    ];
}

// Devolver como JSON
header('Content-Type: application/json');
echo json_encode($data);
?>