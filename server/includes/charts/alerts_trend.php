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

// Obtener parámetros
$days = isset($_GET['days']) ? (int)$_GET['days'] : 30;
$groupBy = isset($_GET['group_by']) && $_GET['group_by'] == 'hour' ? 'hour' : 'day';

// Obtener datos de tendencia
$trend = $alertModel->getAlertsTrend($days, $groupBy);

// Formatear los datos para Chart.js
$data = [];
foreach ($trend as $item) {
    // Formatear la fecha según el agrupamiento
    if ($groupBy == 'hour') {
        $formattedDate = date('d/m H:i', strtotime($item['date_group']));
    } else {
        $formattedDate = date('d/m', strtotime($item['date_group']));
    }
    
    $data[] = [
        'date' => $formattedDate,
        'count' => (int)$item['count']
    ];
}

// Devolver como JSON
header('Content-Type: application/json');
echo json_encode($data);
?>