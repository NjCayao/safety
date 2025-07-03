<?php
/**
 * API para alertas (usado por las Raspberry Pi)
 */

// Encabezados para API
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: POST');
header('Access-Control-Allow-Headers: Access-Control-Allow-Headers, Content-Type, Access-Control-Allow-Methods, Authorization, X-Requested-With');

// Incluir archivos necesarios
require_once '../config/database.php';
require_once '../includes/functions.php';

// En un sistema real, verificarías el token de acceso aquí
// ...

// Verificar método de solicitud
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode([
        'success' => false,
        'message' => 'Método no permitido'
    ]);
    exit;
}

// Obtener datos enviados
$data = json_decode(file_get_contents('php://input'), true);

// Si no hay datos JSON, intentar con POST
if (!$data) {
    $data = $_POST;
}

// Verificar datos mínimos necesarios
if (!isset($data['operator_id']) || !isset($data['type'])) {
    echo json_encode([
        'success' => false,
        'message' => 'Faltan parámetros requeridos'
    ]);
    exit;
}

// Obtener y validar datos
$operatorId = (int)$data['operator_id'];
$type = $data['type'];
$description = $data['description'] ?? null;
$alertTime = $data['alert_time'] ?? date('Y-m-d H:i:s');

// Validar tipo de alerta
$validTypes = ['fatigue', 'distraction', 'yawn', 'behavior'];
if (!in_array($type, $validTypes)) {
    echo json_encode([
        'success' => false,
        'message' => 'Tipo de alerta no válido'
    ]);
    exit;
}

// Verificar si existe el operador
$operator = db_fetch_one(
    "SELECT id, name FROM operators WHERE id = ? AND status = 'active'",
    [$operatorId]
);

if (!$operator) {
    echo json_encode([
        'success' => false,
        'message' => 'Operador no encontrado o inactivo'
    ]);
    exit;
}

// Procesar imagen si se envió
$photoPath = null;
if (isset($_FILES['photo']) && $_FILES['photo']['error'] === UPLOAD_ERR_OK) {
    $result = uploadFile(
        $_FILES['photo'],
        '../uploads/alerts',
        ['jpg', 'jpeg', 'png'],
        5242880 // 5MB
    );
    
    if ($result['success']) {
        $photoPath = $result['path'];
    }
}

// Preparar datos para la inserción
$alertData = [
    'operator_id' => $operatorId,
    'type' => $type,
    'description' => $description,
    'photo_path' => $photoPath ? str_replace('../', '', $photoPath) : null,
    'alert_time' => $alertTime,
    'acknowledged' => 0,
    'acknowledged_at' => null,
    'acknowledged_by' => null
];

// Insertar alerta en la base de datos
$alertId = db_insert('alerts', $alertData);

if ($alertId) {
    echo json_encode([
        'success' => true,
        'message' => 'Alerta registrada correctamente',
        'alert_id' => $alertId
    ]);
} else {
    echo json_encode([
        'success' => false,
        'message' => 'Error al registrar la alerta'
    ]);
}