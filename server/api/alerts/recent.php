<?php
/**
 * API para obtener alertas recientes
 * 
 * Este endpoint devuelve las alertas más recientes para actualizar
 * automáticamente el listado sin necesidad de recargar la página
 */

// Encabezados para permitir AJAX
header('Content-Type: application/json');
header('Cache-Control: no-cache, must-revalidate');

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';

// Crear instancia de Database para obtener la conexión
$database = new Database();
$pdo = $database->getConnection();

// Verificar si la conexión fue establecida
if ($pdo === null) {
    echo json_encode([
        'status' => 'error',
        'message' => 'Error de conexión a la base de datos',
        'alerts' => []
    ]);
    exit;
}

// Parámetros de la solicitud
$lastId = isset($_GET['last_id']) ? (int)$_GET['last_id'] : 0;
$limit = isset($_GET['limit']) ? (int)$_GET['limit'] : 10;
$acknowledgedFilter = isset($_GET['acknowledged']) ? (int)$_GET['acknowledged'] : -1; // -1 = todos, 0 = pendientes, 1 = atendidas

// Validar parámetros
if ($limit > 50) {
    $limit = 50; // Limitar a 50 registros como máximo
}

// Construir consulta SQL con información de dispositivos
$sql = "SELECT a.id, a.alert_type, a.timestamp, a.details, a.image_path, a.acknowledged,
               a.acknowledged_by, a.acknowledgement_time, a.device_id,
               o.id as operator_id, o.name as operator_name, o.dni_number as operator_dni,
               m.id as machine_id, m.name as machine_name,
               d.name as device_name, d.device_type, d.status as device_status
        FROM alerts a
        LEFT JOIN operators o ON a.operator_id = o.id
        LEFT JOIN machines m ON a.machine_id = m.id
        LEFT JOIN devices d ON a.device_id = d.device_id
        WHERE 1=1";

$params = [];

// Filtrar por ID (obtener solo alertas más recientes que la última vista)
if ($lastId > 0) {
    $sql .= " AND a.id > ?";
    $params[] = $lastId;
}

// Filtrar por estado de reconocimiento
if ($acknowledgedFilter >= 0) {
    $sql .= " AND a.acknowledged = ?";
    $params[] = $acknowledgedFilter;
}

// Ordenar y limitar resultados
$sql .= " ORDER BY a.id DESC LIMIT ?";
$params[] = $limit;

// Ejecutar consulta
try {
    $stmt = $pdo->prepare($sql);
    $stmt->execute($params);
    $alerts = $stmt->fetchAll(PDO::FETCH_ASSOC);
} catch (Exception $e) {
    echo json_encode([
        'status' => 'error',
        'message' => 'Error al obtener alertas: ' . $e->getMessage(),
        'alerts' => []
    ]);
    exit;
}

// Preparar respuesta
$response = [
    'status' => 'success',
    'timestamp' => date('Y-m-d H:i:s'),
    'count' => count($alerts),
    'alerts' => []
];

// Procesar alertas
foreach ($alerts as $alert) {
    // Definir etiquetas para los tipos de alerta
    $alertTypeLabels = [
        'fatigue' => 'Fatiga',
        'yawn' => 'Bostezo',
        'phone' => 'Uso de Teléfono',
        'smoking' => 'Fumando',
        'distraction' => 'Distracción',
        'unauthorized' => 'Operador No Autorizado',
        'behavior' => 'Comportamiento Inadecuado',
        'device_error' => 'Error de Dispositivo', // NUEVO
        'other' => 'Otro'
    ];
    
    // Formatear fecha y hora
    $timestamp = date('d/m/Y H:i', strtotime($alert['timestamp']));
    
    // Formatear fecha y hora de atención (si existe)
    $acknowledgementTime = !empty($alert['acknowledgement_time']) 
                          ? date('d/m/Y H:i', strtotime($alert['acknowledgement_time'])) 
                          : null;
    
    // Construir URL completa de la imagen
    $imageUrl = !empty($alert['image_path']) 
                ? (BASE_URL . '/' . $alert['image_path']) 
                : '';
    
    // Extraer una descripción corta de los detalles (primeras 100 caracteres)
    $shortDetails = !empty($alert['details']) 
                   ? (mb_substr($alert['details'], 0, 100) . (mb_strlen($alert['details']) > 100 ? '...' : '')) 
                   : '';
    
    // Agregar alerta a la respuesta
    $response['alerts'][] = [
        'id' => $alert['id'],
        'type' => $alert['alert_type'],
        'type_label' => $alertTypeLabels[$alert['alert_type']] ?? ucfirst($alert['alert_type']),
        'timestamp' => $timestamp,
        'operator' => [
            'id' => $alert['operator_id'],
            'name' => $alert['operator_name'],
            'dni' => $alert['operator_dni']
        ],
        'machine' => [
            'id' => $alert['machine_id'],
            'name' => $alert['machine_name']
        ],
        'device_id' => $alert['device_id'],
        'device_name' => $alert['device_name'],
        'device_type' => $alert['device_type'],
        'device_status' => $alert['device_status'],
        'image_url' => $imageUrl,
        'details' => $shortDetails,
        'acknowledged' => (bool)$alert['acknowledged'],
        'acknowledged_by' => $alert['acknowledged_by'],
        'acknowledgement_time' => $acknowledgementTime,
        'view_url' => 'view.php?id=' . $alert['id']
    ];
}

// Devolver respuesta JSON
echo json_encode($response);
?>