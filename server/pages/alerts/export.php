<?php
// Iniciar sesión
session_start();

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    header('Location: ../../login.php');
    exit;
}

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../models/Alert.php';

// Crear instancia del modelo Alert
$alertModel = new Alert();

// Procesar filtros
$filters = [];

// Tipo de alerta
if (isset($_GET['alert_type']) && !empty($_GET['alert_type'])) {
    $filters['alert_type'] = $_GET['alert_type'];
}

// Operador
if (isset($_GET['operator_id']) && !empty($_GET['operator_id'])) {
    $filters['operator_id'] = $_GET['operator_id'];
}

// Máquina
if (isset($_GET['machine_id']) && !empty($_GET['machine_id'])) {
    $filters['machine_id'] = $_GET['machine_id'];
}

// Fecha desde
if (isset($_GET['date_from']) && !empty($_GET['date_from'])) {
    $filters['date_from'] = $_GET['date_from'];
}

// Fecha hasta
if (isset($_GET['date_to']) && !empty($_GET['date_to'])) {
    $filters['date_to'] = $_GET['date_to'];
}

// Estado (reconocida o no)
if (isset($_GET['acknowledged']) && $_GET['acknowledged'] !== '') {
    $filters['acknowledged'] = (int)$_GET['acknowledged'];
}

// Obtener todas las alertas (sin límite)
$alerts = $alertModel->getAlerts($filters, 0, 0);

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

// Definir etiquetas para los estados
$acknowledgedLabels = [
    0 => 'Pendiente',
    1 => 'Atendida'
];

// Determinar el formato de exportación (por defecto: Excel)
$format = isset($_GET['format']) && $_GET['format'] == 'pdf' ? 'pdf' : 'excel';

if ($format == 'excel') {
    // Exportar a Excel
    header('Content-Type: application/vnd.ms-excel');
    header('Content-Disposition: attachment; filename="alertas_' . date('Y-m-d') . '.xls"');
    header('Pragma: no-cache');
    header('Expires: 0');
    
    echo '<table border="1">';
    echo '<tr>';
    echo '<th>ID</th>';
    echo '<th>Tipo de Alerta</th>';
    echo '<th>Operador</th>';
    echo '<th>Máquina</th>';
    echo '<th>Fecha y Hora</th>';
    echo '<th>Estado</th>';
    echo '<th>Detalles</th>';
    echo '</tr>';
    
    foreach ($alerts as $alert) {
        echo '<tr>';
        echo '<td>' . $alert['id'] . '</td>';
        echo '<td>' . ($alertTypeLabels[$alert['alert_type']] ?? $alert['alert_type']) . '</td>';
        echo '<td>' . ($alert['operator_name'] ?? 'N/A') . '</td>';
        echo '<td>' . ($alert['machine_name'] ?? 'N/A') . '</td>';
        echo '<td>' . date('d/m/Y H:i', strtotime($alert['timestamp'])) . '</td>';
        echo '<td>' . $acknowledgedLabels[$alert['acknowledged']] . '</td>';
        echo '<td>' . $alert['details'] . '</td>';
        echo '</tr>';
    }
    
    echo '</table>';
} else {
    // Exportar a PDF - No implementado aquí, pero podría agregar TCPDF o similar
    // Por ahora, mostramos un mensaje informativo
    header('Content-Type: text/html; charset=utf-8');
    echo '<!DOCTYPE html>';
    echo '<html>';
    echo '<head>';
    echo '<meta charset="UTF-8">';
    echo '<meta name="viewport" content="width=device-width, initial-scale=1.0">';
    echo '<title>Exportar a PDF</title>';
    echo '<style>body { font-family: Arial, sans-serif; margin: 20px; }</style>';
    echo '</head>';
    echo '<body>';
    echo '<h2>Exportación a PDF</h2>';
    echo '<p>La exportación a PDF requiere la biblioteca TCPDF o similar. Por favor, implemente esta funcionalidad según sus necesidades.</p>';
    echo '<p><a href="index.php">Volver al listado de alertas</a></p>';
    echo '</body>';
    echo '</html>';
}
?>