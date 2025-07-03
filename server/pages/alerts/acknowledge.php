<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../models/Alert.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Verificar si el usuario tiene permisos (admin o supervisor)
if (!isset($_SESSION['user_role']) || ($_SESSION['user_role'] != 'admin' && $_SESSION['user_role'] != 'supervisor')) {
    $_SESSION['error_message'] = 'No tiene permisos para realizar esta acción.';
    header('Location: index.php');
    exit;
}

// Verificar si se proporcionó un ID de alerta
if (!isset($_GET['id']) || empty($_GET['id'])) {
    $_SESSION['error_message'] = 'ID de alerta no especificado.';
    header('Location: index.php');
    exit;
}

$alertId = (int)$_GET['id'];

// Crear instancia del modelo Alert
$alertModel = new Alert();

// Obtener detalles de la alerta
$alert = $alertModel->getAlertById($alertId);

// Si la alerta no existe, redirigir al listado
if (!$alert) {
    $_SESSION['error_message'] = 'La alerta especificada no existe.';
    header('Location: index.php');
    exit;
}

// Si la alerta ya está atendida, mostrar mensaje y redirigir
if ($alert['acknowledged']) {
    $_SESSION['info_message'] = 'Esta alerta ya fue atendida previamente.';
    header('Location: view.php?id=' . $alertId);
    exit;
}

// Obtener usuario que atiende la alerta
$acknowledgedBy = $_SESSION['username'] ?? 'Usuario';

// Marcar la alerta como atendida
if ($alertModel->markAsAcknowledged($alertId, $acknowledgedBy)) {
    $_SESSION['success_message'] = 'Alerta marcada como atendida exitosamente.';
    
    // Registrar en el log del sistema usando una función alternativa
    // que sabemos que está disponible en database.php
    try {
        // Crear instancia de la base de datos
        $database = new Database();
        $pdo = $database->getConnection();
        
        // Insertar directamente en la tabla system_logs
        $logSql = "INSERT INTO system_logs (log_type, machine_id, message, details) VALUES (?, ?, ?, ?)";
        $stmt = $pdo->prepare($logSql);
        $stmt->execute([
            'info',
            $alert['machine_id'],
            'Alerta #' . $alertId . ' marcada como atendida por ' . $acknowledgedBy,
            'Tipo: ' . $alert['alert_type'] . ', Operador: ' . $alert['operator_id']
        ]);
    } catch (Exception $e) {
        // Ignorar errores de registro, no queremos interrumpir el flujo principal
        // pero podemos añadir el error al mensaje de sesión si queremos
        // $_SESSION['warning_message'] = 'Nota: Error al registrar en log del sistema: ' . $e->getMessage();
    }
    
    // Redirigir a la vista de la alerta
    header('Location: view.php?id=' . $alertId);
    exit;
} else {
    $_SESSION['error_message'] = 'Error al intentar marcar la alerta como atendida.';
    header('Location: view.php?id=' . $alertId);
    exit;
}
?>