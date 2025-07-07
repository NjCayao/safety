<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Verificar si se proporcionaron los parámetros necesarios
if (!isset($_GET['id']) || !isset($_GET['operator_id'])) {
    header('Location: ../operators/index.php');
    exit;
}

$assignmentId = $_GET['id'];
$operatorId = $_GET['operator_id'];

// Obtener información de la asignación
$assignment = db_fetch_one(
    "SELECT om.*, o.name as operator_name, m.name as machine_name 
     FROM operator_machine om 
     JOIN operators o ON om.operator_id = o.id 
     JOIN machines m ON om.machine_id = m.id 
     WHERE om.id = ? AND om.operator_id = ? AND om.is_current = 1",
    [$assignmentId, $operatorId]
);

// Si no se encuentra la asignación, redirigir a la vista del operador
if (!$assignment) {
    $_SESSION['error_message'] = "Asignación no encontrada.";
    header('Location: ../operators/view.php?id=' . $operatorId);
    exit;
}

// Actualizar la asignación para marcarla como no actual y registrar la fecha de fin
$updateData = [
    'is_current' => 0,
    'end_date' => date('Y-m-d H:i:s')
];

$result = db_update('operator_machine', $updateData, 'id = ?', [$assignmentId]);

if ($result !== false) {
    // Registrar en el log
    log_system_message(
        'info',
        'Máquina desasignada: ' . $assignment['machine_name'] . ' del operador: ' . $assignment['operator_name'],
        $assignment['machine_id'],
        'Usuario: ' . $_SESSION['username']
    );
    
    $_SESSION['success_message'] = "Máquina desasignada correctamente.";
} else {
    $_SESSION['error_message'] = "Error al desasignar la máquina.";
}

// Redirigir a la página de vista del operador o a la página de asignación
if (isset($_GET['redirect']) && $_GET['redirect'] === 'assign') {
    header('Location: ../operators/assign.php?id=' . $operatorId);
} else {
    header('Location: ../operators/view.php?id=' . $operatorId);
}
exit;
?>