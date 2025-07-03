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

// Verificar si se proporcionó un ID de operador
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$operatorId = $_GET['id'];

// Obtener información del operador antes de eliminarlo
$operator = db_fetch_one(
    "SELECT * FROM operators WHERE id = ?",
    [$operatorId]
);

// Si no se encuentra el operador, redirigir al listado
if (!$operator) {
    $_SESSION['error_message'] = "Operador no encontrado.";
    header('Location: index.php');
    exit;
}

// Iniciar una transacción para asegurar integridad de datos
global $pdo;
$pdo->beginTransaction();

try {
    // 1. Eliminar las asignaciones de máquinas
    $deleteAssignments = db_delete('operator_machine', 'operator_id = ?', [$operatorId]);
    
    // 2. Actualizar las alertas relacionadas con este operador (mantenemos las alertas pero eliminamos la relación)
    $updateAlerts = db_update(
        'alerts',
        ['operator_id' => null, 'details' => 'Operador eliminado: ' . $operator['name'] . ' (ID: ' . $operatorId . ')'],
        'operator_id = ?',
        [$operatorId]
    );
    
    // 3. No eliminamos físicamente las fotos para mantener un historial
    // Simplemente se eliminará la referencia en la base de datos
    
    // 4. Finalmente, eliminar el operador
    $deleteOperator = db_delete('operators', 'id = ?', [$operatorId]);
    
    if ($deleteOperator !== false) {
        // Confirmar la transacción
        $pdo->commit();
        
        // Registrar en el log
        log_system_message(
            'info',
            'Operador eliminado: ' . $operator['name'] . ' (ID: ' . $operatorId . ')',
            null,
            'Usuario: ' . $_SESSION['username']
        );
        
        $_SESSION['success_message'] = "Operador eliminado correctamente.";
    } else {
        // Algo salió mal, revertir la transacción
        $pdo->rollBack();
        $_SESSION['error_message'] = "Error al eliminar el operador.";
    }
} catch (Exception $e) {
    // Error en la transacción, revertir
    $pdo->rollBack();
    
    // Registrar el error
    log_system_message(
        'error',
        'Error al eliminar operador: ' . $e->getMessage(),
        null,
        'Operador ID: ' . $operatorId
    );
    
    $_SESSION['error_message'] = "Error al eliminar el operador: " . $e->getMessage();
}

// Redirigir al listado de operadores
header('Location: index.php');
exit;
?>