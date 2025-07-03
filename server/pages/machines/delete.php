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

// Verificar si se proporcionó un ID de máquina
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$machineId = $_GET['id'];

// Obtener información de la máquina antes de eliminarla
$machine = db_fetch_one(
    "SELECT * FROM machines WHERE id = ?",
    [$machineId]
);

// Si no se encuentra la máquina, redirigir al listado
if (!$machine) {
    $_SESSION['error_message'] = "Máquina no encontrada.";
    header('Location: index.php');
    exit;
}

// Iniciar una transacción para asegurar integridad de datos
global $pdo;
$pdo->beginTransaction();

try {
    // 1. Eliminar las asignaciones de operadores
    $deleteAssignments = db_delete('operator_machine', 'machine_id = ?', [$machineId]);
    
    // 2. Actualizar las alertas relacionadas con esta máquina (mantenemos las alertas pero eliminamos la relación)
    $updateAlerts = db_update(
        'alerts',
        ['machine_id' => null, 'details' => 'Máquina eliminada: ' . $machine['name'] . ' (ID: ' . $machineId . ')'],
        'machine_id = ?',
        [$machineId]
    );
    
    // 3. Finalmente, eliminar la máquina
    $deleteMachine = db_delete('machines', 'id = ?', [$machineId]);
    
    if ($deleteMachine !== false) {
        // Confirmar la transacción
        $pdo->commit();
        
        // Registrar en el log
        log_system_message(
            'info',
            'Máquina eliminada: ' . $machine['name'] . ' (ID: ' . $machineId . ')',
            null,
            'Usuario: ' . $_SESSION['username']
        );
        
        $_SESSION['success_message'] = "Máquina eliminada correctamente.";
    } else {
        // Algo salió mal, revertir la transacción
        $pdo->rollBack();
        $_SESSION['error_message'] = "Error al eliminar la máquina.";
    }
} catch (Exception $e) {
    // Error en la transacción, revertir
    $pdo->rollBack();
    
    // Registrar el error
    log_system_message(
        'error',
        'Error al eliminar máquina: ' . $e->getMessage(),
        null,
        'Máquina ID: ' . $machineId
    );
    
    $_SESSION['error_message'] = "Error al eliminar la máquina: " . $e->getMessage();
}

// Redirigir al listado de máquinas
header('Location: index.php');
exit;
?>