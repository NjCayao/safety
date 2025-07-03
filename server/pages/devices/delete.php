<?php
// pages/devices/delete.php

// Primero iniciar sesión si no está iniciada
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Cargar configuración
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/auth.php';

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    header("Location: " . BASE_URL . "/login.php");
    exit();
}

// Obtener ID del dispositivo
$device_id = $_GET['id'] ?? null;

if (!$device_id) {
    header("Location: index.php?error=ID de dispositivo no especificado");
    exit();
}

// Verificar que el dispositivo existe
$device = db_fetch_one("SELECT * FROM devices WHERE id = ?", [$device_id]);

if (!$device) {
    header("Location: index.php?error=Dispositivo no encontrado");
    exit();
}

// Verificar si el dispositivo tiene alertas o eventos asociados
$alertCount = db_fetch_one(
    "SELECT COUNT(*) as count FROM alerts WHERE device_id = ?", 
    [$device['device_id']]
)['count'];

$eventCount = db_fetch_one(
    "SELECT COUNT(*) as count FROM events WHERE device_id = ?", 
    [$device['device_id']]
)['count'];

// Si tiene datos asociados, no permitir eliminación directa
if ($alertCount > 0 || $eventCount > 0) {
    header("Location: index.php?error=No se puede eliminar el dispositivo porque tiene alertas o eventos asociados");
    exit();
}

// Intentar eliminar el dispositivo
$result = db_delete('devices', 'id = ?', [$device_id]);

if ($result) {
    header("Location: index.php?msg=Dispositivo eliminado exitosamente");
} else {
    header("Location: index.php?error=Error al eliminar el dispositivo");
}
exit();
?>