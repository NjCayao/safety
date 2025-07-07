<?php
// Incluir archivos necesarios
require_once 'config/config.php';
require_once 'config/database.php';
require_once 'includes/functions.php';
require_once 'includes/auth.php';

// Iniciar sesión
session_start();

// Cerrar sesión usando el método de la clase Auth
Auth::logout();

// Registrar logout en el log si hay usuario autenticado
if (isset($_SESSION['user_id'])) {
    log_system_message(
        'info',
        'Usuario ' . ($_SESSION['username'] ?? 'desconocido') . ' ha cerrado sesión',
        null,
        'IP: ' . $_SERVER['REMOTE_ADDR']
    );
}

// Destruir la sesión
session_unset();
session_destroy();

// Redirigir al login
header('Location: login.php');
exit;
?>
