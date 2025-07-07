<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';
require_once '../../includes/auth.php';
require_once '../../models/User.php';

// Iniciar sesión
session_start();

// Verificar autenticación y permisos
Auth::requireRole('admin');

// Obtener ID del usuario a eliminar
$id = intval($_GET['id'] ?? 0);

if (!$id) {
    $_SESSION['error_message'] = 'ID de usuario no válido.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// No permitir que el administrador se elimine a sí mismo
if ($_SESSION['user_id'] == $id) {
    $_SESSION['error_message'] = 'No puede eliminar su propio usuario.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Instanciar modelo
$userModel = new User();

// Obtener datos del usuario para el log
$user = $userModel->getUserById($id);

if (!$user) {
    $_SESSION['error_message'] = 'Usuario no encontrado.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Eliminar usuario
$result = $userModel->deleteUser($id);

if ($result) {
    // Registrar en el log
    log_system_message(
        'warning',
        'Usuario ' . $_SESSION['username'] . ' ha eliminado el usuario ' . $user['username'],
        null,
        'ID: ' . $id
    );
    
    $_SESSION['success_message'] = 'Usuario eliminado correctamente.';
} else {
    $_SESSION['error_message'] = 'Error al eliminar el usuario.';
}

// Redireccionar
header('Location: ' . BASE_URL . '/pages/users/index.php');
exit;