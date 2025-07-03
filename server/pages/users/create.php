<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';
require_once '../../includes/auth.php';
require_once '../../models/User.php';
require_once '../../utils/password_helper.php';

// Iniciar sesión
session_start();

// Verificar autenticación y permisos
Auth::requireRole('admin');

// Verificar si el usuario está autenticado
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Instanciar modelo
$userModel = new User();

// Variables para el formulario
$name = '';
$username = '';
$email = '';
$role = '';
$status = 'active';
$errors = [];

// Procesar formulario
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener datos del formulario
    $name = trim($_POST['name'] ?? '');
    $username = trim($_POST['username'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $password = $_POST['password'] ?? '';
    $confirmPassword = $_POST['confirm_password'] ?? '';
    $role = $_POST['role'] ?? '';
    $status = $_POST['status'] ?? 'active';
    
    // Validar datos
    if (empty($name)) {
        $errors[] = 'El nombre es obligatorio.';
    }
    
    if (empty($username)) {
        $errors[] = 'El nombre de usuario es obligatorio.';
    } elseif ($userModel->getUserByUsername($username)) {
        $errors[] = 'El nombre de usuario ya está en uso.';
    }
    
    if (empty($email)) {
        $errors[] = 'El email es obligatorio.';
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $errors[] = 'El formato del email no es válido.';
    }
    
    if (empty($password)) {
        $errors[] = 'La contraseña es obligatoria.';
    } else {
        $passwordValidation = validate_password_strength($password);
        if ($passwordValidation !== true) {
            $errors[] = $passwordValidation;
        }
        
        if ($password !== $confirmPassword) {
            $errors[] = 'Las contraseñas no coinciden.';
        }
    }
    
    if (empty($role)) {
        $errors[] = 'El rol es obligatorio.';
    }
    
    // Si no hay errores, crear usuario
    if (empty($errors)) {
        $userData = [
            'name' => $name,
            'username' => $username,
            'password' => $password,
            'email' => $email,
            'role' => $role,
            'status' => $status
        ];
        
        $userId = $userModel->createUser($userData);
        
        if ($userId) {
            // Registrar en el log
            log_system_message(
                'info',
                'Usuario ' . $_SESSION['username'] . ' ha creado el usuario ' . $username,
                null,
                'ID: ' . $userId
            );
            
            // Redireccionar con mensaje de éxito
            $_SESSION['success_message'] = 'Usuario creado correctamente.';
            header('Location: ' . BASE_URL . '/pages/users/index.php');
            exit;
        } else {
            $errors[] = 'Error al crear el usuario. Inténtelo de nuevo.';
        }
    }
}

// Definir título de la página
$pageTitle = 'Crear Usuario';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => BASE_URL . '/index.php',
    'Usuarios' => BASE_URL . '/pages/users/index.php',
    'Crear Usuario' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Mensajes de error -->
<?php if (!empty($errors)): ?>
    <div class="alert alert-danger">
        <h5><i class="icon fas fa-ban"></i> Error!</h5>
        <ul>
            <?php foreach ($errors as $error): ?>
                <li><?php echo $error; ?></li>
            <?php endforeach; ?>
        </ul>
    </div>
<?php endif; ?>

<div class="card">
    <div class="card-header">
        <h3 class="card-title">Información del Usuario</h3>
    </div>
    <form action="<?php echo $_SERVER['PHP_SELF']; ?>" method="post">
        <div class="card-body">
            <div class="form-group">
                <label for="name">Nombre Completo</label>
                <input type="text" class="form-control" id="name" name="name" value="<?php echo htmlspecialchars($name); ?>" required>
            </div>
            <div class="form-group">
                <label for="username">Nombre de Usuario</label>
                <input type="text" class="form-control" id="username" name="username" value="<?php echo htmlspecialchars($username); ?>" required>
            </div>
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" class="form-control" id="email" name="email" value="<?php echo htmlspecialchars($email); ?>" required>
            </div>
            <div class="form-group">
                <label for="password">Contraseña</label>
                <input type="password" class="form-control" id="password" name="password" required>
                <small class="form-text text-muted">
                    La contraseña debe tener al menos 8 caracteres, incluir mayúsculas, minúsculas, números y caracteres especiales.
                </small>
            </div>
            <div class="form-group">
                <label for="confirm_password">Confirmar Contraseña</label>
                <input type="password" class="form-control" id="confirm_password" name="confirm_password" required>
            </div>
            <div class="form-group">
                <label for="role">Rol</label>
                <select class="form-control" id="role" name="role" required>
                    <option value="">Seleccione un rol</option>
                    <option value="admin" <?php echo ($role === 'admin') ? 'selected' : ''; ?>>Administrador</option>
                    <option value="supervisor" <?php echo ($role === 'supervisor') ? 'selected' : ''; ?>>Supervisor</option>
                    <option value="staff" <?php echo ($role === 'staff') ? 'selected' : ''; ?>>Staff</option>
                </select>
            </div>
            <div class="form-group">
                <label for="status">Estado</label>
                <select class="form-control" id="status" name="status">
                    <option value="active" <?php echo ($status === 'active') ? 'selected' : ''; ?>>Activo</option>
                    <option value="inactive" <?php echo ($status === 'inactive') ? 'selected' : ''; ?>>Inactivo</option>
                </select>
            </div>
        </div>
        <div class="card-footer">
            <button type="submit" class="btn btn-primary">Guardar</button>
            <a href="<?php echo BASE_URL; ?>/pages/users/index.php" class="btn btn-default">Cancelar</a>
        </div>
    </form>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="' . BASE_URL . '/pages/users/index.php" class="btn btn-secondary">
    <i class="fas fa-arrow-left"></i> Volver a Usuarios
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';
?>

<?php require_once '../../includes/footer.php'; ?>