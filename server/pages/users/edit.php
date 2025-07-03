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

// Verificar autenticación
Auth::requireLogin();

// Obtener ID del usuario a editar
$id = intval($_GET['id'] ?? 0);

if (!$id) {
    $_SESSION['error_message'] = 'ID de usuario no válido.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Comprobar que sea el propio usuario o un administrador
if ($_SESSION['user_id'] != $id && !Auth::hasRole('admin')) {
    $_SESSION['error_message'] = 'No tiene permisos para editar este usuario.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Instanciar modelo
$userModel = new User();

// Obtener datos del usuario
$user = $userModel->getUserById($id);

if (!$user) {
    $_SESSION['error_message'] = 'Usuario no encontrado.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Variables para el formulario
$name = $user['name'];
$username = $user['username'];
$email = $user['email'];
$role = $user['role'];
$status = $user['status'];
$errors = [];

// Procesar formulario
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener datos del formulario
    $name = trim($_POST['name'] ?? '');
    $username = trim($_POST['username'] ?? '');
    $email = trim($_POST['email'] ?? '');
    $password = $_POST['password'] ?? '';
    $confirmPassword = $_POST['confirm_password'] ?? '';
    $role = $_POST['role'] ?? $user['role'];
    $status = $_POST['status'] ?? $user['status'];
    
    // Validar datos
    if (empty($name)) {
        $errors[] = 'El nombre es obligatorio.';
    }
    
    if (empty($username)) {
        $errors[] = 'El nombre de usuario es obligatorio.';
    } elseif ($username !== $user['username']) {
        $existingUser = $userModel->getUserByUsername($username);
        if ($existingUser && $existingUser['id'] != $id) {
            $errors[] = 'El nombre de usuario ya está en uso.';
        }
    }
    
    if (empty($email)) {
        $errors[] = 'El email es obligatorio.';
    } elseif (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
        $errors[] = 'El formato del email no es válido.';
    }
    
    // Solo validar contraseña si se proporciona
    if (!empty($password)) {
        $passwordValidation = validate_password_strength($password);
        if ($passwordValidation !== true) {
            $errors[] = $passwordValidation;
        }
        
        if ($password !== $confirmPassword) {
            $errors[] = 'Las contraseñas no coinciden.';
        }
    }
    
    // Solo el admin puede cambiar roles
    if (!Auth::hasRole('admin')) {
        $role = $user['role'];
    }
    
    // Si no hay errores, actualizar usuario
    if (empty($errors)) {
        $userData = [
            'name' => $name,
            'username' => $username,
            'email' => $email,
            'role' => $role,
            'status' => $status
        ];
        
        // Agregar contraseña si se proporcionó
        if (!empty($password)) {
            $userData['password'] = $password;
        }
        
        $result = $userModel->updateUser($id, $userData);
        
        if ($result) {
            // Registrar en el log
            log_system_message(
                'info',
                'Usuario ' . $_SESSION['username'] . ' ha actualizado el usuario ' . $username,
                null,
                'ID: ' . $id
            );
            
            // Si el usuario actualiza sus propios datos, actualizar la sesión
            if ($_SESSION['user_id'] == $id) {
                $_SESSION['name'] = $name;
                $_SESSION['username'] = $username;
                $_SESSION['role'] = $role;
                $_SESSION['user_role'] = $role;
            }
            
            // Redireccionar con mensaje de éxito
            $_SESSION['success_message'] = 'Usuario actualizado correctamente.';
            header('Location: ' . BASE_URL . '/pages/users/index.php');
            exit;
        } else {
            $errors[] = 'Error al actualizar el usuario. Inténtelo de nuevo.';
        }
    }
}

// Incluir header
$pageTitle = 'Editar Usuario';
include '../../includes/header.php';
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Editar Usuario</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/pages/users/index.php">Usuarios</a></li>
                        <li class="breadcrumb-item active">Editar Usuario</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
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
                <form action="<?php echo $_SERVER['PHP_SELF'] . '?id=' . $id; ?>" method="post">
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
                            <input type="password" class="form-control" id="password" name="password">
                            <small class="form-text text-muted">
                                Dejar en blanco para mantener la contraseña actual.
                            </small>
                        </div>
                        <div class="form-group">
                            <label for="confirm_password">Confirmar Contraseña</label>
                            <input type="password" class="form-control" id="confirm_password" name="confirm_password">
                        </div>
                        
                        <?php if (Auth::hasRole('admin')): ?>
                        <div class="form-group">
                            <label for="role">Rol</label>
                            <select class="form-control" id="role" name="role" required>
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
                        <?php else: ?>
                        <input type="hidden" name="role" value="<?php echo $role; ?>">
                        <input type="hidden" name="status" value="<?php echo $status; ?>">
                        <?php endif; ?>
                    </div>
                    <div class="card-footer">
                        <button type="submit" class="btn btn-primary">Guardar Cambios</button>
                        <a href="<?php echo BASE_URL; ?>/pages/users/index.php" class="btn btn-default">Cancelar</a>
                    </div>
                </form>
            </div>
        </div>
    </section>
</div>

<?php include '../../includes/footer.php'; ?>