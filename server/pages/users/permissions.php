<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';
require_once '../../includes/auth.php';
require_once '../../models/User.php';
require_once '../../models/Permission.php';

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

// Obtener ID del usuario
$id = intval($_GET['id'] ?? 0);

if (!$id) {
    $_SESSION['error_message'] = 'ID de usuario no válido.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Instanciar modelos
$userModel = new User();
$permissionModel = new Permission();

// Obtener datos del usuario
$user = $userModel->getUserById($id);

if (!$user) {
    $_SESSION['error_message'] = 'Usuario no encontrado.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// No se pueden editar permisos de administradores
if ($user['role'] === 'admin') {
    $_SESSION['info_message'] = 'Los administradores tienen todos los permisos por defecto.';
    header('Location: ' . BASE_URL . '/pages/users/index.php');
    exit;
}

// Obtener todos los módulos (usando DISTINCT para evitar duplicados)
$query = "SELECT DISTINCT id, name, description FROM modules WHERE status = 'active' ORDER BY `order`";
$modules = db_fetch_all($query);

// Obtener permisos actuales del usuario
$userPermissions = $permissionModel->getUserPermissions($id);

// Procesar formulario
$message = '';
$success_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Verificar si se está aplicando un perfil predeterminado
    if (isset($_POST['apply_profile'])) {
        $profile = $_POST['profile_type'];
        $newPermissions = [];
        
        foreach ($modules as $module) {
            $moduleId = $module['id'];
            
            if ($profile === 'supervisor') {
                // Supervisor tiene todos los permisos
                $permissions = [
                    'view' => true,
                    'create' => true,
                    'edit' => true,
                    'delete' => true
                ];
                $newPermissions[$moduleId] = $permissions;
            } 
            elseif ($profile === 'staff') {
                // Staff solo tiene permiso de ver, excepto en alertas que tiene todos
                $isAlertModule = (strtolower($module['name']) === 'alertas');
                
                $permissions = [
                    'view' => true,
                    'create' => $isAlertModule,
                    'edit' => $isAlertModule,
                    'delete' => $isAlertModule
                ];
                $newPermissions[$moduleId] = $permissions;
            }
        }
        
        // Guardar permisos del perfil
        $result = $permissionModel->assignPermissions($id, $newPermissions);
        
        if ($result) {
            // Registrar en el log
            log_system_message(
                'info',
                'Usuario ' . $_SESSION['username'] . ' ha aplicado el perfil ' . ucfirst($profile) . ' al usuario ' . $user['username'],
                null,
                'ID: ' . $id
            );
            
            $success_message = 'Perfil ' . ucfirst($profile) . ' aplicado correctamente. Permisos actualizados.';
            
            // Actualizar los permisos actuales del usuario para mostrarlos en el formulario
            $userPermissions = $permissionModel->getUserPermissions($id);
        } else {
            $message = 'Error al aplicar el perfil de permisos.';
        }
    } 
    else {
        // Procesamiento normal del formulario de permisos
        $newPermissions = [];
        
        foreach ($modules as $module) {
            $moduleId = $module['id'];
            $permissions = [
                'view' => isset($_POST['view_' . $moduleId]),
                'create' => isset($_POST['create_' . $moduleId]),
                'edit' => isset($_POST['edit_' . $moduleId]),
                'delete' => isset($_POST['delete_' . $moduleId])
            ];
            
            // Solo incluir módulos con al menos un permiso
            if (in_array(true, $permissions)) {
                $newPermissions[$moduleId] = $permissions;
            }
        }
        
        // Guardar permisos
        $result = $permissionModel->assignPermissions($id, $newPermissions);
        
        if ($result) {
            // Registrar en el log
            log_system_message(
                'info',
                'Usuario ' . $_SESSION['username'] . ' ha actualizado los permisos del usuario ' . $user['username'],
                null,
                'ID: ' . $id
            );
            
            $_SESSION['success_message'] = 'Permisos actualizados correctamente.';
            header('Location: ' . BASE_URL . '/pages/users/index.php');
            exit;
        } else {
            $message = 'Error al actualizar los permisos.';
        }
    }
}

// Definir título de la página
$pageTitle = 'Gestión de Permisos - ' . $user['name'];

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => BASE_URL . '/index.php',
    'Usuarios' => BASE_URL . '/pages/users/index.php',
    'Permisos' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Iniciar buffer de salida para el contenido específico
ob_start();
?>

<?php if (!empty($message)): ?>
    <div class="alert alert-danger">
        <?php echo $message; ?>
    </div>
<?php endif; ?>

<?php if (!empty($success_message)): ?>
    <div class="alert alert-success">
        <?php echo $success_message; ?>
    </div>
<?php endif; ?>

<div class="row mb-3">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Perfiles Predeterminados</h3>
            </div>
            <div class="card-body">
                <form method="POST" action="">
                    <div class="row">
                        <div class="col-md-6">
                            <div class="form-group">
                                <label>Seleccionar Perfil</label>
                                <select class="form-control" name="profile_type">
                                    <option value="supervisor">Supervisor (Permisos Completos)</option>
                                    <option value="staff">Staff (Solo Ver, excepto Alertas)</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-6 d-flex align-items-end">
                            <div class="form-group">
                                <button type="submit" name="apply_profile" value="1" class="btn btn-warning">
                                    <i class="fas fa-magic"></i> Aplicar Perfil Predeterminado
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="callout callout-info">
                        <p><strong>Supervisor:</strong> Tendrá acceso completo a todos los módulos del sistema.</p>
                        <p><strong>Staff:</strong> Solo podrá ver todos los módulos, pero tendrá control completo del módulo de Alertas.</p>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>

<div class="card">
    <div class="card-header">
        <h3 class="card-title">Permisos para: <?php echo htmlspecialchars($user['name']); ?> (<?php echo ucfirst($user['role']); ?>)</h3>
    </div>
    <form action="<?php echo $_SERVER['PHP_SELF'] . '?id=' . $id; ?>" method="post">
        <div class="card-body">
            <p>Seleccione los permisos que desea asignar al usuario:</p>
            
            <table class="table table-bordered">
                <thead>
                    <tr>
                        <th>Módulo</th>
                        <th class="text-center">Ver</th>
                        <th class="text-center">Crear</th>
                        <th class="text-center">Editar</th>
                        <th class="text-center">Eliminar</th>
                    </tr>
                </thead>
                <tbody>
                    <?php foreach ($modules as $module): ?>
                        <?php 
                        $currentPermissions = $userPermissions[$module['id']] ?? [
                            'can_view' => 0,
                            'can_create' => 0,
                            'can_edit' => 0,
                            'can_delete' => 0
                        ];
                        ?>
                        <tr>
                            <td><?php echo htmlspecialchars($module['name']); ?></td>
                            <td class="text-center">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" 
                                        id="view_<?php echo $module['id']; ?>" 
                                        name="view_<?php echo $module['id']; ?>"
                                        <?php echo ($currentPermissions['can_view'] == 1) ? 'checked' : ''; ?>>
                                    <label class="custom-control-label" for="view_<?php echo $module['id']; ?>"></label>
                                </div>
                            </td>
                            <td class="text-center">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" 
                                        id="create_<?php echo $module['id']; ?>" 
                                        name="create_<?php echo $module['id']; ?>"
                                        <?php echo ($currentPermissions['can_create'] == 1) ? 'checked' : ''; ?>>
                                    <label class="custom-control-label" for="create_<?php echo $module['id']; ?>"></label>
                                </div>
                            </td>
                            <td class="text-center">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" 
                                        id="edit_<?php echo $module['id']; ?>" 
                                        name="edit_<?php echo $module['id']; ?>"
                                        <?php echo ($currentPermissions['can_edit'] == 1) ? 'checked' : ''; ?>>
                                    <label class="custom-control-label" for="edit_<?php echo $module['id']; ?>"></label>
                                </div>
                            </td>
                            <td class="text-center">
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" 
                                        id="delete_<?php echo $module['id']; ?>" 
                                        name="delete_<?php echo $module['id']; ?>"
                                        <?php echo ($currentPermissions['can_delete'] == 1) ? 'checked' : ''; ?>>
                                    <label class="custom-control-label" for="delete_<?php echo $module['id']; ?>"></label>
                                </div>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                </tbody>
            </table>
        </div>
        <div class="card-footer">
            <button type="submit" class="btn btn-primary">Guardar Permisos</button>
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

<script>
$(function () {
    // Script para actualizar checkboxes relacionados
    $('input[id^="view_"]').change(function() {
        var moduleId = this.id.split('_')[1];
        if (!this.checked) {
            $('#create_' + moduleId).prop('checked', false);
            $('#edit_' + moduleId).prop('checked', false);
            $('#delete_' + moduleId).prop('checked', false);
        }
    });

    $('input[id^="create_"],input[id^="edit_"],input[id^="delete_"]').change(function() {
        var moduleId = this.id.split('_')[1];
        if (this.checked) {
            $('#view_' + moduleId).prop('checked', true);
        }
    });
});
</script>

<?php require_once '../../includes/footer.php'; ?>