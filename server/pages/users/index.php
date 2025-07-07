<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';
require_once '../../includes/auth.php';
require_once '../../models/User.php';

// Iniciar sesión
session_start();

// Verificar autenticación
Auth::requireLogin();

// Verificar si el usuario está autenticado
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Verificar permisos - Solo admin y supervisor pueden ver usuarios
if (!Auth::hasRole(['admin', 'supervisor'])) {
    header('Location: ' . BASE_URL . '/401.php');
    exit;
}

// Instanciar modelo
$userModel = new User();

// Obtener lista de usuarios
$users = $userModel->getAllUsers();

// Definir título de la página
$pageTitle = 'Gestión de Usuarios';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => BASE_URL . '/index.php',
    'Usuarios' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Mensajes de éxito/error -->
<?php 
// Mostrar mensajes de éxito/error
if (isset($_SESSION['success_message'])) {
    echo '<div class="alert alert-success alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-check"></i> Éxito!</h5>
            ' . $_SESSION['success_message'] . '
          </div>';
    unset($_SESSION['success_message']);
}

if (isset($_SESSION['error_message'])) {
    echo '<div class="alert alert-danger alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-ban"></i> Error!</h5>
            ' . $_SESSION['error_message'] . '
          </div>';
    unset($_SESSION['error_message']);
}

if (isset($_SESSION['info_message'])) {
    echo '<div class="alert alert-info alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-info"></i> Información:</h5>
            ' . $_SESSION['info_message'] . '
          </div>';
    unset($_SESSION['info_message']);
}
?>

<div class="card">
    <div class="card-header">
        <h3 class="card-title">Lista de Usuarios</h3>
        <div class="card-tools">
            <?php if (Auth::hasRole('admin')): ?>
            <a href="<?php echo BASE_URL; ?>/pages/users/create.php" class="btn btn-primary btn-sm">
                <i class="fas fa-plus"></i> Nuevo Usuario
            </a>
            <?php endif; ?>
        </div>
    </div>
    <div class="card-body">
        <table class="table table-bordered table-striped" id="usersTable">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Nombre</th>
                    <th>Usuario</th>
                    <th>Email</th>
                    <th>Rol</th>
                    <th>Estado</th>
                    <th>Último Login</th>
                    <th>Acciones</th>
                </tr>
            </thead>
            <tbody>
                <?php foreach ($users as $user): ?>
                <tr>
                    <td><?php echo $user['id']; ?></td>
                    <td><?php echo htmlspecialchars($user['name']); ?></td>
                    <td><?php echo htmlspecialchars($user['username']); ?></td>
                    <td><?php echo htmlspecialchars($user['email']); ?></td>
                    <td>
                        <span class="badge <?php echo ($user['role'] === 'admin') ? 'badge-danger' : (($user['role'] === 'supervisor') ? 'badge-warning' : 'badge-info'); ?>">
                            <?php echo ucfirst($user['role']); ?>
                        </span>
                    </td>
                    <td>
                        <span class="badge <?php echo ($user['status'] === 'active') ? 'badge-success' : 'badge-secondary'; ?>">
                            <?php echo ($user['status'] === 'active') ? 'Activo' : 'Inactivo'; ?>
                        </span>
                    </td>
                    <td><?php echo $user['last_login'] ? date('d/m/Y H:i', strtotime($user['last_login'])) : 'Nunca'; ?></td>
                    <td>
                        <div class="btn-group">
                            <?php if (Auth::hasRole('admin') || ($_SESSION['user_id'] == $user['id'])): ?>
                            <a href="<?php echo BASE_URL; ?>/pages/users/edit.php?id=<?php echo $user['id']; ?>" class="btn btn-info btn-sm">
                                <i class="fas fa-edit"></i>
                            </a>
                            <?php endif; ?>
                            
                            <?php if (Auth::hasRole('admin') && $_SESSION['user_id'] != $user['id']): ?>
                            <a href="<?php echo BASE_URL; ?>/pages/users/permissions.php?id=<?php echo $user['id']; ?>" class="btn btn-warning btn-sm">
                                <i class="fas fa-key"></i>
                            </a>
                            
                            <a href="javascript:void(0)" onclick="confirmDelete(<?php echo $user['id']; ?>, '<?php echo htmlspecialchars($user['name']); ?>')" class="btn btn-danger btn-sm">
                                <i class="fas fa-trash"></i>
                            </a>
                            <?php endif; ?>
                        </div>
                    </td>
                </tr>
                <?php endforeach; ?>
            </tbody>
        </table>
    </div>
</div>

<!-- Modal de confirmación -->
<div class="modal fade" id="deleteModal" tabindex="-1" role="dialog" aria-labelledby="deleteModalLabel" aria-hidden="true">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title" id="deleteModalLabel">Confirmar Eliminación</h5>
                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                ¿Está seguro de que desea eliminar al usuario <span id="userName"></span>?
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <a href="#" id="deleteButton" class="btn btn-danger">Eliminar</a>
            </div>
        </div>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="' . BASE_URL . '/pages/users/create.php" class="btn btn-success">
    <i class="fas fa-plus"></i> Nuevo Usuario
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';
?>

<!-- Agregar scripts específicos para esta página -->
<script>
$(function () {
    $('#usersTable').DataTable({
        "paging": true,
        "lengthChange": true,
        "searching": true,
        "ordering": true,
        "info": true,
        "autoWidth": false,
        "responsive": true,
        "language": {
            "url": "<?php echo ASSETS_URL; ?>/plugins/datatables/Spanish.json"
        }
    });
});

function confirmDelete(id, name) {
    document.getElementById('userName').textContent = name;
    document.getElementById('deleteButton').href = '<?php echo BASE_URL; ?>/pages/users/delete.php?id=' + id;
    $('#deleteModal').modal('show');
}
</script>

<?php require_once '../../includes/footer.php'; ?>