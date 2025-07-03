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

// Definir título de la página
$pageTitle = 'Gestión de Máquinas';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Máquinas' => ''
];

// Obtener filtros
$status = isset($_GET['status']) ? $_GET['status'] : '';
$type = isset($_GET['type']) ? $_GET['type'] : '';
$search = isset($_GET['search']) ? trim($_GET['search']) : '';

// Construir la consulta SQL base
$sql = "SELECT m.*, 
               (SELECT COUNT(*) FROM operator_machine om WHERE om.machine_id = m.id AND om.is_current = 1) as assigned,
               (SELECT o.name FROM operator_machine om JOIN operators o ON om.operator_id = o.id WHERE om.machine_id = m.id AND om.is_current = 1 LIMIT 1) as current_operator
        FROM machines m
        WHERE 1=1";
$params = [];

// Aplicar filtros
if ($status !== '') {
    $sql .= " AND m.status = ?";
    $params[] = $status;
}

if ($type !== '') {
    $sql .= " AND m.type = ?";
    $params[] = $type;
}

if ($search !== '') {
    $sql .= " AND (m.name LIKE ? OR m.id LIKE ? OR m.location LIKE ?)";
    $searchTerm = "%$search%";
    $params[] = $searchTerm;
    $params[] = $searchTerm;
    $params[] = $searchTerm;
}

// Ordenar los resultados
$sql .= " ORDER BY m.name ASC";

// Ejecutar la consulta
$machines = db_fetch_all($sql, $params);

// Obtener tipos de máquinas únicos para el filtro
$machineTypes = db_fetch_all(
    "SELECT DISTINCT type FROM machines ORDER BY type"
);

// Mensajes de sesión
$successMessage = $_SESSION['success_message'] ?? '';
$errorMessage = $_SESSION['error_message'] ?? '';

// Limpiar mensajes de sesión después de usarlos
unset($_SESSION['success_message'], $_SESSION['error_message']);

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Mensajes de error o éxito -->
<?php if (!empty($errorMessage)): ?>
    <div class="alert alert-danger">
        <h5><i class="icon fas fa-ban"></i> Error:</h5>
        <?php echo $errorMessage; ?>
    </div>
<?php endif; ?>

<?php if (!empty($successMessage)): ?>
    <div class="alert alert-success">
        <h5><i class="icon fas fa-check"></i> Éxito:</h5>
        <?php echo $successMessage; ?>
    </div>
<?php endif; ?>

<!-- Filtros y búsqueda -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Filtros</h3>
    </div>
    <div class="card-body">
        <form action="index.php" method="GET" class="form-horizontal">
            <div class="row">
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Estado:</label>
                        <select name="status" class="form-control">
                            <option value="">Todos</option>
                            <option value="active" <?php echo $status === 'active' ? 'selected' : ''; ?>>Activas</option>
                            <option value="maintenance" <?php echo $status === 'maintenance' ? 'selected' : ''; ?>>En Mantenimiento</option>
                            <option value="inactive" <?php echo $status === 'inactive' ? 'selected' : ''; ?>>Inactivas</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Tipo:</label>
                        <select name="type" class="form-control">
                            <option value="">Todos</option>
                            <?php foreach ($machineTypes as $machineType): ?>
                                <option value="<?php echo htmlspecialchars($machineType['type']); ?>" <?php echo $type === $machineType['type'] ? 'selected' : ''; ?>>
                                    <?php echo htmlspecialchars($machineType['type']); ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="form-group">
                        <label>Buscar:</label>
                        <input type="text" name="search" class="form-control" placeholder="Nombre, ID o Ubicación" value="<?php echo htmlspecialchars($search); ?>">
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="form-group">
                        <label>&nbsp;</label>
                        <div>
                            <button type="submit" class="btn btn-primary btn-block">
                                <i class="fas fa-search"></i> Filtrar
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </form>
    </div>
</div>

<!-- Listado de máquinas -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Máquinas Registradas</h3>
        <div class="card-tools">
            <a href="create.php" class="btn btn-primary btn-sm">
                <i class="fas fa-plus"></i> Nueva Máquina
            </a>
        </div>
    </div>
    <div class="card-body table-responsive p-0">
        <table class="table table-hover text-nowrap">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Nombre</th>
                    <th>Tipo</th>
                    <th>Ubicación</th>
                    <th>Estado</th>
                    <th>Operador Actual</th>
                    <th>Último Mantenimiento</th>
                    <th>Acciones</th>
                </tr>
            </thead>
            <tbody>
                <?php if (empty($machines)): ?>
                <tr>
                    <td colspan="8" class="text-center">No se encontraron máquinas</td>
                </tr>
                <?php else: ?>
                    <?php foreach ($machines as $machine): ?>
                    <tr>
                        <td><?php echo htmlspecialchars($machine['id']); ?></td>
                        <td><?php echo htmlspecialchars($machine['name']); ?></td>
                        <td><?php echo htmlspecialchars($machine['type']); ?></td>
                        <td><?php echo htmlspecialchars($machine['location'] ?? 'No especificada'); ?></td>
                        <td>
                            <?php 
                            switch($machine['status']) {
                                case 'active':
                                    echo '<span class="badge badge-success">Activa</span>';
                                    break;
                                case 'maintenance':
                                    echo '<span class="badge badge-warning">Mantenimiento</span>';
                                    break;
                                case 'inactive':
                                    echo '<span class="badge badge-danger">Inactiva</span>';
                                    break;
                                default:
                                    echo '<span class="badge badge-secondary">Desconocido</span>';
                            }
                            ?>
                        </td>
                        <td>
                            <?php if ($machine['assigned'] > 0): ?>
                                <span class="badge badge-info"><?php echo htmlspecialchars($machine['current_operator']); ?></span>
                            <?php else: ?>
                                <span class="badge badge-secondary">Sin asignar</span>
                            <?php endif; ?>
                        </td>
                        <td>
                            <?php if ($machine['last_maintenance']): ?>
                                <?php echo date('d/m/Y', strtotime($machine['last_maintenance'])); ?>
                            <?php else: ?>
                                <span class="text-muted">Nunca</span>
                            <?php endif; ?>
                        </td>
                        <td>
                            <div class="btn-group">
                                <a href="view.php?id=<?php echo $machine['id']; ?>" class="btn btn-info btn-sm">
                                    <i class="fas fa-eye"></i>
                                </a>
                                <a href="edit.php?id=<?php echo $machine['id']; ?>" class="btn btn-warning btn-sm">
                                    <i class="fas fa-edit"></i>
                                </a>
                                <?php if ($machine['status'] == 'active' && $machine['assigned'] == 0): ?>
                                <a href="maintenance.php?id=<?php echo $machine['id']; ?>" class="btn btn-primary btn-sm">
                                    <i class="fas fa-tools"></i>
                                </a>
                                <?php endif; ?>
                                <button type="button" class="btn btn-danger btn-sm" data-toggle="modal" data-target="#deleteModal<?php echo $machine['id']; ?>">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                            
                            <!-- Modal de confirmación para eliminar -->
                            <div class="modal fade" id="deleteModal<?php echo $machine['id']; ?>" tabindex="-1" role="dialog" aria-labelledby="deleteModalLabel" aria-hidden="true">
                                <div class="modal-dialog" role="document">
                                    <div class="modal-content">
                                        <div class="modal-header">
                                            <h5 class="modal-title" id="deleteModalLabel">Confirmar Eliminación</h5>
                                            <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                                <span aria-hidden="true">&times;</span>
                                            </button>
                                        </div>
                                        <div class="modal-body">
                                            ¿Está seguro que desea eliminar la máquina <strong><?php echo htmlspecialchars($machine['name']); ?></strong>?
                                            <?php if ($machine['assigned'] > 0): ?>
                                            <div class="alert alert-warning mt-3">
                                                <i class="fas fa-exclamation-triangle"></i> Esta máquina está asignada a un operador. Al eliminarla, se eliminará también la asignación.
                                            </div>
                                            <?php endif; ?>
                                        </div>
                                        <div class="modal-footer">
                                            <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                                            <a href="delete.php?id=<?php echo $machine['id']; ?>" class="btn btn-danger">Eliminar</a>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </td>
                    </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="create.php" class="btn btn-primary">
    <i class="fas fa-plus"></i> Nueva Máquina
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>