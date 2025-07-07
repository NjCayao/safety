<?php
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

// Definir el título de la página
$pageTitle = "Gestión de Dispositivos";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener lista de dispositivos
$devices = db_fetch_all("
    SELECT d.*, m.name as machine_name
    FROM devices d
    LEFT JOIN machines m ON d.machine_id = m.id
    ORDER BY d.created_at DESC
");

// Función para mostrar el estado con color
function getStatusBadge($status) {
    $badges = [
        'online' => '<span class="badge badge-success">En línea</span>',
        'offline' => '<span class="badge badge-danger">Fuera de línea</span>',
        'syncing' => '<span class="badge badge-warning">Sincronizando</span>',
        'error' => '<span class="badge badge-danger">Error</span>'
    ];
    return $badges[$status] ?? '<span class="badge badge-secondary">Desconocido</span>';
}

// Función para calcular tiempo desde última conexión
function getLastSeenText($lastAccess) {
    if (!$lastAccess) return 'Nunca';
    
    $now = new DateTime();
    $last = new DateTime($lastAccess);
    $diff = $now->diff($last);
    
    if ($diff->days > 0) {
        return "Hace {$diff->days} días";
    } elseif ($diff->h > 0) {
        return "Hace {$diff->h} horas";
    } else {
        return "Hace {$diff->i} minutos";
    }
}
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Gestión de Dispositivos</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item active">Dispositivos</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            <?php if (isset($_GET['msg'])): ?>
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    <?php echo htmlspecialchars($_GET['msg']); ?>
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            <?php endif; ?>
            
            <?php if (isset($_GET['error'])): ?>
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <?php echo htmlspecialchars($_GET['error']); ?>
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            <?php endif; ?>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Dispositivos Registrados</h3>
                    <div class="card-tools">
                        <a href="create.php" class="btn btn-primary btn-sm">
                            <i class="fas fa-plus"></i> Registrar Dispositivo
                        </a>
                    </div>
                </div>
                
                <div class="card-body">
                    <div class="table-responsive">
                        <table class="table table-bordered table-striped" id="devicesTable">
                            <thead>
                                <tr>
                                    <th>ID Dispositivo</th>
                                    <th>Nombre</th>
                                    <th>Tipo</th>
                                    <th>Máquina Asignada</th>
                                    <th>Estado</th>
                                    <th>Última Conexión</th>
                                    <th>IP</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($devices as $device): ?>
                                <tr>
                                    <td><?php echo htmlspecialchars($device['device_id']); ?></td>
                                    <td><?php echo htmlspecialchars($device['name'] ?? 'Sin nombre'); ?></td>
                                    <td><?php echo htmlspecialchars($device['device_type']); ?></td>
                                    <td><?php echo htmlspecialchars($device['machine_name'] ?? 'No asignada'); ?></td>
                                    <td><?php echo getStatusBadge($device['status']); ?></td>
                                    <td>
                                        <?php echo getLastSeenText($device['last_access']); ?>
                                        <?php if ($device['last_access']): ?>
                                            <br><small class="text-muted"><?php echo date('d/m/Y H:i', strtotime($device['last_access'])); ?></small>
                                        <?php endif; ?>
                                    </td>
                                    <td><?php echo htmlspecialchars($device['ip_address'] ?? 'N/A'); ?></td>
                                    <td>
                                        <a href="view.php?id=<?php echo $device['id']; ?>" class="btn btn-info btn-sm" title="Ver detalles">
                                            <i class="fas fa-eye"></i>
                                        </a>
                                        <a href="edit.php?id=<?php echo $device['id']; ?>" class="btn btn-warning btn-sm" title="Editar">
                                            <i class="fas fa-edit"></i>
                                        </a>
                                        <button type="button" class="btn btn-danger btn-sm" onclick="confirmDelete(<?php echo $device['id']; ?>)" title="Eliminar">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    </td>
                                </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </section>
</div>

<!-- Modal de confirmación -->
<div class="modal fade" id="deleteModal" tabindex="-1" role="dialog">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Confirmar eliminación</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                ¿Está seguro de que desea eliminar este dispositivo?
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <button type="button" class="btn btn-danger" id="confirmDeleteBtn">Eliminar</button>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    $('#devicesTable').DataTable({
        "language": {
            "url": "//cdn.datatables.net/plug-ins/1.11.5/i18n/es-ES.json"
        },
        "order": [[5, "desc"]], // Ordenar por última conexión
        "pageLength": 25
    });
});

let deviceIdToDelete = null;

function confirmDelete(id) {
    deviceIdToDelete = id;
    $('#deleteModal').modal('show');
}

$('#confirmDeleteBtn').click(function() {
    if (deviceIdToDelete) {
        window.location.href = 'delete.php?id=' + deviceIdToDelete;
    }
});
</script>