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

// Obtener información de la máquina
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

// Obtener asignación actual de la máquina
$currentAssignment = db_fetch_one(
    "SELECT om.*, o.name as operator_name, o.position as operator_position 
     FROM operator_machine om 
     JOIN operators o ON om.operator_id = o.id 
     WHERE om.machine_id = ? AND om.is_current = 1",
    [$machineId]
);

// Obtener historial de asignaciones de la máquina
$assignmentHistory = db_fetch_all(
    "SELECT om.*, o.name as operator_name, o.position as operator_position 
     FROM operator_machine om 
     JOIN operators o ON om.operator_id = o.id 
     WHERE om.machine_id = ? AND om.is_current = 0
     ORDER BY om.assigned_date DESC",
    [$machineId]
);

// Obtener historial de mantenimiento (simulado por ahora)
// En una implementación real, esto vendría de una tabla de mantenimientos
$maintenanceHistory = [];

// Obtener alertas recientes asociadas a esta máquina
$recentAlerts = db_fetch_all(
    "SELECT a.*, o.name as operator_name 
     FROM alerts a 
     LEFT JOIN operators o ON a.operator_id = o.id 
     WHERE a.machine_id = ? 
     ORDER BY a.timestamp DESC 
     LIMIT 10",
    [$machineId]
);

// Definir título de la página
$pageTitle = 'Detalles de la Máquina: ' . htmlspecialchars($machine['name']);

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Máquinas' => 'index.php',
    'Detalles' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<div class="row">
    <div class="col-md-4">
        <!-- Tarjeta con información principal de la máquina -->
        <div class="card card-primary card-outline">
            <div class="card-header">
                <h3 class="card-title">Información General</h3>
            </div>
            <div class="card-body box-profile">
                <h3 class="profile-username text-center"><?php echo htmlspecialchars($machine['name']); ?></h3>
                <p class="text-muted text-center"><?php echo htmlspecialchars($machine['type']); ?></p>

                <ul class="list-group list-group-unbordered mb-3">
                    <li class="list-group-item">
                        <b>ID</b> <a class="float-right"><?php echo htmlspecialchars($machine['id']); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>Ubicación</b> <a class="float-right"><?php echo htmlspecialchars($machine['location'] ?? 'No especificada'); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>Estado</b> 
                        <span class="float-right">
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
                        </span>
                    </li>
                    <li class="list-group-item">
                        <b>Último Mantenimiento</b> 
                        <span class="float-right">
                            <?php if ($machine['last_maintenance']): ?>
                                <?php echo date('d/m/Y', strtotime($machine['last_maintenance'])); ?>
                            <?php else: ?>
                                <span class="text-muted">Nunca</span>
                            <?php endif; ?>
                        </span>
                    </li>
                </ul>

                <div class="row">
                    <div class="col-md-6">
                        <a href="edit.php?id=<?php echo $machine['id']; ?>" class="btn btn-warning btn-block">
                            <i class="fas fa-edit"></i> Editar
                        </a>
                    </div>
                    <div class="col-md-6">
                        <button type="button" class="btn btn-danger btn-block" data-toggle="modal" data-target="#deleteModal">
                            <i class="fas fa-trash"></i> Eliminar
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Información del operador actual -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Operador Actual</h3>
            </div>
            <div class="card-body">
                <?php if ($currentAssignment): ?>
                    <div class="text-center mb-3">
                        <a href="../operators/view.php?id=<?php echo $currentAssignment['operator_id']; ?>">
                            <h5><?php echo htmlspecialchars($currentAssignment['operator_name']); ?></h5>
                        </a>
                        <p class="text-muted"><?php echo htmlspecialchars($currentAssignment['operator_position'] ?? 'Sin cargo especificado'); ?></p>
                    </div>
                    <p><strong>Asignado desde:</strong> <?php echo date('d/m/Y', strtotime($currentAssignment['assigned_date'])); ?></p>
                    <div class="text-center">
                        <a href="unassign.php?id=<?php echo $currentAssignment['id']; ?>&machine_id=<?php echo $machine['id']; ?>" class="btn btn-sm btn-danger" onclick="return confirm('¿Está seguro que desea desasignar este operador?')">
                            <i class="fas fa-unlink"></i> Desasignar Operador
                        </a>
                    </div>
                <?php else: ?>
                    <div class="text-center">
                        <p class="text-muted">No hay operador asignado actualmente.</p>
                        <a href="assign_operator.php?id=<?php echo $machine['id']; ?>" class="btn btn-primary">
                            <i class="fas fa-user-plus"></i> Asignar Operador
                        </a>
                    </div>
                <?php endif; ?>
            </div>
        </div>

        <!-- Notas de la máquina -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Notas</h3>
            </div>
            <div class="card-body">
                <?php if (!empty($machine['notes'])): ?>
                    <p><?php echo nl2br(htmlspecialchars($machine['notes'])); ?></p>
                <?php else: ?>
                    <p class="text-muted">No hay notas registradas para esta máquina.</p>
                <?php endif; ?>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header p-2">
                <ul class="nav nav-pills">
                    <li class="nav-item">
                        <a class="nav-link active" href="#alerts" data-toggle="tab">Alertas Recientes</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#history" data-toggle="tab">Historial de Operadores</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#maintenance" data-toggle="tab">Mantenimiento</a>
                    </li>
                </ul>
            </div>
            <div class="card-body">
                <div class="tab-content">
                    <!-- Alertas recientes -->
                    <div class="tab-pane active" id="alerts">
                        <?php if (empty($recentAlerts)): ?>
                            <div class="alert alert-success">
                                <i class="icon fas fa-check"></i> No se han registrado alertas para esta máquina.
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Tipo</th>
                                            <th>Operador</th>
                                            <th>Fecha</th>
                                            <th>Estado</th>
                                            <th>Acciones</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($recentAlerts as $alert): ?>
                                            <tr>
                                                <td>
                                                    <?php 
                                                    $alertClass = 'secondary';
                                                    $alertText = ucfirst($alert['alert_type']);
                                                    
                                                    switch ($alert['alert_type']) {
                                                        case 'fatigue':
                                                            $alertClass = 'danger';
                                                            $alertText = 'Fatiga';
                                                            break;
                                                        case 'distraction':
                                                            $alertClass = 'warning';
                                                            $alertText = 'Distracción';
                                                            break;
                                                        case 'phone':
                                                            $alertClass = 'warning';
                                                            $alertText = 'Teléfono';
                                                            break;
                                                        case 'smoking':
                                                            $alertClass = 'danger';
                                                            $alertText = 'Fumando';
                                                            break;
                                                        case 'yawn':
                                                            $alertClass = 'info';
                                                            $alertText = 'Bostezo';
                                                            break;
                                                    }
                                                    ?>
                                                    <span class="badge badge-<?php echo $alertClass; ?>">
                                                        <?php echo $alertText; ?>
                                                    </span>
                                                </td>
                                                <td>
                                                    <?php if (!empty($alert['operator_id'])): ?>
                                                        <a href="../operators/view.php?id=<?php echo htmlspecialchars($alert['operator_id']); ?>">
                                                            <?php echo htmlspecialchars($alert['operator_name']); ?>
                                                        </a>
                                                    <?php else: ?>
                                                        <span class="text-muted">Desconocido</span>
                                                    <?php endif; ?>
                                                </td>
                                                <td><?php echo date('d/m/Y H:i', strtotime($alert['timestamp'])); ?></td>
                                                <td>
                                                    <?php if ($alert['acknowledged']): ?>
                                                        <span class="badge badge-success">Atendida</span>
                                                    <?php else: ?>
                                                        <span class="badge badge-danger">Pendiente</span>
                                                    <?php endif; ?>
                                                </td>
                                                <td>
                                                    <a href="../alerts/view.php?id=<?php echo $alert['id']; ?>" class="btn btn-sm btn-info">
                                                        <i class="fas fa-eye"></i> Ver
                                                    </a>
                                                </td>
                                            </tr>
                                        <?php endforeach; ?>
                                    </tbody>
                                </table>
                            </div>
                            <a href="../alerts/index.php?machine_id=<?php echo $machine['id']; ?>" class="btn btn-info mt-3">
                                <i class="fas fa-list"></i> Ver Todas las Alertas
                            </a>
                        <?php endif; ?>
                    </div>
                    
                    <!-- Historial de operadores -->
                    <div class="tab-pane" id="history">
                        <?php if (empty($assignmentHistory) && !$currentAssignment): ?>
                            <div class="alert alert-info">
                                <i class="icon fas fa-info"></i> Esta máquina nunca ha sido asignada a ningún operador.
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Operador</th>
                                            <th>Cargo</th>
                                            <th>Fecha Inicio</th>
                                            <th>Fecha Fin</th>
                                            <th>Estado</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php if ($currentAssignment): ?>
                                            <tr>
                                                <td>
                                                    <a href="../operators/view.php?id=<?php echo htmlspecialchars($currentAssignment['operator_id']); ?>">
                                                        <?php echo htmlspecialchars($currentAssignment['operator_name']); ?>
                                                    </a>
                                                </td>
                                                <td><?php echo htmlspecialchars($currentAssignment['operator_position'] ?? 'No especificado'); ?></td>
                                                <td><?php echo date('d/m/Y', strtotime($currentAssignment['assigned_date'])); ?></td>
                                                <td>-</td>
                                                <td><span class="badge badge-success">Actual</span></td>
                                            </tr>
                                        <?php endif; ?>
                                        
                                        <?php foreach ($assignmentHistory as $assignment): ?>
                                            <tr>
                                                <td>
                                                    <a href="../operators/view.php?id=<?php echo htmlspecialchars($assignment['operator_id']); ?>">
                                                        <?php echo htmlspecialchars($assignment['operator_name']); ?>
                                                    </a>
                                                </td>
                                                <td><?php echo htmlspecialchars($assignment['operator_position'] ?? 'No especificado'); ?></td>
                                                <td><?php echo date('d/m/Y', strtotime($assignment['assigned_date'])); ?></td>
                                                <td>
                                                    <?php echo isset($assignment['end_date']) ? date('d/m/Y', strtotime($assignment['end_date'])) : 'No registrado'; ?>
                                                </td>
                                                <td><span class="badge badge-secondary">Anterior</span></td>
                                            </tr>
                                        <?php endforeach; ?>
                                    </tbody>
                                </table>
                            </div>
                        <?php endif; ?>
                    </div>
                    
                    <!-- Mantenimiento -->
                    <div class="tab-pane" id="maintenance">
                        <div class="d-flex justify-content-between align-items-center mb-3">
                            <h4>Historial de Mantenimiento</h4>
                            <a href="maintenance.php?id=<?php echo $machine['id']; ?>" class="btn btn-primary">
                                <i class="fas fa-tools"></i> Registrar Mantenimiento
                            </a>
                        </div>
                        
                        <?php if (empty($maintenanceHistory)): ?>
                            <div class="alert alert-info">
                                <i class="icon fas fa-info"></i> No hay registros de mantenimiento para esta máquina.
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Fecha</th>
                                            <th>Tipo</th>
                                            <th>Realizado por</th>
                                            <th>Descripción</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($maintenanceHistory as $maintenance): ?>
                                            <tr>
                                                <td><?php echo date('d/m/Y', strtotime($maintenance['date'])); ?></td>
                                                <td><?php echo htmlspecialchars($maintenance['type']); ?></td>
                                                <td><?php echo htmlspecialchars($maintenance['performed_by']); ?></td>
                                                <td><?php echo htmlspecialchars($maintenance['description']); ?></td>
                                            </tr>
                                        <?php endforeach; ?>
                                    </tbody>
                                </table>
                            </div>
                        <?php endif; ?>
                        
                        <div class="mt-4">
                            <h5>Programar Próximo Mantenimiento</h5>
                            <form action="schedule_maintenance.php" method="POST">
                                <input type="hidden" name="machine_id" value="<?php echo $machine['id']; ?>">
                                <div class="row">
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>Fecha Programada</label>
                                            <input type="date" class="form-control" name="scheduled_date" required min="<?php echo date('Y-m-d'); ?>">
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>Tipo de Mantenimiento</label>
                                            <select class="form-control" name="maintenance_type" required>
                                                <option value="preventive">Preventivo</option>
                                                <option value="corrective">Correctivo</option>
                                                <option value="predictive">Predictivo</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>&nbsp;</label>
                                            <button type="submit" class="btn btn-success btn-block">
                                                <i class="fas fa-calendar-plus"></i> Programar
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Modal para confirmar eliminación -->
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
                <p>¿Está seguro que desea eliminar la máquina <strong><?php echo htmlspecialchars($machine['name']); ?></strong>?</p>
                
                <?php if ($currentAssignment): ?>
                <div class="alert alert-warning">
                    <i class="icon fas fa-exclamation-triangle"></i> Esta máquina está asignada actualmente a un operador. 
                    Si la elimina, la asignación también será eliminada.
                </div>
                <?php endif; ?>
                
                <?php if (!empty($recentAlerts)): ?>
                <div class="alert alert-warning">
                    <i class="icon fas fa-exclamation-triangle"></i> Esta máquina tiene alertas registradas. 
                    Si la elimina, se perderá la referencia a esta máquina en las alertas.
                </div>
                <?php endif; ?>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <a href="delete.php?id=<?php echo $machine['id']; ?>" class="btn btn-danger">
                    <i class="fas fa-trash"></i> Eliminar de todas formas
                </a>
            </div>
        </div>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<div class="btn-group">
    <a href="index.php" class="btn btn-secondary">
        <i class="fas fa-arrow-left"></i> Volver
    </a>
    <a href="edit.php?id=' . $machine['id'] . '" class="btn btn-warning">
        <i class="fas fa-edit"></i> Editar
    </a>
    <a href="maintenance.php?id=' . $machine['id'] . '" class="btn btn-primary">
        <i class="fas fa-tools"></i> Registrar Mantenimiento
    </a>
</div>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>