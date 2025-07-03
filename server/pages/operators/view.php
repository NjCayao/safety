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

// Verificar si se proporcionó un ID de operador
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$operatorId = $_GET['id'];

// Obtener información del operador
$operator = db_fetch_one(
    "SELECT * FROM operators WHERE id = ?",
    [$operatorId]
);

// Si no se encuentra el operador, redirigir al listado
if (!$operator) {
    $_SESSION['error_message'] = "Operador no encontrado.";
    header('Location: index.php');
    exit;
}

// Obtener asignaciones actuales del operador
$currentAssignments = db_fetch_all(
    "SELECT om.*, m.name as machine_name, m.type as machine_type, m.location as machine_location 
     FROM operator_machine om 
     JOIN machines m ON om.machine_id = m.id 
     WHERE om.operator_id = ? AND om.is_current = 1
     ORDER BY om.assigned_date DESC",
    [$operatorId]
);

// Obtener historial de asignaciones del operador
$assignmentHistory = db_fetch_all(
    "SELECT om.*, m.name as machine_name, m.type as machine_type 
     FROM operator_machine om 
     JOIN machines m ON om.machine_id = m.id 
     WHERE om.operator_id = ? AND om.is_current = 0
     ORDER BY om.assigned_date DESC",
    [$operatorId]
);

// Obtener alertas recientes del operador
$recentAlerts = db_fetch_all(
    "SELECT a.*, m.name as machine_name 
     FROM alerts a 
     LEFT JOIN machines m ON a.machine_id = m.id 
     WHERE a.operator_id = ? 
     ORDER BY a.timestamp DESC 
     LIMIT 10",
    [$operatorId]
);

// Definir título de la página
$pageTitle = 'Detalles del Operador: ' . htmlspecialchars($operator['name']);

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Operadores' => 'index.php',
    'Detalles' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<div class="row">
    <div class="col-md-2">
        <a href="index.php" class="btn btn-secondary btn-block">
        <i class=""></i> Atras</a>
    </div>                    
</div>
<br>

<div class="row">
    <div class="col-md-4">
        <!-- Tarjeta de perfil del operador -->
        <div class="card card-primary card-outline">
            <div class="card-body box-profile">
                <div class="text-center">
                    <?php if (!empty($operator['photo_path'])): ?>
                        <img class="profile-user-img img-fluid img-circle" 
                             src="<?php echo htmlspecialchars($operator['photo_path']); ?>" 
                             alt="Foto del operador">
                    <?php else: ?>
                        <img class="profile-user-img img-fluid img-circle" 
                             src="../../assets/dist/img/user-default.jpg" 
                             alt="Foto del operador">
                    <?php endif; ?>
                </div>

                <h3 class="profile-username text-center"><?php echo htmlspecialchars($operator['name']); ?></h3>
                <p class="text-muted text-center"><?php echo htmlspecialchars($operator['position'] ?? 'Sin cargo asignado'); ?></p>

                <ul class="list-group list-group-unbordered mb-3">
                    <li class="list-group-item">
                        <b>ID</b> <a class="float-right"><?php echo htmlspecialchars($operator['id']); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>DNI/CE</b> <a class="float-right"><?php echo htmlspecialchars($operator['dni_number'] ?? 'No registrado'); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>Estado</b> 
                        <span class="float-right">
                            <?php if ($operator['status'] == 'active'): ?>
                                <span class="badge badge-success">Activo</span>
                            <?php else: ?>
                                <span class="badge badge-danger">Inactivo</span>
                            <?php endif; ?>
                        </span>
                    </li>
                    <li class="list-group-item">
                        <b>Registro</b> <a class="float-right"><?php echo date('d/m/Y', strtotime($operator['registration_date'])); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>Último acceso</b> 
                        <a class="float-right">
                            <?php echo ($operator['last_login']) ? date('d/m/Y H:i', strtotime($operator['last_login'])) : 'Nunca'; ?>
                        </a>
                    </li>
                </ul>

                <div class="row">
                    <div class="col-md-6">
                        <a href="edit.php?id=<?php echo $operator['id']; ?>" class="btn btn-warning btn-block">
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

        <!-- Información de licencia -->
        <div class="card card-primary">
            <div class="card-header">
                <h3 class="card-title">Información de Licencia</h3>
            </div>
            <div class="card-body">
                <?php if (empty($operator['license_number'])): ?>
                    <p class="text-muted">No hay licencia registrada para este operador.</p>
                <?php else: ?>
                    <p><strong>Número de Licencia:</strong> <?php echo htmlspecialchars($operator['license_number']); ?></p>
                    
                    <?php if (!empty($operator['license_expiry'])): ?>
                        <p>
                            <strong>Fecha de Vencimiento:</strong> 
                            <?php echo date('d/m/Y', strtotime($operator['license_expiry'])); ?>
                        </p>
                    <?php endif; ?>
                    
                    <p>
                        <strong>Estado:</strong> 
                        <?php if ($operator['license_status'] == 'active'): ?>
                            <span class="badge badge-success">Activa</span>
                        <?php else: ?>
                            <span class="badge badge-danger">Vencida</span>
                        <?php endif; ?>
                    </p>
                    
                    <?php 
                    // Calcular días restantes o vencidos
                    if (!empty($operator['license_expiry'])) {
                        $today = new DateTime();
                        $expiry = new DateTime($operator['license_expiry']);
                        $diff = $today->diff($expiry);
                        
                        if ($expiry > $today) {
                            echo '<div class="alert alert-info">';
                            echo '<i class="icon fas fa-info"></i> ';
                            echo 'La licencia vence en ' . $diff->days . ' días.';
                            echo '</div>';
                        } else {
                            echo '<div class="alert alert-danger">';
                            echo '<i class="icon fas fa-exclamation-triangle"></i> ';
                            echo 'La licencia está vencida hace ' . $diff->days . ' días.';
                            echo '</div>';
                        }
                    }
                    ?>
                <?php endif; ?>
            </div>
        </div>

        <!-- Notas del operador -->
        <div class="card card-primary">
            <div class="card-header">
                <h3 class="card-title">Notas</h3>
            </div>
            <div class="card-body">
                <?php if (!empty($operator['notes'])): ?>
                    <p><?php echo nl2br(htmlspecialchars($operator['notes'])); ?></p>
                <?php else: ?>
                    <p class="text-muted">No hay notas registradas para este operador.</p>
                <?php endif; ?>
            </div>
        </div>
    </div>
    
    <div class="col-md-8">
        <div class="card">
            <div class="card-header p-2">
                <ul class="nav nav-pills">
                    <li class="nav-item">
                        <a class="nav-link active" href="#assignments" data-toggle="tab">Asignaciones Actuales</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#history" data-toggle="tab">Historial</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#alerts" data-toggle="tab">Alertas Recientes</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="#photos" data-toggle="tab">Fotos Reconocimiento</a>
                    </li>
                </ul>
            </div>
            <div class="card-body">
                <div class="tab-content">
                    <!-- Asignaciones actuales -->
                    <div class="tab-pane active" id="assignments">
                        <?php if (empty($currentAssignments)): ?>
                            <div class="alert alert-info">
                                <i class="icon fas fa-info"></i> Este operador no tiene máquinas asignadas actualmente.
                            </div>
                            <a href="assign.php?id=<?php echo $operator['id']; ?>" class="btn btn-primary">
                                <i class="fas fa-link"></i> Asignar a Máquina
                            </a>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Máquina</th>
                                            <th>Tipo</th>
                                            <th>Ubicación</th>
                                            <th>Fecha Asignación</th>
                                            <th>Acciones</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($currentAssignments as $assignment): ?>
                                            <tr>
                                                <td>
                                                    <a href="../machines/view.php?id=<?php echo htmlspecialchars($assignment['machine_id']); ?>">
                                                        <?php echo htmlspecialchars($assignment['machine_name']); ?>
                                                    </a>
                                                </td>
                                                <td><?php echo htmlspecialchars($assignment['machine_type']); ?></td>
                                                <td><?php echo htmlspecialchars($assignment['machine_location']); ?></td>
                                                <td><?php echo date('d/m/Y', strtotime($assignment['assigned_date'])); ?></td>
                                                <td>
                                                    <a href="../machines/unassign.php?id=<?php echo $assignment['id']; ?>&operator_id=<?php echo $operator['id']; ?>" class="btn btn-sm btn-danger">
                                                        <i class="fas fa-unlink"></i> Desasignar
                                                    </a>
                                                </td>
                                            </tr>
                                        <?php endforeach; ?>
                                    </tbody>
                                </table>
                            </div>
                            <a href="assign.php?id=<?php echo $operator['id']; ?>" class="btn btn-primary mt-3">
                                <i class="fas fa-link"></i> Asignar a Otra Máquina
                            </a>
                        <?php endif; ?>
                    </div>
                    
                    <!-- Historial de asignaciones -->
                    <div class="tab-pane" id="history">
                        <?php if (empty($assignmentHistory)): ?>
                            <div class="alert alert-info">
                                <i class="icon fas fa-info"></i> No hay historial de asignaciones anteriores.
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Máquina</th>
                                            <th>Tipo</th>
                                            <th>Fecha Asignación</th>
                                            <th>Fecha Fin</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($assignmentHistory as $assignment): ?>
                                            <tr>
                                                <td>
                                                    <a href="../machines/view.php?id=<?php echo htmlspecialchars($assignment['machine_id']); ?>">
                                                        <?php echo htmlspecialchars($assignment['machine_name']); ?>
                                                    </a>
                                                </td>
                                                <td><?php echo htmlspecialchars($assignment['machine_type']); ?></td>
                                                <td><?php echo date('d/m/Y', strtotime($assignment['assigned_date'])); ?></td>
                                                <td><?php echo isset($assignment['end_date']) ? date('d/m/Y', strtotime($assignment['end_date'])) : 'N/A'; ?></td>
                                            </tr>
                                        <?php endforeach; ?>
                                    </tbody>
                                </table>
                            </div>
                        <?php endif; ?>
                    </div>
                    
                    <!-- Alertas recientes -->
                    <div class="tab-pane" id="alerts">
                        <?php if (empty($recentAlerts)): ?>
                            <div class="alert alert-success">
                                <i class="icon fas fa-check"></i> No se han registrado alertas para este operador.
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-striped">
                                    <thead>
                                        <tr>
                                            <th>Tipo</th>
                                            <th>Máquina</th>
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
                                                    <?php if (!empty($alert['machine_id'])): ?>
                                                        <a href="../machines/view.php?id=<?php echo htmlspecialchars($alert['machine_id']); ?>">
                                                            <?php echo htmlspecialchars($alert['machine_name']); ?>
                                                        </a>
                                                    <?php else: ?>
                                                        <span class="text-muted">No especificada</span>
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
                            <a href="../alerts/index.php?operator_id=<?php echo $operator['id']; ?>" class="btn btn-info mt-3">
                                <i class="fas fa-list"></i> Ver Todas las Alertas
                            </a>
                        <?php endif; ?>
                    </div>
                    
                    <!-- Fotos de reconocimiento facial -->
                    <div class="tab-pane" id="photos">
                        <h4 class="mb-3">Fotografías para Reconocimiento Facial</h4>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h3 class="card-title">Foto de Perfil</h3>
                                    </div>
                                    <div class="card-body text-center">
                                        <?php if (!empty($operator['photo_path'])): ?>
                                            <img src="../../<?php echo htmlspecialchars($operator['photo_path']); ?>" 
                                                 alt="Foto de perfil" 
                                                 class="img-fluid" 
                                                 style="max-height: 200px;">
                                        <?php else: ?>
                                            <div class="alert alert-warning">
                                                <i class="icon fas fa-exclamation-triangle"></i> No hay foto de perfil registrada.
                                            </div>
                                        <?php endif; ?>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="card">
                                    <div class="card-header">
                                        <h3 class="card-title">Fotos Adicionales</h3>
                                    </div>
                                    <div class="card-body">
                                        <div class="row">
                                            <?php
                                            $hasFacialPhotos = false;
                                            $facialPhotos = ['face_photo1', 'face_photo2', 'face_photo3'];
                                            
                                            foreach ($facialPhotos as $index => $photoField):
                                                if (!empty($operator[$photoField])):
                                                    $hasFacialPhotos = true;
                                            ?>
                                                <div class="col-md-4 text-center mb-3">
                                                    <img src="../../<?php echo htmlspecialchars($operator[$photoField]); ?>" 
                                                         alt="Foto facial <?php echo $index + 1; ?>" 
                                                         class="img-fluid" 
                                                         style="max-height: 100px;">
                                                    <p class="mt-2">Foto <?php echo $index + 1; ?></p>
                                                </div>
                                            <?php
                                                endif;
                                            endforeach;
                                            
                                            if (!$hasFacialPhotos):
                                            ?>
                                                <div class="col-12">
                                                    <div class="alert alert-warning">
                                                        <i class="icon fas fa-exclamation-triangle"></i> No hay fotos adicionales registradas para reconocimiento facial.
                                                    </div>
                                                </div>
                                            <?php endif; ?>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-3">
                            <a href="edit.php?id=<?php echo $operator['id']; ?>#photos" class="btn btn-primary">
                                <i class="fas fa-camera"></i> Gestionar Fotos
                            </a>
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
                <p>¿Está seguro que desea eliminar al operador <strong><?php echo htmlspecialchars($operator['name']); ?></strong>?</p>
                
                <?php if (!empty($currentAssignments)): ?>
                <div class="alert alert-warning">
                    <i class="icon fas fa-exclamation-triangle"></i> Este operador tiene máquinas asignadas actualmente. 
                    Si lo elimina, todas las asignaciones también serán eliminadas.
                </div>
                <?php endif; ?>
                
                <?php if (!empty($recentAlerts)): ?>
                <div class="alert alert-warning">
                    <i class="icon fas fa-exclamation-triangle"></i> Este operador tiene alertas registradas. 
                    Si lo elimina, se perderá la referencia a este operador en las alertas.
                </div>
                <?php endif; ?>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <a href="delete.php?id=<?php echo $operator['id']; ?>" class="btn btn-danger">
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
    <a href="edit.php?id=' . $operator['id'] . '" class="btn btn-warning">
        <i class="fas fa-edit"></i> Editar
    </a>
    <a href="assign.php?id=' . $operator['id'] . '" class="btn btn-primary">
        <i class="fas fa-link"></i> Asignar Máquina
    </a>
</div>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>