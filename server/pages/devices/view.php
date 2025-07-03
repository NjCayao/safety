<?php
// pages/devices/view.php

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
$pageTitle = "Detalles del Dispositivo";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener ID del dispositivo
$device_id = $_GET['id'] ?? null;

if (!$device_id) {
    header("Location: index.php?error=ID de dispositivo no especificado");
    exit();
}

// Obtener información del dispositivo
$device = db_fetch_one("
    SELECT d.*, m.name as machine_name
    FROM devices d
    LEFT JOIN machines m ON d.machine_id = m.id
    WHERE d.id = ?
", [$device_id]);

if (!$device) {
    header("Location: index.php?error=Dispositivo no encontrado");
    exit();
}

// Obtener últimas alertas del dispositivo
$alerts = db_fetch_all("
    SELECT a.*, o.name as operator_name, 
           CASE 
               WHEN a.alert_type = 'fatigue' THEN 'Fatiga'
               WHEN a.alert_type = 'phone' THEN 'Uso de teléfono'
               WHEN a.alert_type = 'smoking' THEN 'Fumando'
               WHEN a.alert_type = 'yawn' THEN 'Bostezo'
               WHEN a.alert_type = 'distraction' THEN 'Distracción'
               ELSE a.alert_type
           END as alert_type_label
    FROM alerts a
    LEFT JOIN operators o ON a.operator_id = o.id
    WHERE a.device_id = ?
    ORDER BY a.timestamp DESC
    LIMIT 10
", [$device['device_id']]);

// Obtener últimos eventos del dispositivo
$events = db_fetch_all("
    SELECT e.*, o.name as operator_name
    FROM events e
    LEFT JOIN operators o ON e.operator_id = o.id
    WHERE e.device_id = ?
    ORDER BY e.event_time DESC
    LIMIT 10
", [$device['device_id']]);

// Verificar si es un dispositivo recién creado
$isNew = isset($_GET['new']) && $_GET['new'] == '1';
$newApiKey = null;

if ($isNew && isset($_SESSION['new_device_api_key']) && isset($_SESSION['new_device_id'])) {
    if ($_SESSION['new_device_id'] === $device['device_id']) {
        $newApiKey = $_SESSION['new_device_api_key'];
        // Limpiar la sesión
        unset($_SESSION['new_device_api_key']);
        unset($_SESSION['new_device_id']);
    }
}

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
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Detalles del Dispositivo</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Detalles</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            <?php if ($newApiKey): ?>
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    <h4 class="alert-heading">¡Dispositivo registrado exitosamente!</h4>
                    <p>API Key generada (guárdela, no se mostrará nuevamente):</p>
                    <div class="bg-white p-2 rounded">
                        <code style="font-size: 1.1em;"><?php echo htmlspecialchars($newApiKey); ?></code>
                    </div>
                    <hr>
                    <p class="mb-0">Configure esta API Key en el dispositivo cliente para autenticación.</p>
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            <?php endif; ?>

            <div class="row">
                <!-- Información general -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">Información General</h3>
                            <div class="card-tools">
                                <a href="edit.php?id=<?php echo $device['id']; ?>" class="btn btn-primary btn-sm">
                                    <i class="fas fa-edit"></i> Editar
                                </a>
                            </div>
                        </div>
                        <div class="card-body">
                            <dl class="row">
                                <dt class="col-sm-4">ID Dispositivo:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['device_id']); ?></dd>
                                
                                <dt class="col-sm-4">Nombre:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['name'] ?? 'Sin nombre'); ?></dd>
                                
                                <dt class="col-sm-4">Tipo:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['device_type']); ?></dd>
                                
                                <dt class="col-sm-4">Estado:</dt>
                                <dd class="col-sm-8"><?php echo getStatusBadge($device['status']); ?></dd>
                                
                                <dt class="col-sm-4">Máquina Asignada:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['machine_name'] ?? 'No asignada'); ?></dd>
                                
                                <dt class="col-sm-4">Ubicación:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['location'] ?? 'No especificada'); ?></dd>
                                
                                <dt class="col-sm-4">IP Actual:</dt>
                                <dd class="col-sm-8"><?php echo htmlspecialchars($device['ip_address'] ?? 'N/A'); ?></dd>
                            </dl>
                        </div>
                    </div>
                </div>

                <!-- Estado de conexión -->
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">Estado de Conexión</h3>
                        </div>
                        <div class="card-body">
                            <dl class="row">
                                <dt class="col-sm-4">Último Acceso:</dt>
                                <dd class="col-sm-8">
                                    <?php if ($device['last_access']): ?>
                                        <?php echo date('d/m/Y H:i:s', strtotime($device['last_access'])); ?>
                                    <?php else: ?>
                                        Nunca
                                    <?php endif; ?>
                                </dd>
                                
                                <dt class="col-sm-4">Última Sincronización:</dt>
                                <dd class="col-sm-8">
                                    <?php if ($device['last_sync']): ?>
                                        <?php echo date('d/m/Y H:i:s', strtotime($device['last_sync'])); ?>
                                    <?php else: ?>
                                        Nunca
                                    <?php endif; ?>
                                </dd>
                                
                                <dt class="col-sm-4">Fecha de Registro:</dt>
                                <dd class="col-sm-8"><?php echo date('d/m/Y H:i:s', strtotime($device['created_at'])); ?></dd>
                            </dl>
                            
                            <?php if ($device['status'] === 'offline'): ?>
                                <div class="alert alert-warning">
                                    <i class="fas fa-exclamation-triangle"></i>
                                    Este dispositivo está fuera de línea. Verifique la conexión.
                                </div>
                            <?php endif; ?>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Alertas recientes -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Últimas Alertas</h3>
                </div>
                <div class="card-body">
                    <?php if (empty($alerts)): ?>
                        <p class="text-muted">No hay alertas registradas para este dispositivo.</p>
                    <?php else: ?>
                        <div class="table-responsive">
                            <table class="table table-bordered table-striped">
                                <thead>
                                    <tr>
                                        <th>Fecha/Hora</th>
                                        <th>Tipo</th>
                                        <th>Operador</th>
                                        <th>Estado</th>
                                        <th>Acciones</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <?php foreach ($alerts as $alert): ?>
                                    <tr>
                                        <td><?php echo date('d/m/Y H:i', strtotime($alert['timestamp'])); ?></td>
                                        <td>
                                            <span class="badge badge-warning">
                                                <?php echo htmlspecialchars($alert['alert_type_label']); ?>
                                            </span>
                                        </td>
                                        <td><?php echo htmlspecialchars($alert['operator_name'] ?? 'Desconocido'); ?></td>
                                        <td>
                                            <?php if ($alert['acknowledged']): ?>
                                                <span class="badge badge-success">Atendida</span>
                                            <?php else: ?>
                                                <span class="badge badge-danger">Pendiente</span>
                                            <?php endif; ?>
                                        </td>
                                        <td>
                                            <a href="../alerts/view.php?id=<?php echo $alert['id']; ?>" 
                                               class="btn btn-info btn-sm">
                                                <i class="fas fa-eye"></i> Ver
                                            </a>
                                        </td>
                                    </tr>
                                    <?php endforeach; ?>
                                </tbody>
                            </table>
                        </div>
                    <?php endif; ?>
                </div>
            </div>

            <!-- Eventos recientes -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Últimos Eventos</h3>
                </div>
                <div class="card-body">
                    <?php if (empty($events)): ?>
                        <p class="text-muted">No hay eventos registrados para este dispositivo.</p>
                    <?php else: ?>
                        <div class="table-responsive">
                            <table class="table table-bordered table-striped">
                                <thead>
                                    <tr>
                                        <th>Fecha/Hora</th>
                                        <th>Tipo</th>
                                        <th>Operador</th>
                                        <th>Estado Sincronización</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <?php foreach ($events as $event): ?>
                                    <tr>
                                        <td><?php echo date('d/m/Y H:i', strtotime($event['event_time'])); ?></td>
                                        <td>
                                            <span class="badge badge-info">
                                                <?php echo ucfirst(str_replace('_', ' ', $event['event_type'])); ?>
                                            </span>
                                        </td>
                                        <td><?php echo htmlspecialchars($event['operator_name'] ?? 'Desconocido'); ?></td>
                                        <td>
                                            <?php if ($event['is_synced']): ?>
                                                <span class="badge badge-success">Sincronizado</span>
                                            <?php else: ?>
                                                <span class="badge badge-warning">Pendiente</span>
                                            <?php endif; ?>
                                        </td>
                                    </tr>
                                    <?php endforeach; ?>
                                </tbody>
                            </table>
                        </div>
                    <?php endif; ?>
                </div>
            </div>
        </div>
    </section>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Auto-cerrar la alerta de API Key después de 30 segundos
    setTimeout(function() {
        $('.alert').alert('close');
    }, 30000);
});
</script>