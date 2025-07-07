<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Incluir necesarios
require_once '../../config/config.php';
require_once '../../config/database.php'; // Añadido para acceso a la base de datos
require_once '../../models/Alert.php';
require_once '../../models/Operator.php';
require_once '../../models/Machine.php';

// 🆕 NUEVO: Incluir sistema de configuración de dispositivos
require_once '../../includes/device_config.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Inicializar modelos
$alertModel = new Alert();
$operatorModel = new Operator();
$machineModel = new Machine();

// Obtener estadísticas generales
$today = date('Y-m-d');
$weekStart = date('Y-m-d', strtotime('monday this week'));
$monthStart = date('Y-m-01');
$alertsToday = $alertModel->countAlerts(['date_from' => $today]);
$alertsThisWeek = $alertModel->countAlerts(['date_from' => $weekStart]);
$alertsThisMonth = $alertModel->countAlerts(['date_from' => $monthStart]);
$alertsPending = $alertModel->countAlerts(['acknowledged' => 0]);

// Obtener últimas alertas
$recentAlerts = $alertModel->getAlerts([], 10, 0);

// 🆕 NUEVO: Obtener estadísticas de dispositivos para el dashboard
$device_stats = [];
try {
    $device_stats = db_fetch_all("
        SELECT 
            COUNT(*) as total_devices,
            SUM(CASE WHEN status = 'online' THEN 1 ELSE 0 END) as online_devices,
            SUM(CASE WHEN status = 'offline' THEN 1 ELSE 0 END) as offline_devices,
            SUM(CASE WHEN config_pending = 1 THEN 1 ELSE 0 END) as pending_config,
            SUM(CASE WHEN TIMESTAMPDIFF(MINUTE, last_access, NOW()) > 10 AND status = 'online' THEN 1 ELSE 0 END) as stale_devices
        FROM devices
    ");
    
    if (!empty($device_stats)) {
        $device_stats = $device_stats[0];
    } else {
        $device_stats = [
            'total_devices' => 0,
            'online_devices' => 0,
            'offline_devices' => 0,
            'pending_config' => 0,
            'stale_devices' => 0
        ];
    }
} catch (Exception $e) {
    // Si hay error (tabla no existe), usar valores por defecto
    $device_stats = [
        'total_devices' => 0,
        'online_devices' => 0,
        'offline_devices' => 0,
        'pending_config' => 0,
        'stale_devices' => 0
    ];
}

// 🆕 NUEVO: Dispositivos con alertas recientes
$recent_device_alerts = [];
try {
    $recent_device_alerts = db_fetch_all("
        SELECT d.device_id, d.name as device_name, d.status, d.last_access,
               COUNT(a.id) as alert_count
        FROM devices d
        LEFT JOIN alerts a ON d.device_id = a.device_id 
            AND a.timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            AND a.acknowledged = 0
        GROUP BY d.device_id, d.name, d.status, d.last_access
        HAVING alert_count > 0
        ORDER BY alert_count DESC, d.last_access DESC
        LIMIT 5
    ");
} catch (Exception $e) {
    $recent_device_alerts = [];
}

// 🆕 NUEVO: Configuraciones pendientes
$pending_configs = [];
try {
    $pending_configs = db_fetch_all("
        SELECT d.device_id, d.name as device_name, d.config_version, 
               dch.created_at as config_created, dch.changes_summary,
               u.username as changed_by
        FROM devices d
        JOIN device_config_history dch ON d.device_id = dch.device_id
        LEFT JOIN users u ON dch.changed_by = u.id
        WHERE d.config_pending = 1
        AND dch.applied_successfully IS NULL
        ORDER BY dch.created_at DESC
        LIMIT 5
    ");
} catch (Exception $e) {
    $pending_configs = [];
}

// Definir tipos de alerta para mostrar nombre legible
$alertTypes = [
    'fatigue' => 'Fatiga',
    'phone' => 'Uso de celular',
    'smoking' => 'Fumando',
    'unauthorized' => 'No autorizado',
    'yawn' => 'Bostezo',
    'distraction' => 'Distracción',
    'behavior' => 'Comportamiento anómalo',
    'other' => 'Otro'
];

// Obtener total de operadores y máquinas
$totalOperators = $operatorModel->countOperators();
$totalMachines = $machineModel->countMachines();

// Definir título de la página
$pageTitle = 'Dashboard del Sistema';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Estadísticas' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Main content -->
<section class="content">
    <div class="container-fluid">
        
        <!-- 🆕 NUEVO: Widgets de Dispositivos (Primera fila) -->
        <?php if ($device_stats['total_devices'] > 0): ?>
        <div class="row mb-3">
            <div class="col-12">
                <h4 class="mb-3">
                    <i class="fas fa-microchip text-primary"></i>
                    Estado de Dispositivos IoT
                </h4>
            </div>
        </div>
        <div class="row">
            <!-- Total Dispositivos -->
            <div class="col-lg-3 col-6">
                <div class="small-box bg-info">
                    <div class="inner">
                        <h3><?php echo $device_stats['total_devices']; ?></h3>
                        <p>Total Dispositivos</p>
                    </div>
                    <div class="icon">
                        <i class="fas fa-microchip"></i>
                    </div>
                    <a href="<?php echo BASE_URL; ?>/pages/devices/index.php" class="small-box-footer">
                        Ver todos <i class="fas fa-arrow-circle-right"></i>
                    </a>
                </div>
            </div>

            <!-- Dispositivos Online -->
            <div class="col-lg-3 col-6">
                <div class="small-box bg-success">
                    <div class="inner">
                        <h3><?php echo $device_stats['online_devices']; ?></h3>
                        <p>En Línea</p>
                    </div>
                    <div class="icon">
                        <i class="fas fa-wifi"></i>
                    </div>
                    <a href="<?php echo BASE_URL; ?>/pages/devices/index.php?filter=online" class="small-box-footer">
                        Ver detalles <i class="fas fa-arrow-circle-right"></i>
                    </a>
                </div>
            </div>

            <!-- Dispositivos Offline -->
            <div class="col-lg-3 col-6">
                <div class="small-box bg-danger">
                    <div class="inner">
                        <h3><?php echo $device_stats['offline_devices']; ?></h3>
                        <p>Fuera de Línea</p>
                    </div>
                    <div class="icon">
                        <i class="fas fa-power-off"></i>
                    </div>
                    <a href="<?php echo BASE_URL; ?>/pages/devices/index.php?filter=offline" class="small-box-footer">
                        Revisar <i class="fas fa-arrow-circle-right"></i>
                    </a>
                </div>
            </div>

            <!-- Configuraciones Pendientes -->
            <div class="col-lg-3 col-6">
                <div class="small-box bg-warning">
                    <div class="inner">
                        <h3><?php echo $device_stats['pending_config']; ?></h3>
                        <p>Config. Pendientes</p>
                    </div>
                    <div class="icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <a href="<?php echo BASE_URL; ?>/pages/devices/config.php?filter=pending" class="small-box-footer">
                        Gestionar <i class="fas fa-arrow-circle-right"></i>
                    </a>
                </div>
            </div>
        </div>
        
        <!-- Separador -->
        <hr class="my-4">
        <?php endif; ?>

        <!-- Estadísticas de Alertas (Fila original) -->
        <div class="row mb-3">
            <div class="col-12">
                <h4 class="mb-3">
                    <i class="fas fa-exclamation-triangle text-warning"></i>
                    Estadísticas de Alertas
                </h4>
            </div>
        </div>
        <div class="row">
            <div class="col-12 col-sm-6 col-md-3">
                <div class="info-box">
                    <span class="info-box-icon bg-info"><i class="fas fa-bell"></i></span>
                    <div class="info-box-content">
                        <span class="info-box-text">Alertas Hoy</span>
                        <span class="info-box-number"><?php echo $alertsToday; ?></span>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-md-3">
                <div class="info-box">
                    <span class="info-box-icon bg-success"><i class="fas fa-calendar-week"></i></span>
                    <div class="info-box-content">
                        <span class="info-box-text">Alertas Esta Semana</span>
                        <span class="info-box-number"><?php echo $alertsThisWeek; ?></span>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-md-3">
                <div class="info-box">
                    <span class="info-box-icon bg-warning"><i class="fas fa-calendar-alt"></i></span>
                    <div class="info-box-content">
                        <span class="info-box-text">Alertas Este Mes</span>
                        <span class="info-box-number"><?php echo $alertsThisMonth; ?></span>
                    </div>
                </div>
            </div>
            <div class="col-12 col-sm-6 col-md-3">
                <div class="info-box">
                    <span class="info-box-icon bg-danger"><i class="fas fa-exclamation-circle"></i></span>
                    <div class="info-box-content">
                        <span class="info-box-text">Alertas Pendientes</span>
                        <span class="info-box-number"><?php echo $alertsPending; ?></span>
                    </div>
                </div>
            </div>
        </div>

        <!-- 🆕 NUEVO: Widgets de Gestión de Dispositivos -->
        <?php if ($device_stats['total_devices'] > 0): ?>
        <div class="row">
            <!-- Dispositivos con Alertas Recientes -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-exclamation-triangle text-warning"></i>
                            Dispositivos con Alertas (24h)
                        </h3>
                        <div class="card-tools">
                            <a href="<?php echo BASE_URL; ?>/pages/devices/index.php" class="btn btn-tool">
                                <i class="fas fa-expand"></i>
                            </a>
                        </div>
                    </div>
                    <div class="card-body p-0">
                        <?php if (empty($recent_device_alerts)): ?>
                            <div class="p-3 text-center text-muted">
                                <i class="fas fa-check-circle fa-2x mb-2 text-success"></i>
                                <p>No hay dispositivos con alertas recientes</p>
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-sm table-hover">
                                    <thead>
                                        <tr>
                                            <th>Dispositivo</th>
                                            <th>Estado</th>
                                            <th>Alertas</th>
                                            <th>Última Conexión</th>
                                            <th>Acciones</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($recent_device_alerts as $device): ?>
                                        <tr>
                                            <td>
                                                <strong><?php echo htmlspecialchars($device['device_name'] ?: $device['device_id']); ?></strong>
                                                <br><small class="text-muted"><?php echo htmlspecialchars($device['device_id']); ?></small>
                                            </td>
                                            <td>
                                                <?php
                                                $status_class = $device['status'] === 'online' ? 'success' : 'danger';
                                                echo "<span class=\"badge badge-{$status_class}\">" . ucfirst($device['status']) . "</span>";
                                                ?>
                                            </td>
                                            <td>
                                                <span class="badge badge-danger"><?php echo $device['alert_count']; ?></span>
                                            </td>
                                            <td>
                                                <?php if ($device['last_access']): ?>
                                                    <small><?php echo date('d/m H:i', strtotime($device['last_access'])); ?></small>
                                                <?php else: ?>
                                                    <small class="text-muted">Nunca</small>
                                                <?php endif; ?>
                                            </td>
                                            <td>
                                                <a href="<?php echo BASE_URL; ?>/pages/devices/view.php?id=<?php echo $device['device_id']; ?>" 
                                                   class="btn btn-sm btn-info" title="Ver detalles">
                                                    <i class="fas fa-eye"></i>
                                                </a>
                                                <a href="<?php echo BASE_URL; ?>/pages/devices/configure.php?id=<?php echo $device['device_id']; ?>" 
                                                   class="btn btn-sm btn-warning" title="Configurar">
                                                    <i class="fas fa-cog"></i>
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
            </div>

            <!-- Configuraciones Pendientes -->
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-clock text-warning"></i>
                            Configuraciones Pendientes
                        </h3>
                        <div class="card-tools">
                            <a href="<?php echo BASE_URL; ?>/pages/devices/config.php" class="btn btn-tool">
                                <i class="fas fa-expand"></i>
                            </a>
                        </div>
                    </div>
                    <div class="card-body p-0">
                        <?php if (empty($pending_configs)): ?>
                            <div class="p-3 text-center text-muted">
                                <i class="fas fa-check-circle fa-2x mb-2 text-success"></i>
                                <p>Todas las configuraciones están aplicadas</p>
                            </div>
                        <?php else: ?>
                            <div class="table-responsive">
                                <table class="table table-sm table-hover">
                                    <thead>
                                        <tr>
                                            <th>Dispositivo</th>
                                            <th>Cambios</th>
                                            <th>Usuario</th>
                                            <th>Fecha</th>
                                            <th>Acciones</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        <?php foreach ($pending_configs as $config): ?>
                                        <tr>
                                            <td>
                                                <strong><?php echo htmlspecialchars($config['device_name'] ?: $config['device_id']); ?></strong>
                                                <br><small class="text-muted">v<?php echo $config['config_version']; ?></small>
                                            </td>
                                            <td>
                                                <small class="text-muted">
                                                    <?php 
                                                    $summary = $config['changes_summary'];
                                                    echo htmlspecialchars(strlen($summary) > 40 ? substr($summary, 0, 40) . '...' : $summary);
                                                    ?>
                                                </small>
                                            </td>
                                            <td><?php echo htmlspecialchars($config['changed_by'] ?: 'Sistema'); ?></td>
                                            <td>
                                                <small><?php echo date('d/m H:i', strtotime($config['config_created'])); ?></small>
                                            </td>
                                            <td>
                                                <a href="<?php echo BASE_URL; ?>/pages/devices/configure.php?id=<?php echo $config['device_id']; ?>" 
                                                   class="btn btn-sm btn-primary" title="Gestionar">
                                                    <i class="fas fa-cog"></i>
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
            </div>
        </div>

        <!-- Acciones Rápidas de Dispositivos -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-tools"></i>
                            Acciones Rápidas - Gestión de Dispositivos
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <a href="<?php echo BASE_URL; ?>/pages/devices/create.php" class="btn btn-success btn-block btn-action">
                                    <i class="fas fa-plus"></i>
                                    Registrar Dispositivo
                                </a>
                            </div>
                            <div class="col-md-3">
                                <a href="<?php echo BASE_URL; ?>/pages/devices/config.php" class="btn btn-primary btn-block btn-action">
                                    <i class="fas fa-cogs"></i>
                                    Gestionar Configuraciones
                                </a>
                            </div>
                            <div class="col-md-3">
                                <a href="<?php echo BASE_URL; ?>/pages/devices/index.php?filter=offline" class="btn btn-warning btn-block btn-action">
                                    <i class="fas fa-exclamation-triangle"></i>
                                    Revisar Dispositivos Offline
                                </a>
                            </div>
                            <div class="col-md-3">
                                <button type="button" class="btn btn-info btn-block btn-action" onclick="refreshDeviceStatus()">
                                    <i class="fas fa-sync"></i>
                                    Actualizar Estado General
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Separador -->
        <hr class="my-4">
        <?php endif; ?>

        <!-- Gráficos principales (contenido original) -->
        <div class="row mb-3">
            <div class="col-12">
                <h4 class="mb-3">
                    <i class="fas fa-chart-pie text-info"></i>
                    Análisis y Tendencias
                </h4>
            </div>
        </div>
        <div class="row">
            <div class="col-md-6">
                <!-- Gráfico de Alertas por Tipo -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-chart-pie mr-1"></i>
                            Distribución de Alertas por Tipo
                        </h3>
                        <div class="card-tools">
                            <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                <i class="fas fa-minus"></i>
                            </button>
                            <button type="button" class="btn btn-tool" data-card-widget="remove">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsByTypeChart" style="height: 300px;"></canvas>
                    </div>
                </div>
                <!-- Gráfico de Alertas por Operador -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-user mr-1"></i>
                            Top Operadores con Más Alertas
                        </h3>
                        <div class="card-tools">
                            <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                <i class="fas fa-minus"></i>
                            </button>
                            <button type="button" class="btn btn-tool" data-card-widget="remove">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsByOperatorChart" style="height: 300px;"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <!-- Gráfico de Tendencia de Alertas -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-chart-line mr-1"></i>
                            Tendencia de Alertas
                        </h3>
                        <div class="card-tools">
                            <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                <i class="fas fa-minus"></i>
                            </button>
                            <button type="button" class="btn btn-tool" data-card-widget="remove">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsTrendChart" style="height: 300px;"></canvas>
                    </div>
                </div>
                <!-- Estadísticas rápidas -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-info-circle mr-1"></i>
                            Información del Sistema
                        </h3>
                        <div class="card-tools">
                            <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                <i class="fas fa-minus"></i>
                            </button>
                            <button type="button" class="btn btn-tool" data-card-widget="remove">
                                <i class="fas fa-times"></i>
                            </button>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="info-box bg-light">
                                    <div class="info-box-content">
                                        <span class="info-box-text text-center text-muted">Total Operadores</span>
                                        <span class="info-box-number text-center text-muted mb-0"><?php echo $totalOperators; ?></span>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="info-box bg-light">
                                    <div class="info-box-content">
                                        <span class="info-box-text text-center text-muted">Total Máquinas</span>
                                        <span class="info-box-number text-center text-muted mb-0"><?php echo $totalMachines; ?></span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <!-- Estado de los dispositivos -->
                        <h5 class="mt-4 mb-2">Estado de Dispositivos Registrados</h5>
                        <?php
                        // Obtener estado de dispositivos
                        $devicesQuery = "SELECT device_id, name, status, last_access FROM devices ORDER BY name LIMIT 10";
                        $devices = [];
                        
                        // Verificar si existe la tabla devices
                        try {
                            $devices = db_fetch_all($devicesQuery);
                        } catch (Exception $e) {
                            // Si hay error, probablemente la tabla no existe
                            echo '<div class="alert alert-info">No hay información de dispositivos disponible.</div>';
                        }
                        
                        if (empty($devices)) {
                            echo '<div class="alert alert-info">No hay dispositivos registrados.</div>';
                        } else {
                            foreach ($devices as $device) {
                                $statusClass = isset($device['status']) && $device['status'] == 'online' ? 'success' : 'danger';
                                $statusText = isset($device['status']) && $device['status'] == 'online' ? 'En línea' : 'Desconectado';
                                $deviceName = $device['name'] ?: $device['device_id'];
                                
                                // Calcular tiempo desde última conexión
                                $lastSeenText = '';
                                if ($device['last_access']) {
                                    $lastSeen = new DateTime($device['last_access']);
                                    $now = new DateTime();
                                    $diff = $now->diff($lastSeen);
                                    
                                    if ($diff->days > 0) {
                                        $lastSeenText = "hace {$diff->days}d";
                                    } elseif ($diff->h > 0) {
                                        $lastSeenText = "hace {$diff->h}h";
                                    } else {
                                        $lastSeenText = "hace {$diff->i}m";
                                    }
                                } else {
                                    $lastSeenText = 'nunca';
                                }
                        ?>
                                <div class="progress-group">
                                    <?php echo htmlspecialchars($deviceName); ?>
                                    <span class="float-right">
                                        <small class="text-muted mr-2"><?php echo $lastSeenText; ?></small>
                                        <span class="badge bg-<?php echo $statusClass; ?>">
                                            <?php echo $statusText; ?>
                                        </span>
                                    </span>
                                </div>
                        <?php
                            }
                            
                            // Si hay más de 10 dispositivos, mostrar enlace para ver todos
                            $totalDevicesInDb = db_fetch_one("SELECT COUNT(*) as total FROM devices")['total'] ?? 0;
                            if ($totalDevicesInDb > 10) {
                                echo '<div class="text-center mt-3">';
                                echo '<a href="' . BASE_URL . '/pages/devices/index.php" class="btn btn-sm btn-outline-primary">';
                                echo '<i class="fas fa-list"></i> Ver todos los ' . $totalDevicesInDb . ' dispositivos';
                                echo '</a>';
                                echo '</div>';
                            }
                        }
                        ?>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="export.php" class="btn btn-success">
    <i class="fas fa-file-excel"></i> Exportar Estadísticas
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';
?>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Gráfico de alertas por tipo
    const alertTypeCtx = document.getElementById('alertsByTypeChart').getContext('2d');
    const alertTypeData = {
        labels: [],
        datasets: [{
            label: 'Cantidad de Alertas',
            data: [],
            backgroundColor: [
                'rgba(255, 99, 132, 0.6)',
                'rgba(54, 162, 235, 0.6)',
                'rgba(255, 206, 86, 0.6)',
                'rgba(75, 192, 192, 0.6)',
                'rgba(153, 102, 255, 0.6)',
                'rgba(255, 159, 64, 0.6)',
                'rgba(199, 199, 199, 0.6)',
                'rgba(83, 102, 255, 0.6)'
            ],
            borderWidth: 1
        }]
    };

    const alertTypeChart = new Chart(alertTypeCtx, {
        type: 'pie',
        data: alertTypeData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Distribución de Alertas por Tipo'
                }
            }
        }
    });

    // Gráfico de operadores
    const operatorCtx = document.getElementById('alertsByOperatorChart').getContext('2d');
    const operatorData = {
        labels: [],
        datasets: [{
            label: 'Cantidad de Alertas',
            data: [],
            backgroundColor: 'rgba(60, 141, 188, 0.7)',
            borderColor: 'rgba(60, 141, 188, 1)',
            borderWidth: 1
        }]
    };

    const operatorChart = new Chart(operatorCtx, {
        type: 'bar',
        data: operatorData,
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Top Operadores con Más Alertas'
                }
            },
            scales: {
                x: {
                    beginAtZero: true
                }
            }
        }
    });

    // Datos para el gráfico de tendencias
    const trendCtx = document.getElementById('alertsTrendChart').getContext('2d');
    const trendData = {
        labels: [],
        datasets: [{
            label: 'Alertas por Día',
            data: [],
            borderColor: 'rgba(75, 192, 192, 1)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            borderWidth: 2,
            fill: true,
            tension: 0.1
        }]
    };

    const trendChart = new Chart(trendCtx, {
        type: 'line',
        data: trendData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Tendencia de Alertas en los Últimos 30 Días'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Número de Alertas'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Fecha'
                    }
                }
            }
        }
    });

    // Cargar datos de los gráficos con AJAX
    fetch('../../includes/charts/alerts_by_type.php')
        .then(response => response.json())
        .then(data => {
            alertTypeChart.data.labels = data.map(item => item.label);
            alertTypeChart.data.datasets[0].data = data.map(item => item.count);
            alertTypeChart.update();
        })
        .catch(error => console.error('Error al cargar datos de tipos de alertas:', error));

    fetch('../../includes/charts/alerts_by_operator.php')
        .then(response => response.json())
        .then(data => {
            operatorChart.data.labels = data.map(item => item.name);
            operatorChart.data.datasets[0].data = data.map(item => item.alert_count);
            operatorChart.update();
        })
        .catch(error => console.error('Error al cargar datos de operadores:', error));

    fetch('../../includes/charts/alerts_trend.php')
        .then(response => response.json())
        .then(data => {
            trendChart.data.labels = data.map(item => item.date);
            trendChart.data.datasets[0].data = data.map(item => item.count);
            trendChart.update();
        })
        .catch(error => console.error('Error al cargar datos de tendencias:', error));
});

// 🆕 NUEVO: Función para actualizar estado de dispositivos
function refreshDeviceStatus() {
    const btn = $('button:contains("Actualizar Estado")');
    const originalHtml = btn.html();
    
    btn.html('<i class="fas fa-spinner fa-spin"></i> Actualizando...')
       .prop('disabled', true);
    
    $.ajax({
        url: '<?php echo BASE_URL; ?>/pages/devices/refresh_status.php',
        method: 'POST',
        success: function(response) {
            if (response.success) {
                showNotification('success', 'Estado de dispositivos actualizado');
                // Recargar página después de 2 segundos
                setTimeout(() => location.reload(), 2000);
            } else {
                showNotification('error', 'Error al actualizar estado');
            }
        },
        error: function() {
            showNotification('error', 'Error de conexión');
        },
        complete: function() {
            btn.html(originalHtml).prop('disabled', false);
        }
    });
}

// 🆕 NUEVO: Auto-refresh cada 5 minutos (solo para stats)
<?php if ($device_stats['total_devices'] > 0): ?>
setInterval(function() {
    // Actualizar solo los números sin recargar toda la página
    $.get('<?php echo BASE_URL; ?>/pages/devices/get_stats.php', function(data) {
        if (data.success) {
            // Actualizar widgets con nuevos números
            $('.small-box .inner h3').each(function(index) {
                const stats = ['total_devices', 'online_devices', 'offline_devices', 'pending_config'];
                if (stats[index] && data.stats[stats[index]] !== undefined) {
                    $(this).text(data.stats[stats[index]]);
                }
            });
        }
    }).fail(function() {
        console.log('Error actualizando estadísticas de dispositivos');
    });
}, 300000); // 5 minutos
<?php endif; ?>

// 🆕 NUEVO: Mostrar notificaciones
function showNotification(type, message) {
    const alertClass = type === 'success' ? 'alert-success' : 
                      type === 'warning' ? 'alert-warning' : 'alert-danger';
    
    const notification = $(`
        <div class="alert ${alertClass} alert-dismissible fade show" style="position: fixed; top: 20px; right: 20px; z-index: 9999; min-width: 300px;">
            <button type="button" class="close" data-dismiss="alert">
                <span aria-hidden="true">&times;</span>
            </button>
            ${message}
        </div>
    `);
    
    $('body').append(notification);
    
    setTimeout(function() {
        notification.alert('close');
    }, 5000);
}
</script>

<style>
/* 🆕 NUEVO: Estilos para widgets de dispositivos */
.small-box {
    border-radius: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    transition: transform 0.2s;
}

.small-box:hover {
    transform: translateY(-2px);
}

.small-box .inner h3 {
    font-size: 2.2rem;
    font-weight: bold;
}

.card {
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border: none;
    border-radius: 10px;
}

.table-sm td {
    padding: 0.5rem;
}

.btn-action {
    margin-bottom: 10px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
    transition: all 0.3s;
}

.btn-action:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.btn-action i {
    font-size: 1.2rem;
    margin-bottom: 5px;
}

.badge {
    font-size: 0.8rem;
}

.progress-group {
    margin-bottom: 10px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 5px;
    border-left: 3px solid #dee2e6;
}

/* Separadores visuales */
hr.my-4 {
    border-top: 2px solid #e9ecef;
    margin: 2rem 0;
}

/* Mejorar info-boxes */
.info-box {
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    border-radius: 10px;
    transition: transform 0.2s;
}

.info-box:hover {
    transform: translateY(-2px);
}
</style>

<?php
// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>