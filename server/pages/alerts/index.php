<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../models/Alert.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Definir título de la página
$pageTitle = 'Dashboard de Alertas';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Alertas' => ''
];

// Crear instancia del modelo Alert
$alertModel = new Alert();

// Procesar filtros
$filters = [];
$page = isset($_GET['page']) ? (int)$_GET['page'] : 1;
$limit = 10; // Alertas por página
$offset = ($page - 1) * $limit;

// Tipo de alerta
if (isset($_GET['alert_type']) && !empty($_GET['alert_type'])) {
    $filters['alert_type'] = $_GET['alert_type'];
}

// Operador
if (isset($_GET['operator_id']) && !empty($_GET['operator_id'])) {
    $filters['operator_id'] = $_GET['operator_id'];
}

// Máquina
if (isset($_GET['machine_id']) && !empty($_GET['machine_id'])) {
    $filters['machine_id'] = $_GET['machine_id'];
}

// filtro de dispositivo
if (isset($_GET['device_id']) && !empty($_GET['device_id'])) {
    $filters['device_id'] = $_GET['device_id'];
}

// Fecha desde
if (isset($_GET['date_from']) && !empty($_GET['date_from'])) {
    $filters['date_from'] = $_GET['date_from'];
}

// Fecha hasta
if (isset($_GET['date_to']) && !empty($_GET['date_to'])) {
    $filters['date_to'] = $_GET['date_to'];
}

// Estado (reconocida o no)
if (isset($_GET['acknowledged']) && $_GET['acknowledged'] !== '') {
    $filters['acknowledged'] = (int)$_GET['acknowledged'];
}

// Obtener alertas con los filtros aplicados
$alerts = $alertModel->getAlerts($filters, $limit, $offset);

// Obtener recuento total para paginación
$totalAlerts = $alertModel->countAlerts($filters);
$totalPages = ceil($totalAlerts / $limit);

// Obtener operadores y máquinas para los filtros
$operatorsQuery = "SELECT id, name FROM operators ORDER BY name";
$operators = db_fetch_all($operatorsQuery);

$machinesQuery = "SELECT id, name FROM machines ORDER BY name";
$machines = db_fetch_all($machinesQuery);

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


<style>
    /* Estilos para el indicador de actualización */
    .alert-refresh-indicator {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin-bottom: 10px;
    }

    .refresh-spinner {
        animation: spin 1s infinite linear;
        margin-right: 5px;
    }

    @keyframes spin {
        0% {
            transform: rotate(0deg);
        }

        100% {
            transform: rotate(360deg);
        }
    }

    /* Efecto para destacar filas nuevas */
    .new-alert {
        animation: highlightNew 3s ease-out;
    }

    @keyframes highlightNew {
        0% {
            background-color: rgba(255, 193, 7, 0.3);
        }

        100% {
            background-color: transparent;
        }
    }
</style>




<div class="row mb-4">
    <div class="col-12">
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">
                    <i class="fas fa-sync"></i> Monitor de Reportes
                </h3>
                <div class="card-tools">
                    <button type="button" class="btn btn-tool" data-card-widget="collapse">
                        <i class="fas fa-minus"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-9">
                        <div id="monitor-status-container">
                            <div class="alert alert-light">
                                <i class="fas fa-info-circle mr-2"></i>
                                <span>Monitor iniciando...</span>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3 text-right">
                        <div class="btn-group">
                            <button id="execute-monitor" class="btn btn-primary">
                                <i class="fas fa-play"></i> Ejecutar Ahora
                            </button>
                            <button id="toggle-auto-execute" class="btn btn-danger">
                                <i class="fas fa-pause"></i> Detener Auto
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Resumen de alertas -->
<div class="row">
    <div class="col-lg-3 col-6">
        <!-- small box -->
        <div class="small-box bg-danger">
            <div class="inner">
                <?php
                $totalPendingQuery = "SELECT COUNT(*) as count FROM alerts WHERE acknowledged = 0";
                $pending = db_fetch_one($totalPendingQuery);
                ?>
                <h3><?php echo $pending['count']; ?></h3>
                <p>Alertas Pendientes</p>
            </div>
            <div class="icon">
                <i class="fas fa-exclamation-triangle"></i>
            </div>
            <a href="index.php?acknowledged=0" class="small-box-footer">
                Ver pendientes <i class="fas fa-arrow-circle-right"></i>
            </a>
        </div>
    </div>
    <!-- ./col -->
    <div class="col-lg-3 col-6">
        <!-- small box -->
        <div class="small-box bg-warning">
            <?php
            $todayQuery = "SELECT COUNT(*) as count FROM alerts WHERE DATE(timestamp) = CURDATE()";
            $today = db_fetch_one($todayQuery);
            ?>
            <div class="inner">
                <h3><?php echo $today['count']; ?></h3>
                <p>Alertas Hoy</p>
            </div>
            <div class="icon">
                <i class="far fa-calendar-alt"></i>
            </div>
            <a href="index.php?date_from=<?php echo date('Y-m-d'); ?>&date_to=<?php echo date('Y-m-d'); ?>" class="small-box-footer">
                Ver de hoy <i class="fas fa-arrow-circle-right"></i>
            </a>
        </div>
    </div>
    <!-- ./col -->
    <div class="col-lg-3 col-6">
        <!-- small box -->
        <div class="small-box bg-info">
            <?php
            $fatigueQuery = "SELECT COUNT(*) as count FROM alerts WHERE alert_type = 'fatigue'";
            $fatigue = db_fetch_one($fatigueQuery);
            ?>
            <div class="inner">
                <h3><?php echo $fatigue['count']; ?></h3>
                <p>Alertas de Fatiga</p>
            </div>
            <div class="icon">
                <i class="fas fa-bed"></i>
            </div>
            <a href="index.php?alert_type=fatigue" class="small-box-footer">
                Ver alertas de fatiga <i class="fas fa-arrow-circle-right"></i>
            </a>
        </div>
    </div>
    <!-- ./col -->
    <div class="col-lg-3 col-6">
        <!-- small box -->
        <div class="small-box bg-success">
            <?php
            $acknowledgedQuery = "SELECT COUNT(*) as count FROM alerts WHERE acknowledged = 1";
            $acknowledged = db_fetch_one($acknowledgedQuery);
            ?>
            <div class="inner">
                <h3><?php echo $acknowledged['count']; ?></h3>
                <p>Alertas Atendidas</p>
            </div>
            <div class="icon">
                <i class="fas fa-check-circle"></i>
            </div>
            <a href="index.php?acknowledged=1" class="small-box-footer">
                Ver atendidas <i class="fas fa-arrow-circle-right"></i>
            </a>
        </div>
    </div>


    <div class="col-lg-3 col-6">
        <div class="small-box bg-primary">
            <?php
            $devicesOfflineQuery = "SELECT COUNT(*) as count FROM devices WHERE status = 'offline'";
            $devicesOffline = db_fetch_one($devicesOfflineQuery);
            ?>
            <div class="inner">
                <h3><?php echo $devicesOffline['count']; ?></h3>
                <p>Dispositivos Desconectados</p>
            </div>
            <div class="icon">
                <i class="fas fa-microchip"></i>
            </div>
            <a href="../devices/index.php?status=offline" class="small-box-footer">
                Ver dispositivos <i class="fas fa-arrow-circle-right"></i>
            </a>
        </div>
    </div>
    <!-- ./col -->
</div>

<!-- Gráficos de Alertas -->
<div class="row">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Alertas por Tipo</h3>
            </div>
            <div class="card-body" style="height: 350px !important;">
                <canvas id="alertsByTypeChart" style="height: 250px;"></canvas>
            </div>
        </div>
    </div>
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Tendencia de Alertas</h3>
            </div>
            <div class="card-body">
                <canvas id="alertsTrendChart" style="height: 300px;"></canvas>
            </div>
        </div>
    </div>
</div>

<!-- Filtros -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Filtros</h3>
        <div class="card-tools">
            <button type="button" class="btn btn-tool" data-card-widget="collapse">
                <i class="fas fa-minus"></i>
            </button>
        </div>
    </div>
    <div class="card-body">
        <form method="GET" action="">
            <div class="row">
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Tipo de Alerta</label>
                        <select class="form-control" name="alert_type">
                            <option value="">Todas</option>
                            <option value="fatigue" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'fatigue') ? 'selected' : ''; ?>>Fatiga</option>
                            <option value="phone" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'phone') ? 'selected' : ''; ?>>Uso de Teléfono</option>
                            <option value="smoking" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'smoking') ? 'selected' : ''; ?>>Fumando</option>
                            <option value="yawn" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'yawn') ? 'selected' : ''; ?>>Bostezo</option>
                            <option value="distraction" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'distraction') ? 'selected' : ''; ?>>Distracción</option>
                            <option value="unauthorized" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'unauthorized') ? 'selected' : ''; ?>>No Autorizado</option>
                            <option value="behavior" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'behavior') ? 'selected' : ''; ?>>Comportamiento</option>
                            <option value="other" <?php echo (isset($_GET['alert_type']) && $_GET['alert_type'] == 'other') ? 'selected' : ''; ?>>Otro</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Operador</label>
                        <select class="form-control" name="operator_id">
                            <option value="">Todos</option>
                            <?php foreach ($operators as $operator): ?>
                                <option value="<?php echo $operator['id']; ?>" <?php echo (isset($_GET['operator_id']) && $_GET['operator_id'] == $operator['id']) ? 'selected' : ''; ?>>
                                    <?php echo $operator['name']; ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Máquina</label>
                        <select class="form-control" name="machine_id">
                            <option value="">Todas</option>
                            <?php foreach ($machines as $machine): ?>
                                <option value="<?php echo $machine['id']; ?>" <?php echo (isset($_GET['machine_id']) && $_GET['machine_id'] == $machine['id']) ? 'selected' : ''; ?>>
                                    <?php echo $machine['name']; ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Dispositivo</label>
                        <select class="form-control" name="device_id">
                            <option value="">Todos</option>
                            <?php
                            $devicesQuery = "SELECT id, device_id, name FROM devices ORDER BY name";
                            $devices = db_fetch_all($devicesQuery);
                            foreach ($devices as $device):
                            ?>
                                <option value="<?php echo $device['device_id']; ?>"
                                    <?php echo (isset($_GET['device_id']) && $_GET['device_id'] == $device['device_id']) ? 'selected' : ''; ?>>
                                    <?php echo $device['name'] ?? $device['device_id']; ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Estado</label>
                        <select class="form-control" name="acknowledged">
                            <option value="">Todos</option>
                            <option value="0" <?php echo (isset($_GET['acknowledged']) && $_GET['acknowledged'] == '0') ? 'selected' : ''; ?>>Pendiente</option>
                            <option value="1" <?php echo (isset($_GET['acknowledged']) && $_GET['acknowledged'] == '1') ? 'selected' : ''; ?>>Atendida</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Desde</label>
                        <input type="date" class="form-control" name="date_from" value="<?php echo isset($_GET['date_from']) ? $_GET['date_from'] : ''; ?>">
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Hasta</label>
                        <input type="date" class="form-control" name="date_to" value="<?php echo isset($_GET['date_to']) ? $_GET['date_to'] : ''; ?>">
                    </div>
                </div>
                <div class="col-md-6 d-flex align-items-end">
                    <div class="form-group w-100">
                        <button type="submit" class="btn btn-primary mr-2">
                            <i class="fas fa-search"></i> Filtrar
                        </button>
                        <a href="index.php" class="btn btn-default">
                            <i class="fas fa-times"></i> Limpiar
                        </a>
                        <a href="export.php<?php echo isset($_SERVER['QUERY_STRING']) && !empty($_SERVER['QUERY_STRING']) ? '?' . $_SERVER['QUERY_STRING'] : ''; ?>" class="btn btn-success float-right">
                            <i class="fas fa-file-excel"></i> Exportar
                        </a>
                    </div>
                </div>
            </div>
        </form>
    </div>
</div>


<!-- Reportes en procesamiento en tiempo real -->
<div class="card">
    <div class="card-header bg-warning">
        <div class="d-flex justify-content-between align-items-center">
            <h5 class="mb-0">
                <i class="fas fa-camera"></i> Reportes en Procesamiento
                <span class="badge badge-light reportes-counter">0</span>
            </h5>
            <div>
                <button type="button" id="actualizar-ahora" class="btn btn-info btn-sm">
                    <i class="fas fa-sync"></i> Actualizar Ahora
                </button>
                <div class="custom-control custom-switch d-inline-block ml-2">
                    <input type="checkbox" class="custom-control-input" id="auto-checkbox" checked>
                    <label class="custom-control-label" for="auto-checkbox">
                        Auto (<span class="countdown-timer">10</span>s)
                    </label>
                </div>
            </div>
        </div>
    </div>
    <div class="card-body">
        <div class="alert alert-info">
            <i class="fas fa-info-circle"></i> Estos reportes están siendo procesados y aparecerán en el listado de alertas automáticamente.
        </div>
        <div id="actualizando-mensaje" class="text-center" style="display: none;">
            <div class="spinner-border text-warning" role="status">
                <span class="sr-only">Actualizando...</span>
            </div>
            <p><i class="fas fa-sync fa-spin"></i> Actualizando...</p>
        </div>
        <div id="reportes-container">
            <div class="text-center p-3">Cargando reportes...</div>
        </div>
    </div>
</div>

<!-- Lista de Alertas -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Listado de Alertas</h3>
    </div>
    <div class="card-body">
        <div class="table-responsive">
            <table class="table table-bordered table-striped table-alertas">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Tipo</th>
                        <th>Operador</th>
                        <th>Máquina</th>
                        <th>Dispositivo</th>
                        <th>Fecha y Hora</th>
                        <th>Estado</th>
                        <th>Acciones</th>
                    </tr>
                </thead>
                <tbody>
                    <?php if (empty($alerts)): ?>
                        <tr>
                            <td colspan="7" class="text-center">No se encontraron alertas con los filtros aplicados</td>
                        </tr>
                    <?php else: ?>
                        <?php foreach ($alerts as $alert): ?>
                            <tr>
                                <td><?php echo $alert['id']; ?></td>
                                <td>
                                    <?php
                                    $alertTypeLabels = [
                                        'fatigue' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'danger') . '">Fatiga</span>',
                                        'yawn' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'warning') . '">Bostezo</span>',
                                        'phone' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'info') . '">Teléfono</span>',
                                        'smoking' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'secondary') . '">Fumando</span>',
                                        'distraction' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'primary') . '">Distracción</span>',
                                        'unauthorized' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'dark') . '">No Autorizado</span>',
                                        'behavior' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'info') . '">Comportamiento</span>',
                                        'other' => '<span class="badge badge-' . ($alert['acknowledged'] ? 'success' : 'light') . '">Otro</span>'
                                    ];
                                    echo $alertTypeLabels[$alert['alert_type']] ?? $alert['alert_type'];
                                    ?>
                                </td>
                                <td><?php echo $alert['operator_name'] ?? 'N/A'; ?></td>
                                <td><?php echo $alert['machine_name'] ?? 'N/A'; ?></td>
                                <td><?php echo $alert['device_id'] ?? 'N/A'; ?></td>
                                <td><?php echo date('d/m/Y H:i', strtotime($alert['timestamp'])); ?></td>
                                <td>
                                    <?php if ($alert['acknowledged']): ?>
                                        <span class="badge badge-success">Atendida</span>
                                        <?php if (!empty($alert['acknowledged_by'])): ?>
                                            <small class="text-muted "> por <?php echo $alert['acknowledged_by']; ?></small>
                                        <?php endif; ?>
                                    <?php else: ?>
                                        <span class="badge badge-danger">Pendiente</span>
                                    <?php endif; ?>
                                </td>
                                <td>
                                    <div class="btn-group">
                                        <a href="view.php?id=<?php echo $alert['id']; ?>" class="btn btn-sm btn-info">
                                            <i class="fas fa-eye"></i>
                                        </a>
                                        <?php if (!$alert['acknowledged'] && isset($_SESSION['user_role']) && ($_SESSION['user_role'] == 'admin' || $_SESSION['user_role'] == 'supervisor')): ?>
                                            <a href="acknowledge.php?id=<?php echo $alert['id']; ?>" class="btn btn-sm btn-success">
                                                <i class="fas fa-check"></i>
                                            </a>
                                        <?php endif; ?>
                                    </div>
                                </td>
                            </tr>
                        <?php endforeach; ?>
                    <?php endif; ?>
                </tbody>
            </table>
        </div>
    </div>
    <!-- Paginación -->
    <div class="card-footer clearfix">
        <ul class="pagination pagination-sm m-0 float-right">
            <?php for ($i = 1; $i <= $totalPages; $i++): ?>
                <?php
                // Conservar los parámetros de la URL al cambiar de página
                $queryParams = $_GET;
                $queryParams['page'] = $i;
                $queryString = http_build_query($queryParams);
                ?>
                <li class="page-item <?php echo ($page == $i) ? 'active' : ''; ?>">
                    <a class="page-link" href="?<?php echo $queryString; ?>"><?php echo $i; ?></a>
                </li>
            <?php endfor; ?>
        </ul>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="export.php" class="btn btn-success">
    <i class="fas fa-file-excel"></i> Exportar Alertas
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';
?>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
    // Datos para el gráfico de alertas por tipo
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
    document.addEventListener('DOMContentLoaded', function() {
        // Cargar datos para el gráfico de tipos
        fetch('../../includes/charts/alerts_by_type.php')
            .then(response => response.json())
            .then(data => {
                alertTypeChart.data.labels = data.map(item => item.label);
                alertTypeChart.data.datasets[0].data = data.map(item => item.count);
                alertTypeChart.update();
            })
            .catch(error => console.error('Error al cargar datos de tipos de alertas:', error));

        // Cargar datos para el gráfico de tendencias
        fetch('../../includes/charts/alerts_trend.php')
            .then(response => response.json())
            .then(data => {
                trendChart.data.labels = data.map(item => item.date);
                trendChart.data.datasets[0].data = data.map(item => item.count);
                trendChart.update();
            })
            .catch(error => console.error('Error al cargar datos de tendencias:', error));
    });
</script>

<?php
// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>