<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Incluir necesarios
require_once '../../config/config.php';
require_once '../../config/database.php'; // Añadido para consultas directas a BD
require_once '../../models/Alert.php';
require_once '../../models/Operator.php';
require_once '../../models/Machine.php';

// Procesar filtros
$filters = [];
if (isset($_GET['date_from']) && !empty($_GET['date_from'])) {
    $filters['date_from'] = $_GET['date_from'];
}
if (isset($_GET['date_to']) && !empty($_GET['date_to'])) {
    $filters['date_to'] = $_GET['date_to'];
}
if (isset($_GET['operator_id']) && !empty($_GET['operator_id'])) {
    $filters['operator_id'] = $_GET['operator_id'];
}
if (isset($_GET['machine_id']) && !empty($_GET['machine_id'])) {
    $filters['machine_id'] = $_GET['machine_id'];
}

$alertModel = new Alert();
$operatorModel = new Operator();
$machineModel = new Machine();

// Obtener lista de operadores para filtro
$operators = $operatorModel->getAllOperators();

// Obtener lista de máquinas para filtro
$machines = $machineModel->getAllMachines();

// Obtener estadísticas por operador
$alertsByOperator = $alertModel->getAlertsByOperator(10, $filters['date_from'] ?? null, $filters['date_to'] ?? null);

// Obtener datos para el gráfico de tendencias
$trendDays = isset($_GET['trend_days']) ? intval($_GET['trend_days']) : 30;
$trendGroupBy = isset($_GET['trend_group_by']) && $_GET['trend_group_by'] == 'hour' ? 'hour' : 'day';

// Definir título de la página
$pageTitle = 'Estadísticas Detalladas';

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
        <!-- Filtros -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">
                    <i class="fas fa-filter"></i>
                    Filtros de Estadísticas
                </h3>
                <div class="card-tools">
                    <button type="button" class="btn btn-tool" data-card-widget="collapse">
                        <i class="fas fa-minus"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <form method="GET" id="filter-form">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Fecha desde</label>
                                <div class="input-group date" id="date-from" data-target-input="nearest">
                                    <input type="date" class="form-control" name="date_from" value="<?php echo isset($filters['date_from']) ? $filters['date_from'] : ''; ?>">
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Fecha hasta</label>
                                <div class="input-group date" id="date-to" data-target-input="nearest">
                                    <input type="date" class="form-control" name="date_to" value="<?php echo isset($filters['date_to']) ? $filters['date_to'] : ''; ?>">
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Operador</label>
                                <select class="form-control select2" name="operator_id" style="width: 100%;">
                                    <option value="">Todos los operadores</option>
                                    <?php foreach ($operators as $operator): ?>
                                    <option value="<?php echo $operator['id']; ?>" <?php echo (isset($filters['operator_id']) && $filters['operator_id'] == $operator['id']) ? 'selected' : ''; ?>>
                                        <?php echo $operator['name'] . ' (' . (isset($operator['dni_number']) ? $operator['dni_number'] : $operator['dni']) . ')'; ?>
                                    </option>
                                    <?php endforeach; ?>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Máquina</label>
                                <select class="form-control select2" name="machine_id" style="width: 100%;">
                                    <option value="">Todas las máquinas</option>
                                    <?php foreach ($machines as $machine): ?>
                                    <option value="<?php echo $machine['id']; ?>" <?php echo (isset($filters['machine_id']) && $filters['machine_id'] == $machine['id']) ? 'selected' : ''; ?>>
                                        <?php echo $machine['name']; ?>
                                    </option>
                                    <?php endforeach; ?>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="row">
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Días de tendencia</label>
                                <select class="form-control" name="trend_days">
                                    <option value="7" <?php echo $trendDays == 7 ? 'selected' : ''; ?>>Última semana</option>
                                    <option value="14" <?php echo $trendDays == 14 ? 'selected' : ''; ?>>Últimos 14 días</option>
                                    <option value="30" <?php echo $trendDays == 30 ? 'selected' : ''; ?>>Último mes</option>
                                    <option value="90" <?php echo $trendDays == 90 ? 'selected' : ''; ?>>Últimos 3 meses</option>
                                    <option value="180" <?php echo $trendDays == 180 ? 'selected' : ''; ?>>Últimos 6 meses</option>
                                    <option value="365" <?php echo $trendDays == 365 ? 'selected' : ''; ?>>Último año</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Agrupar por</label>
                                <select class="form-control" name="trend_group_by">
                                    <option value="day" <?php echo $trendGroupBy == 'day' ? 'selected' : ''; ?>>Día</option>
                                    <option value="hour" <?php echo $trendGroupBy == 'hour' ? 'selected' : ''; ?>>Hora</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="form-group" style="margin-top: 32px;">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-search"></i> Filtrar
                                </button>
                                <a href="stats.php" class="btn btn-default">
                                    <i class="fas fa-sync-alt"></i> Limpiar filtros
                                </a>
                                <button type="button" id="export-excel" class="btn btn-success float-right ml-2">
                                    <i class="fas fa-file-excel"></i> Exportar Excel
                                </button>
                                <button type="button" id="export-pdf" class="btn btn-danger float-right">
                                    <i class="fas fa-file-pdf"></i> Exportar PDF
                                </button>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        </div>
        
        <!-- Gráficos y estadísticas -->
        <div class="row">
            <div class="col-md-6">
                <!-- Alertas por Tipo -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-chart-pie mr-1"></i>
                            Distribución de Alertas por Tipo
                        </h3>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsByTypeChart" style="height: 300px;"></canvas>
                    </div>
                </div>
                
                <!-- Alertas por Operador -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-user mr-1"></i>
                            Top Operadores con Más Alertas
                        </h3>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsByOperatorChart" style="height: 300px;"></canvas>
                    </div>
                </div>
                
                <!-- Tabla de resumen por operador -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-table mr-1"></i>
                            Resumen por Operador
                        </h3>
                    </div>
                    <div class="card-body table-responsive p-0">
                        <table class="table table-hover text-nowrap">
                            <thead>
                                <tr>
                                    <th>Operador</th>
                                    <th>DNI</th>
                                    <th>Total Alertas</th>
                                    <th>% del Total</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php 
                                // Calcular el total de alertas para porcentajes
                                $totalAlertCount = array_sum(array_column($alertsByOperator, 'alert_count'));
                                
                                foreach ($alertsByOperator as $operator): 
                                    $percentage = $totalAlertCount > 0 ? 
                                        round(($operator['alert_count'] / $totalAlertCount) * 100, 2) : 0;
                                ?>
                                <tr>
                                    <td>
                                        <a href="../operators/view.php?id=<?php echo $operator['id']; ?>">
                                            <?php echo $operator['name']; ?>
                                        </a>
                                    </td>
                                    <td><?php echo isset($operator['dni_number']) ? $operator['dni_number'] : $operator['dni']; ?></td>
                                    <td><?php echo $operator['alert_count']; ?></td>
                                    <td>
                                        <div class="progress progress-xs">
                                            <div class="progress-bar bg-primary" style="width: <?php echo $percentage; ?>%"></div>
                                        </div>
                                        <span class="badge bg-primary"><?php echo $percentage; ?>%</span>
                                    </td>
                                </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <!-- Tendencia de Alertas -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-chart-line mr-1"></i>
                            Tendencia de Alertas
                        </h3>
                    </div>
                    <div class="card-body">
                        <canvas id="alertsTrendChart" style="height: 300px;"></canvas>
                    </div>
                </div>
                
                <!-- Patrones por Hora del Día -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-clock mr-1"></i>
                            Distribución de Alertas por Hora del Día
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <i class="icon fas fa-info"></i>
                            Próximamente: Gráfico de distribución de alertas por hora del día.
                        </div>
                    </div>
                </div>
                
                <!-- Distribución por Día de la Semana -->
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">
                            <i class="fas fa-calendar-week mr-1"></i>
                            Distribución de Alertas por Día de la Semana
                        </h3>
                    </div>
                    <div class="card-body">
                        <div class="alert alert-info">
                            <i class="icon fas fa-info"></i>
                            Próximamente: Gráfico de distribución de alertas por día de la semana.
                        </div>
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
<button type="button" id="export-stats-btn" class="btn btn-success">
    <i class="fas fa-file-excel"></i> Exportar Estadísticas
</button>';

// Incluir archivo de contenido
require_once '../../includes/content.php';
?>

<!-- Chart.js -->
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar Select2 si existe
    if (typeof $.fn.select2 !== 'undefined') {
        $('.select2').select2();
    }
    
    // Inicializar DateTimePicker si existe
    if (typeof $.fn.datetimepicker !== 'undefined') {
        $('#date-from, #date-to').datetimepicker({
            format: 'YYYY-MM-DD'
        });
    }
    
    // Exportar a Excel
    $('#export-excel').click(function(e) {
        e.preventDefault();
        var url = 'export_stats.php?format=excel&' + $('#filter-form').serialize();
        window.location.href = url;
    });
    
    // Exportar a PDF
    $('#export-pdf').click(function(e) {
        e.preventDefault();
        var url = 'export_stats.php?format=pdf&' + $('#filter-form').serialize();
        window.location.href = url;
    });

    // Exportar desde botón de header
    $('#export-stats-btn').click(function(e) {
        e.preventDefault();
        var url = 'export_stats.php?format=excel&' + $('#filter-form').serialize();
        window.location.href = url;
    });

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
                    text: 'Tendencia de Alertas en los Últimos ' + <?php echo $trendDays; ?> + ' Días'
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

    // Construir el querystring para pasar a los endpoints de gráficos
    const buildQueryString = () => {
        let queryParams = new URLSearchParams();
        
        // Añadir parámetros de filtros
        const dateFrom = document.querySelector('input[name="date_from"]');
        if (dateFrom && dateFrom.value) {
            queryParams.append('date_from', dateFrom.value);
        }
        
        const dateTo = document.querySelector('input[name="date_to"]');
        if (dateTo && dateTo.value) {
            queryParams.append('date_to', dateTo.value);
        }
        
        const operatorId = document.querySelector('select[name="operator_id"]');
        if (operatorId && operatorId.value) {
            queryParams.append('operator_id', operatorId.value);
        }
        
        const machineId = document.querySelector('select[name="machine_id"]');
        if (machineId && machineId.value) {
            queryParams.append('machine_id', machineId.value);
        }
        
        const trendDays = document.querySelector('select[name="trend_days"]');
        if (trendDays && trendDays.value) {
            queryParams.append('days', trendDays.value);
        }
        
        const trendGroupBy = document.querySelector('select[name="trend_group_by"]');
        if (trendGroupBy && trendGroupBy.value) {
            queryParams.append('group_by', trendGroupBy.value);
        }
        
        return queryParams.toString();
    };

    // Función para cargar los datos de los gráficos
    const loadChartData = () => {
        const queryString = buildQueryString();
        
        // Cargar datos para el gráfico de tipos de alertas
        fetch('../../includes/charts/alerts_by_type.php?' + queryString)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                alertTypeChart.data.labels = data.map(item => item.label);
                alertTypeChart.data.datasets[0].data = data.map(item => item.count);
                alertTypeChart.update();
            })
            .catch(error => {
                console.error('Error al cargar datos de tipos de alertas:', error);
                // Mostrar mensaje de error en el gráfico
                alertTypeChart.data.labels = ['Error'];
                alertTypeChart.data.datasets[0].data = [0];
                alertTypeChart.update();
            });

        // Añadir límite para el gráfico de operadores
        let operatorQueryString = queryString;
        if (operatorQueryString.length > 0) {
            operatorQueryString += '&limit=10';
        } else {
            operatorQueryString = 'limit=10';
        }
        
        // Cargar datos para el gráfico de operadores
        fetch('../../includes/charts/alerts_by_operator.php?' + operatorQueryString)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                operatorChart.data.labels = data.map(item => item.name);
                operatorChart.data.datasets[0].data = data.map(item => item.alert_count);
                operatorChart.update();
            })
            .catch(error => {
                console.error('Error al cargar datos de operadores:', error);
                // Mostrar mensaje de error en el gráfico
                operatorChart.data.labels = ['Error'];
                operatorChart.data.datasets[0].data = [0];
                operatorChart.update();
            });
        
        // Cargar datos para el gráfico de tendencias
        fetch('../../includes/charts/alerts_trend.php?' + queryString)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor: ' + response.status);
                }
                return response.json();
            })
            .then(data => {
                trendChart.data.labels = data.map(item => item.date);
                trendChart.data.datasets[0].data = data.map(item => item.count);
                trendChart.update();
            })
            .catch(error => {
                console.error('Error al cargar datos de tendencias:', error);
                // Mostrar mensaje de error en el gráfico
                trendChart.data.labels = ['Error'];
                trendChart.data.datasets[0].data = [0];
                trendChart.update();
            });
    };
    
    // Cargar los datos inicialmente
    loadChartData();
    
    // Recargar datos al enviar el formulario
    document.getElementById('filter-form').addEventListener('submit', function(e) {
        e.preventDefault();
        loadChartData();
        
        // Actualizar la URL para mantener los parámetros
        const formData = new FormData(this);
        const queryString = new URLSearchParams(formData).toString();
        history.pushState(null, '', '?' + queryString);
    });
});
</script>

<?php
require_once '../../includes/footer.php';
?>