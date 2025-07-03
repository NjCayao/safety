<?php
// Incluir archivos necesarios
require_once 'config/config.php';
require_once 'config/database.php';
require_once 'includes/functions.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
  header('Location: login.php');
  exit;
}

// Obtener estadísticas para el dashboard
$stats = get_dashboard_stats();

// Obtener alertas recientes
$recentAlerts = get_active_alerts(5);

// Obtener datos para los gráficos
$alertsByType = get_chart_data('alerts_by_type');
$alertsByDay = get_chart_data('alerts_by_day');

// Definir título de la página
$pageTitle = 'Dashboard';

// Incluir archivos de cabecera y barra lateral
require_once 'includes/header.php';
require_once 'includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Small boxes (Stat box) -->
<div class="row">
  <div class="col-lg-3 col-6">
    <!-- small box -->
    <div class="small-box bg-info">
      <div class="inner">
        <h3><?php echo $stats['active_operators']; ?></h3>
        <p>Operadores Activos</p>
      </div>
      <div class="icon">
        <i class="ion ion-person"></i>
      </div>
      <a href="pages/operators/index.php" class="small-box-footer">Más info <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>
  <!-- ./col -->
  <div class="col-lg-3 col-6">
    <!-- small box -->
    <div class="small-box bg-success">
      <div class="inner">
        <h3><?php echo $stats['active_machines']; ?></h3>
        <p>Máquinas Activas</p>
      </div>
      <div class="icon">
        <i class="ion ion-android-bus"></i>
      </div>
      <a href="pages/machines/index.php" class="small-box-footer">Más info <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>
  <!-- ./col -->
  <div class="col-lg-3 col-6">
    <!-- small box -->
    <div class="small-box bg-warning">
      <div class="inner">
        <h3><?php echo $stats['today_alerts']; ?></h3>
        <p>Alertas Hoy</p>
      </div>
      <div class="icon">
        <i class="ion ion-alert-circled"></i>
      </div>
      <a href="pages/alerts/index.php" class="small-box-footer">Más info <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>
  <!-- ./col -->
  <div class="col-lg-3 col-6">
    <!-- small box -->
    <div class="small-box bg-danger">
      <div class="inner">
        <h3><?php echo $stats['pending_alerts']; ?></h3>
        <p>Alertas Pendientes</p>
      </div>
      <div class="icon">
        <i class="ion ion-ios-bell"></i>
      </div>
      <a href="pages/alerts/index.php?acknowledged=0" class="small-box-footer">Más info <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>
  <!-- ./col -->
</div>


<div class="row">
  <div class="col-lg-3 col-6">
    <!-- Dispositivos en línea -->
    <div class="small-box bg-primary">
      <div class="inner">
        <?php
        $devicesOnline = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'online'")['count'];
        ?>
        <h3><?php echo $devicesOnline; ?></h3>
        <p>Dispositivos en Línea</p>
      </div>
      <div class="icon">
        <i class="fas fa-microchip"></i>
      </div>
      <a href="pages/devices/index.php" class="small-box-footer">Ver dispositivos <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>

  <div class="col-lg-3 col-6">
    <!-- Dispositivos offline -->
    <div class="small-box bg-secondary">
      <div class="inner">
        <?php
        $devicesOffline = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'offline'")['count'];
        ?>
        <h3><?php echo $devicesOffline; ?></h3>
        <p>Dispositivos Desconectados</p>
      </div>
      <div class="icon">
        <i class="fas fa-unlink"></i>
      </div>
      <a href="pages/devices/index.php?status=offline" class="small-box-footer">Ver desconectados <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>

  <div class="col-lg-3 col-6">
    <!-- Alertas de dispositivos -->
    <div class="small-box bg-purple">
      <div class="inner">
        <?php
        $deviceAlerts = db_fetch_one("SELECT COUNT(*) as count FROM alerts WHERE alert_type = 'device_error' AND acknowledged = 0")['count'];
        ?>
        <h3><?php echo $deviceAlerts; ?></h3>
        <p>Alertas de Dispositivos</p>
      </div>
      <div class="icon">
        <i class="fas fa-exclamation-triangle"></i>
      </div>
      <a href="pages/alerts/index.php?alert_type=device_error" class="small-box-footer">Ver alertas <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>

  <div class="col-lg-3 col-6">
    <!-- Dispositivos sin máquina asignada -->
    <div class="small-box bg-dark">
      <div class="inner">
        <?php
        $devicesUnassigned = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE machine_id IS NULL")['count'];
        ?>
        <h3><?php echo $devicesUnassigned; ?></h3>
        <p>Sin Máquina Asignada</p>
      </div>
      <div class="icon">
        <i class="fas fa-question-circle"></i>
      </div>
      <a href="pages/devices/index.php?unassigned=1" class="small-box-footer">Ver sin asignar <i class="fas fa-arrow-circle-right"></i></a>
    </div>
  </div>
</div>
<!-- /.row -->

<!-- Main row -->
<div class="row">
  <!-- Left col -->
  <section class="col-lg-7 connectedSortable">
    <!-- Custom tabs (Charts with tabs)-->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-chart-pie mr-1"></i>
          Alertas por Tipo (Últimos 7 días)
        </h3>
      </div><!-- /.card-header -->
      <div class="card-body">
        <div class="tab-content p-0">
          <!-- Morris chart - Sales -->
          <div class="chart tab-pane active" id="revenue-chart" style="position: relative; height: 300px;">
            <canvas id="alerts-chart-canvas" height="300" style="height: 300px;"></canvas>
          </div>
        </div>
      </div><!-- /.card-body -->
    </div>
    <!-- /.card -->

    <!-- Alerts by day -->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-chart-line mr-1"></i>
          Tendencia de Alertas (Últimos 7 días)
        </h3>
      </div>
      <div class="card-body">
        <div class="chart">
          <canvas id="alerts-by-day-canvas" height="250" style="height: 250px;"></canvas>
        </div>
      </div>
    </div>

    <!-- Estado de Dispositivos Chart -->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-chart-bar mr-1"></i>
          Estado de Dispositivos
        </h3>
      </div>
      <div class="card-body">
        <div class="chart">
          <canvas id="devices-status-canvas" height="180" style="height: 180px;"></canvas>
        </div>
      </div>
    </div>

    <!-- /.card -->
  </section>
  <!-- /.Left col -->

  <!-- right col (We are only adding the ID to make the widgets sortable)-->
  <section class="col-lg-5 connectedSortable">

    <!-- Alerts List -->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-exclamation-triangle mr-1"></i>
          Alertas Recientes
        </h3>
      </div>

      <div class="card-body p-0">
        <ul class="products-list product-list-in-card pl-2 pr-2">
          <?php if (empty($recentAlerts)): ?>
            <li class="item">
              <div class="product-info">
                <span class="product-description text-center">
                  No hay alertas recientes
                </span>
              </div>
            </li>
          <?php else: ?>
            <?php foreach ($recentAlerts as $alert): ?>
              <li class="item">
                <div class="product-img">
                  <?php if (!empty($alert['image_path'])): ?>
                    <img src="<?php echo $alert['image_path']; ?>" alt="Alert Image" class="img-size-50">
                  <?php else: ?>
                    <img src="<?php echo ASSETS_URL; ?>/dist/img/default-150x150.png" alt="Default" class="img-size-50">
                  <?php endif; ?>
                </div>
                <div class="product-info">
                  <a href="pages/alerts/view.php?id=<?php echo $alert['id']; ?>" class="product-title">
                    <?php echo !empty($alert['operator_name']) ? $alert['operator_name'] : 'Operador desconocido'; ?>
                    <?php
                    $badgeClass = 'badge-info';
                    $alertTypeName = 'Otro';

                    switch ($alert['alert_type']) {
                      case 'fatigue':
                        $badgeClass = 'badge-danger';
                        $alertTypeName = 'Fatiga';
                        break;
                      case 'distraction':
                        $badgeClass = 'badge-warning';
                        $alertTypeName = 'Distracción';
                        break;
                      case 'yawn':
                        $badgeClass = 'badge-primary';
                        $alertTypeName = 'Bostezo';
                        break;
                      case 'phone':
                        $badgeClass = 'badge-warning';
                        $alertTypeName = 'Teléfono';
                        break;
                      case 'smoking':
                        $badgeClass = 'badge-danger';
                        $alertTypeName = 'Fumando';
                        break;
                      case 'unauthorized':
                        $badgeClass = 'badge-dark';
                        $alertTypeName = 'No autorizado';
                        break;
                      default:
                        $alertTypeName = ucfirst($alert['alert_type']);
                    }
                    ?>
                    <span class="badge <?php echo $badgeClass; ?> float-right"><?php echo $alertTypeName; ?></span>
                  </a>
                  <span class="product-description">
                    <?php
                    echo 'Máquina: ' . (!empty($alert['machine_name']) ? $alert['machine_name'] : 'Desconocida');

                    // información del dispositivo
                    if (!empty($alert['device_id'])) {
                      echo ' | Dispositivo: ' . $alert['device_id'];
                    }

                    // Función helper para el tiempo
                    if (function_exists('timeAgo')) {
                      echo ' - ' . timeAgo($alert['timestamp']);
                    } else {
                      echo ' - ' . date('d/m/Y H:i', strtotime($alert['timestamp']));
                    }
                    ?>
                  </span>
                </div>
              </li>
            <?php endforeach; ?>
          <?php endif; ?>
        </ul>
      </div>
      <!-- /.card-body -->
      <div class="card-footer text-center">
        <a href="pages/alerts/index.php" class="uppercase">Ver Todas Las Alertas</a>
      </div>
      <!-- /.card-footer -->
    </div>
    <!-- /.card -->

    <!-- Quick actions card -->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-bolt mr-1"></i>
          Acciones Rápidas
        </h3>
      </div>
      <div class="card-body">
        <div class="row">
          <div class="col-md-6">
            <a href="pages/operators/create.php" class="btn btn-info btn-block mb-3">
              <i class="fas fa-user-plus mr-2"></i> Nuevo Operador
            </a>
          </div>
          <div class="col-md-6">
            <a href="pages/machines/create.php" class="btn btn-success btn-block mb-3">
              <i class="fas fa-plus-circle mr-2"></i> Nueva Máquina
            </a>
          </div>
        </div>
        <div class="row">
          <div class="col-md-6">
            <a href="pages/operators/assign.php" class="btn btn-warning btn-block mb-3">
              <i class="fas fa-link mr-2"></i> Asignar Operador
            </a>
          </div>
          <div class="col-md-6">
            <a href="pages/reports/daily.php" class="btn btn-primary btn-block mb-3">
              <i class="fas fa-chart-bar mr-2"></i> Reporte Diario
            </a>
          </div>
        </div>

        <div class="row">
          <div class="col-md-6">
            <a href="pages/devices/create.php" class="btn btn-dark btn-block mb-3">
              <i class="fas fa-microchip mr-2"></i> Nuevo Dispositivo
            </a>
          </div>
          <div class="col-md-6">
            <a href="pages/devices/index.php?status=offline" class="btn btn-secondary btn-block mb-3">
              <i class="fas fa-unlink mr-2"></i> Dispositivos Offline
            </a>
          </div>
        </div>

        <div class="row">
          <div class="col-md-12">
            <a href="pages/alerts/index.php?acknowledged=0" class="btn btn-danger btn-block">
              <i class="fas fa-exclamation-circle mr-2"></i> Gestionar Alertas Pendientes
            </a>
          </div>
        </div>
      </div>
    </div>


    <!-- Estado de Dispositivos -->
    <div class="card">
      <div class="card-header">
        <h3 class="card-title">
          <i class="fas fa-microchip mr-1"></i>
          Estado de Dispositivos
        </h3>
        <div class="card-tools">
          <button type="button" class="btn btn-tool" data-card-widget="collapse">
            <i class="fas fa-minus"></i>
          </button>
        </div>
      </div>
      <div class="card-body">
        <?php
        $devices = db_fetch_all("
      SELECT d.*, m.name as machine_name,
             TIMESTAMPDIFF(MINUTE, last_access, NOW()) as minutes_offline
      FROM devices d
      LEFT JOIN machines m ON d.machine_id = m.id
      ORDER BY d.status DESC, d.name
      LIMIT 5
    ");
        ?>

        <?php if (empty($devices)): ?>
          <p class="text-muted text-center">No hay dispositivos registrados</p>
        <?php else: ?>
          <ul class="products-list product-list-in-card pl-2 pr-2">
            <?php foreach ($devices as $device): ?>
              <li class="item">
                <div class="product-img">
                  <?php if ($device['status'] == 'online'): ?>
                    <i class="fas fa-check-circle fa-2x text-success"></i>
                  <?php elseif ($device['status'] == 'offline'): ?>
                    <i class="fas fa-times-circle fa-2x text-danger"></i>
                  <?php else: ?>
                    <i class="fas fa-exclamation-circle fa-2x text-warning"></i>
                  <?php endif; ?>
                </div>
                <div class="product-info">
                  <a href="pages/devices/view.php?id=<?php echo $device['id']; ?>" class="product-title">
                    <?php echo $device['name'] ?? $device['device_id']; ?>
                    <span class="badge <?php echo $device['status'] == 'online' ? 'badge-success' : 'badge-danger'; ?> float-right">
                      <?php echo ucfirst($device['status']); ?>
                    </span>
                  </a>
                  <span class="product-description">
                    <?php echo $device['machine_name'] ?? 'Sin máquina asignada'; ?>
                    <?php if ($device['status'] == 'offline' && $device['minutes_offline']): ?>
                      <br><small class="text-muted">Offline hace <?php echo $device['minutes_offline']; ?> min</small>
                    <?php endif; ?>
                  </span>
                </div>
              </li>
            <?php endforeach; ?>
          </ul>
        <?php endif; ?>
      </div>
      <div class="card-footer text-center">
        <a href="pages/devices/index.php" class="uppercase">Ver Todos Los Dispositivos</a>
      </div>
    </div>
    <!-- /.card -->
  </section>
  <!-- right col -->
</div>
<!-- /.row (main row) -->

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Incluir archivo de contenido
require_once 'includes/content.php';

// JavaScript específico para esta página
$extraJs = '
<script>
  $(function () {
    // Datos para el gráfico de tipos de alertas
    var alertsByTypeData = {
      labels: [';

// Generar dinámicamente las etiquetas para el gráfico de alertas
$typeLabels = [];
foreach ($alertsByType as $item) {
  $typeLabels[] = '"' . $item['label'] . '"';
}
$extraJs .= implode(', ', $typeLabels);

$extraJs .= '],
      datasets: [{
        data: [';

// Generar dinámicamente los valores para el gráfico de alertas
$typeValues = [];
foreach ($alertsByType as $item) {
  $typeValues[] = $item['value'];
}
$extraJs .= implode(', ', $typeValues);

$extraJs .= '],
        backgroundColor: ["#f56954", "#00a65a", "#f39c12", "#00c0ef", "#3c8dbc", "#d2d6de"],
      }]
    };

    // Crear gráfico de pie para tipos de alertas
    var pieChartCanvas = $("#alerts-chart-canvas").get(0).getContext("2d");
    new Chart(pieChartCanvas, {
      type: "pie",
      data: alertsByTypeData,
      options: {
        maintainAspectRatio: false,
        responsive: true,
      }
    });

    // Datos para el gráfico de alertas por día
    var alertsByDayData = {
      labels: [';

// Generar dinámicamente las etiquetas para el gráfico de días
$dayLabels = [];
foreach ($alertsByDay as $item) {
  $dayLabels[] = '"' . $item['label'] . '"';
}
$extraJs .= implode(', ', $dayLabels);

$extraJs .= '],
      datasets: [{
        label: "Alertas",
        backgroundColor: "rgba(60,141,188,0.9)",
        borderColor: "rgba(60,141,188,0.8)",
        pointRadius: 3,
        pointColor: "#3b8bba",
        pointStrokeColor: "rgba(60,141,188,1)",
        pointHighlightFill: "#fff",
        pointHighlightStroke: "rgba(60,141,188,1)",
        data: [';

// Generar dinámicamente los valores para el gráfico de días
$dayValues = [];
foreach ($alertsByDay as $item) {
  $dayValues[] = $item['value'];
}
$extraJs .= implode(', ', $dayValues);

$extraJs .= ']
      }]
    };

    // Crear gráfico de línea para alertas por día
    var lineChartCanvas = $("#alerts-by-day-canvas").get(0).getContext("2d");
    new Chart(lineChartCanvas, {
      type: "line",
      data: alertsByDayData,
      options: {
        maintainAspectRatio: false,
        responsive: true,
        scales: {
          xAxes: [{
            gridLines: {
              display: false,
            }
          }],
          yAxes: [{
            gridLines: {
              display: false,
            },
            ticks: {
              beginAtZero: true,
              stepSize: 1
            }
          }]
        }
      }
    });

    // NUEVO GRÁFICO DE ESTADO DE DISPOSITIVOS
    // Datos para el gráfico de estado de dispositivos
    var deviceStatusData = {
      labels: ["En línea", "Desconectados", "Error", "Sincronizando"],
      datasets: [{
        label: "Dispositivos",
        backgroundColor: ["#28a745", "#dc3545", "#ffc107", "#17a2b8"],
        borderColor: ["#28a745", "#dc3545", "#ffc107", "#17a2b8"],
        data: [';

// Obtener datos de dispositivos
$online = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'online'")['count'];
$offline = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'offline'")['count'];
$error = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'error'")['count'];
$syncing = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'syncing'")['count'];

$extraJs .= "$online, $offline, $error, $syncing";

$extraJs .= ']
      }]
    };

    // Crear gráfico de barras para estado de dispositivos
    var barChartCanvas = $("#devices-status-canvas").get(0).getContext("2d");
    new Chart(barChartCanvas, {
      type: "bar",
      data: deviceStatusData,
      options: {
        maintainAspectRatio: false,
        responsive: true,
        legend: {
          display: false
        },
        scales: {
          yAxes: [{
            ticks: {
              beginAtZero: true,
              stepSize: 1
            }
          }]
        }
      }
    });
  });
</script>
';

// Incluir archivo de pie de página
require_once 'includes/footer.php';
?>