<?php
/**
 * Página de estado en tiempo real de dispositivos
 * server/pages/devices/status.php
 */

// Primero iniciar sesión si no está iniciada
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Cargar configuración y verificar autenticación
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/auth.php';
require_once '../../includes/device_config.php';

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    header("Location: " . BASE_URL . "/login.php");
    exit();
}

// Definir el título de la página
$pageTitle = "Estado en Tiempo Real - Dispositivos";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener dispositivos con información de estado actualizada
$devices = db_fetch_all("
    SELECT d.*, 
           m.name as machine_name,
           m.location as machine_location,
           CASE 
               WHEN d.last_access IS NULL THEN 'never_connected'
               WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 2 MINUTE) AND d.status = 'online' THEN 'stale_online'
               WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 5 MINUTE) THEN 'offline'
               WHEN d.last_access < DATE_SUB(NOW(), INTERVAL 1 HOUR) THEN 'long_offline'
               ELSE d.status
           END as computed_status,
           TIMESTAMPDIFF(MINUTE, d.last_access, NOW()) as minutes_since_last_access,
           (SELECT COUNT(*) FROM alerts a WHERE a.device_id = d.device_id AND a.acknowledged = 0 AND a.timestamp > DATE_SUB(NOW(), INTERVAL 24 HOUR)) as pending_alerts,
           (SELECT COUNT(*) FROM events e WHERE e.device_id = d.device_id AND e.is_synced = 0) as pending_events
    FROM devices d
    LEFT JOIN machines m ON d.machine_id = m.id
    ORDER BY d.last_access DESC, d.device_id ASC
");

// Funciones auxiliares
function getStatusInfo($computed_status, $minutes_since_last_access) {
    switch ($computed_status) {
        case 'online':
            return [
                'class' => 'success',
                'icon' => 'fa-circle',
                'text' => 'En línea',
                'description' => 'Conectado y funcionando'
            ];
        case 'stale_online':
            return [
                'class' => 'warning',
                'icon' => 'fa-circle',
                'text' => 'Conexión irregular',
                'description' => 'Marcado como online pero sin actividad reciente'
            ];
        case 'offline':
            $time_desc = $minutes_since_last_access < 60 ? 
                $minutes_since_last_access . ' min' : 
                round($minutes_since_last_access / 60, 1) . ' hrs';
            return [
                'class' => 'danger',
                'icon' => 'fa-circle',
                'text' => 'Fuera de línea',
                'description' => "Última conexión hace $time_desc"
            ];
        case 'long_offline':
            $hours = round($minutes_since_last_access / 60);
            $time_desc = $hours > 24 ? round($hours / 24, 1) . ' días' : $hours . ' hrs';
            return [
                'class' => 'secondary',
                'icon' => 'fa-circle',
                'text' => 'Offline prolongado',
                'description' => "Sin conexión hace $time_desc"
            ];
        case 'syncing':
            return [
                'class' => 'info',
                'icon' => 'fa-sync fa-spin',
                'text' => 'Sincronizando',
                'description' => 'Transfiriendo datos'
            ];
        case 'error':
            return [
                'class' => 'danger',
                'icon' => 'fa-exclamation-triangle',
                'text' => 'Error',
                'description' => 'Error de comunicación'
            ];
        case 'never_connected':
            return [
                'class' => 'dark',
                'icon' => 'fa-question-circle',
                'text' => 'Nunca conectado',
                'description' => 'Dispositivo no se ha conectado'
            ];
        default:
            return [
                'class' => 'secondary',
                'icon' => 'fa-question',
                'text' => 'Desconocido',
                'description' => 'Estado no determinado'
            ];
    }
}

function getLastSeenText($last_access, $minutes_since_last_access) {
    if (!$last_access) return 'Nunca';
    
    if ($minutes_since_last_access < 1) return 'Ahora mismo';
    if ($minutes_since_last_access < 60) return "Hace {$minutes_since_last_access} min";
    
    $hours = round($minutes_since_last_access / 60);
    if ($hours < 24) return "Hace {$hours} hrs";
    
    $days = round($hours / 24, 1);
    return "Hace {$days} días";
}

// Estadísticas de resumen
$stats = [
    'total' => count($devices),
    'online' => 0,
    'offline' => 0,
    'error' => 0,
    'never_connected' => 0,
    'pending_alerts' => 0,
    'pending_events' => 0
];

foreach ($devices as $device) {
    switch ($device['computed_status']) {
        case 'online':
            $stats['online']++;
            break;
        case 'offline':
        case 'long_offline':
        case 'stale_online':
            $stats['offline']++;
            break;
        case 'error':
            $stats['error']++;
            break;
        case 'never_connected':
            $stats['never_connected']++;
            break;
    }
    $stats['pending_alerts'] += $device['pending_alerts'];
    $stats['pending_events'] += $device['pending_events'];
}
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Estado en Tiempo Real</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Estado en Tiempo Real</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            
            <!-- Estadísticas de resumen -->
            <div class="row">
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-info">
                        <div class="inner">
                            <h3><?php echo $stats['total']; ?></h3>
                            <p>Total Dispositivos</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-microchip"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-success">
                        <div class="inner">
                            <h3><?php echo $stats['online']; ?></h3>
                            <p>En Línea</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-wifi"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-danger">
                        <div class="inner">
                            <h3><?php echo $stats['offline'] + $stats['error']; ?></h3>
                            <p>Offline/Error</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-warning">
                        <div class="inner">
                            <h3><?php echo $stats['pending_alerts']; ?></h3>
                            <p>Alertas Pendientes</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-bell"></i>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Controles y filtros -->
            <div class="card collapsed-card">
                <div class="card-header">
                    <h3 class="card-title">Controles y Filtros</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-tool" data-card-widget="collapse">
                            <i class="fas fa-plus"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Filtrar por Estado:</label>
                                <select class="form-control" id="statusFilter">
                                    <option value="">Todos</option>
                                    <option value="online">En línea</option>
                                    <option value="offline">Fuera de línea</option>
                                    <option value="error">Con errores</option>
                                    <option value="never_connected">Nunca conectados</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Auto-actualizar:</label>
                                <div class="custom-control custom-switch">
                                    <input type="checkbox" class="custom-control-input" id="autoRefresh" checked>
                                    <label class="custom-control-label" for="autoRefresh">Cada 30 segundos</label>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Vista:</label>
                                <div class="btn-group btn-group-toggle" data-toggle="buttons">
                                    <label class="btn btn-secondary active">
                                        <input type="radio" name="viewMode" id="cardView" checked> Tarjetas
                                    </label>
                                    <label class="btn btn-secondary">
                                        <input type="radio" name="viewMode" id="tableView"> Tabla
                                    </label>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>&nbsp;</label>
                                <button type="button" class="btn btn-primary btn-block" onclick="refreshStatus()">
                                    <i class="fas fa-sync"></i> Actualizar Ahora
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Vista de tarjetas (por defecto) -->
            <div id="cardView" class="device-view">
                <div class="row" id="deviceCards">
                    <?php foreach ($devices as $device): ?>
                        <?php $status_info = getStatusInfo($device['computed_status'], $device['minutes_since_last_access']); ?>
                        <div class="col-md-6 col-lg-4 device-card" 
                             data-status="<?php echo $device['computed_status']; ?>"
                             data-device-id="<?php echo $device['device_id']; ?>">
                            <div class="card card-outline card-<?php echo $status_info['class']; ?>">
                                <div class="card-header">
                                    <h3 class="card-title">
                                        <i class="fas <?php echo $status_info['icon']; ?> text-<?php echo $status_info['class']; ?>"></i>
                                        <?php echo htmlspecialchars($device['name'] ?: $device['device_id']); ?>
                                    </h3>
                                    <div class="card-tools">
                                        <span class="badge badge-<?php echo $status_info['class']; ?>">
                                            <?php echo $status_info['text']; ?>
                                        </span>
                                    </div>
                                </div>
                                <div class="card-body">
                                    <div class="info-item">
                                        <strong>ID:</strong> <?php echo htmlspecialchars($device['device_id']); ?>
                                    </div>
                                    <div class="info-item">
                                        <strong>Tipo:</strong> <?php echo htmlspecialchars($device['device_type']); ?>
                                    </div>
                                    <div class="info-item">
                                        <strong>Máquina:</strong> 
                                        <?php echo htmlspecialchars($device['machine_name'] ?: 'No asignada'); ?>
                                    </div>
                                    <div class="info-item">
                                        <strong>Última conexión:</strong> 
                                        <?php echo getLastSeenText($device['last_access'], $device['minutes_since_last_access']); ?>
                                    </div>
                                    <div class="info-item">
                                        <strong>IP:</strong> <?php echo htmlspecialchars($device['ip_address'] ?: 'N/A'); ?>
                                    </div>
                                    
                                    <?php if ($device['pending_alerts'] > 0): ?>
                                        <div class="info-item">
                                            <span class="badge badge-warning">
                                                <i class="fas fa-bell"></i> <?php echo $device['pending_alerts']; ?> alertas pendientes
                                            </span>
                                        </div>
                                    <?php endif; ?>
                                    
                                    <?php if ($device['config_pending']): ?>
                                        <div class="info-item">
                                            <span class="badge badge-info">
                                                <i class="fas fa-cog"></i> Configuración pendiente
                                            </span>
                                        </div>
                                    <?php endif; ?>
                                </div>
                                <div class="card-footer">
                                    <div class="btn-group btn-group-sm" role="group">
                                        <a href="view.php?id=<?php echo $device['id']; ?>" 
                                           class="btn btn-outline-info" title="Ver detalles">
                                            <i class="fas fa-eye"></i>
                                        </a>
                                        <button type="button" 
                                                class="btn btn-outline-primary" 
                                                onclick="testConnection('<?php echo $device['device_id']; ?>')"
                                                title="Probar conexión">
                                            <i class="fas fa-wifi"></i>
                                        </button>
                                        <?php if ($device['config_pending']): ?>
                                        <a href="configure.php?id=<?php echo $device['device_id']; ?>" 
                                           class="btn btn-outline-warning" title="Configurar">
                                            <i class="fas fa-cog"></i>
                                        </a>
                                        <?php endif; ?>
                                    </div>
                                </div>
                            </div>
                        </div>
                    <?php endforeach; ?>
                </div>
            </div>

            <!-- Vista de tabla (oculta por defecto) -->
            <div id="tableView" class="device-view" style="display: none;">
                <div class="card">
                    <div class="card-body">
                        <table class="table table-hover table-striped" id="devicesTable">
                            <thead>
                                <tr>
                                    <th>Estado</th>
                                    <th>Dispositivo</th>
                                    <th>Tipo</th>
                                    <th>Máquina</th>
                                    <th>Última Conexión</th>
                                    <th>IP</th>
                                    <th>Alertas</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($devices as $device): ?>
                                    <?php $status_info = getStatusInfo($device['computed_status'], $device['minutes_since_last_access']); ?>
                                    <tr data-status="<?php echo $device['computed_status']; ?>" 
                                        data-device-id="<?php echo $device['device_id']; ?>">
                                        <td>
                                            <i class="fas <?php echo $status_info['icon']; ?> text-<?php echo $status_info['class']; ?>" 
                                               title="<?php echo $status_info['description']; ?>"></i>
                                            <span class="badge badge-<?php echo $status_info['class']; ?>">
                                                <?php echo $status_info['text']; ?>
                                            </span>
                                        </td>
                                        <td>
                                            <strong><?php echo htmlspecialchars($device['name'] ?: $device['device_id']); ?></strong>
                                            <br><small class="text-muted"><?php echo htmlspecialchars($device['device_id']); ?></small>
                                        </td>
                                        <td><?php echo htmlspecialchars($device['device_type']); ?></td>
                                        <td><?php echo htmlspecialchars($device['machine_name'] ?: 'No asignada'); ?></td>
                                        <td>
                                            <?php echo getLastSeenText($device['last_access'], $device['minutes_since_last_access']); ?>
                                            <?php if ($device['last_access']): ?>
                                                <br><small class="text-muted">
                                                    <?php echo date('d/m/Y H:i', strtotime($device['last_access'])); ?>
                                                </small>
                                            <?php endif; ?>
                                        </td>
                                        <td><?php echo htmlspecialchars($device['ip_address'] ?: 'N/A'); ?></td>
                                        <td>
                                            <?php if ($device['pending_alerts'] > 0): ?>
                                                <span class="badge badge-warning">
                                                    <?php echo $device['pending_alerts']; ?>
                                                </span>
                                            <?php else: ?>
                                                <span class="text-muted">0</span>
                                            <?php endif; ?>
                                            
                                            <?php if ($device['config_pending']): ?>
                                                <br><small class="badge badge-info">Config. pendiente</small>
                                            <?php endif; ?>
                                        </td>
                                        <td>
                                            <div class="btn-group btn-group-sm">
                                                <a href="view.php?id=<?php echo $device['id']; ?>" 
                                                   class="btn btn-info" title="Ver detalles">
                                                    <i class="fas fa-eye"></i>
                                                </a>
                                                <button type="button" 
                                                        class="btn btn-primary" 
                                                        onclick="testConnection('<?php echo $device['device_id']; ?>')"
                                                        title="Probar conexión">
                                                    <i class="fas fa-wifi"></i>
                                                </button>
                                            </div>
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

<!-- Modal para detalles de conexión -->
<div class="modal fade" id="connectionModal" tabindex="-1" role="dialog">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Prueba de Conexión</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="connectionResult">
                    <!-- Contenido se carga dinámicamente -->
                </div>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Inicializar DataTable para vista de tabla
    $('#devicesTable').DataTable({
        "paging": true,
        "lengthChange": true,
        "searching": true,
        "ordering": true,
        "info": true,
        "autoWidth": false,
        "responsive": true,
        "pageLength": 25,
        "order": [[4, "desc"]], // Ordenar por última conexión
        "language": {
            "url": "<?php echo ASSETS_URL; ?>/plugins/datatables/Spanish.json"
        }
    });

    // Configurar auto-refresh
    setupAutoRefresh();
    
    // Configurar filtros
    setupFilters();
    
    // Configurar cambio de vista
    setupViewToggle();
});

// Auto-refresh functionality
let refreshInterval;

function setupAutoRefresh() {
    const autoRefreshToggle = $('#autoRefresh');
    
    function startAutoRefresh() {
        refreshInterval = setInterval(function() {
            if (autoRefreshToggle.is(':checked')) {
                refreshStatus(true); // Refresh silencioso
            }
        }, 30000); // 30 segundos
    }
    
    function stopAutoRefresh() {
        if (refreshInterval) {
            clearInterval(refreshInterval);
        }
    }
    
    autoRefreshToggle.on('change', function() {
        if ($(this).is(':checked')) {
            startAutoRefresh();
        } else {
            stopAutoRefresh();
        }
    });
    
    // Iniciar auto-refresh si está habilitado
    if (autoRefreshToggle.is(':checked')) {
        startAutoRefresh();
    }
}

// Refresh status
function refreshStatus(silent = false) {
    if (!silent) {
        showLoadingIndicator();
    }
    
    $.ajax({
        url: 'get_device_status.php',
        method: 'GET',
        dataType: 'json',
        success: function(response) {
            if (response.success) {
                updateDeviceStatus(response.devices);
                updateStats(response.stats);
                if (!silent) {
                    showNotification('success', 'Estado actualizado correctamente');
                }
            } else {
                if (!silent) {
                    showNotification('error', 'Error al actualizar estado');
                }
            }
        },
        error: function() {
            if (!silent) {
                showNotification('error', 'Error de conexión al actualizar estado');
            }
        },
        complete: function() {
            hideLoadingIndicator();
        }
    });
}

// Update device status in real-time
function updateDeviceStatus(devices) {
    devices.forEach(function(device) {
        const cardElement = $(`.device-card[data-device-id="${device.device_id}"]`);
        const tableRow = $(`tr[data-device-id="${device.device_id}"]`);
        
        // Actualizar tarjeta
        if (cardElement.length) {
            updateDeviceCard(cardElement, device);
        }
        
        // Actualizar fila de tabla
        if (tableRow.length) {
            updateDeviceRow(tableRow, device);
        }
    });
}

function updateDeviceCard(cardElement, device) {
    // Actualizar clase de estado
    cardElement.attr('data-status', device.computed_status);
    
    // Actualizar badge de estado
    const statusBadge = cardElement.find('.badge');
    statusBadge.removeClass().addClass(`badge badge-${device.status_info.class}`);
    statusBadge.text(device.status_info.text);
    
    // Actualizar icono
    const statusIcon = cardElement.find('.card-title i');
    statusIcon.removeClass().addClass(`fas ${device.status_info.icon} text-${device.status_info.class}`);
    
    // Actualizar última conexión
    cardElement.find('.info-item:contains("Última conexión")').html(
        '<strong>Última conexión:</strong> ' + device.last_seen_text
    );
    
    // Actualizar IP si cambió
    if (device.ip_address) {
        cardElement.find('.info-item:contains("IP")').html(
            '<strong>IP:</strong> ' + device.ip_address
        );
    }
}

function updateDeviceRow(tableRow, device) {
    // Actualizar estado
    const statusCell = tableRow.find('td:first');
    statusCell.html(`
        <i class="fas ${device.status_info.icon} text-${device.status_info.class}" 
           title="${device.status_info.description}"></i>
        <span class="badge badge-${device.status_info.class}">
            ${device.status_info.text}
        </span>
    `);
    
    // Actualizar última conexión
    const connectionCell = tableRow.find('td:nth-child(5)');
    let connectionHtml = device.last_seen_text;
    if (device.last_access) {
        connectionHtml += `<br><small class="text-muted">${device.last_access_formatted}</small>`;
    }
    connectionCell.html(connectionHtml);
}

// Update statistics
function updateStats(stats) {
    $('.small-box .inner h3').each(function(index) {
        switch(index) {
            case 0: $(this).text(stats.total); break;
            case 1: $(this).text(stats.online); break;
            case 2: $(this).text(stats.offline + stats.error); break;
            case 3: $(this).text(stats.pending_alerts); break;
        }
    });
}

// Setup filters
function setupFilters() {
    $('#statusFilter').on('change', function() {
        const filterValue = $(this).val();
        
        if (filterValue === '') {
            $('.device-card, tr[data-device-id]').show();
        } else {
            $('.device-card, tr[data-device-id]').hide();
            $(`.device-card[data-status*="${filterValue}"], tr[data-status*="${filterValue}"]`).show();
        }
    });
}

// Setup view toggle
function setupViewToggle() {
    $('input[name="viewMode"]').on('change', function() {
        if ($(this).attr('id') === 'cardView') {
            $('#cardView').show();
            $('#tableView').hide();
        } else {
            $('#cardView').hide();
            $('#tableView').show();
        }
    });
}

// Test connection
function testConnection(deviceId) {
    $('#connectionModal').modal('show');
    $('#connectionResult').html(`
        <div class="text-center">
            <i class="fas fa-spinner fa-spin fa-2x"></i>
            <h4>Probando conexión con ${deviceId}...</h4>
        </div>
    `);
    
    $.ajax({
        url: 'test_connection.php',
        method: 'POST',
        data: { device_id: deviceId },
        success: function(response) {
            if (response.success) {
                $('#connectionResult').html(`
                    <div class="alert alert-success">
                        <h5><i class="fas fa-check-circle"></i> Conexión exitosa</h5>
                        <p><strong>Última conexión:</strong> ${response.last_seen}</p>
                        <p><strong>Estado:</strong> ${response.connection_status}</p>
                        ${response.ping_result ? 
                            `<p><strong>Ping:</strong> ${response.ping_result.success ? 
                                '✓ Responde (' + (response.ping_result.response_time || 'N/A') + ')' : 
                                '✗ No responde'}</p>` : ''}
                    </div>
                `);
            } else {
                $('#connectionResult').html(`
                    <div class="alert alert-warning">
                        <h5><i class="fas fa-exclamation-triangle"></i> ${response.message}</h5>
                        ${response.last_seen ? `<p><strong>Última conexión:</strong> ${response.last_seen}</p>` : ''}
                        ${response.recommendations ? 
                            '<h6>Recomendaciones:</h6><ul>' + 
                            response.recommendations.map(r => `<li>${r}</li>`).join('') + 
                            '</ul>' : ''}
                    </div>
                `);
            }
        },
        error: function() {
            $('#connectionResult').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-times-circle"></i>
                    Error de conexión al probar dispositivo.
                </div>
            `);
        }
    });
}

// Loading indicator
function showLoadingIndicator() {
    if (!$('#loadingIndicator').length) {
        $('body').append(`
            <div id="loadingIndicator" style="position: fixed; top: 20px; right: 20px; z-index: 9999;">
                <div class="alert alert-info">
                    <i class="fas fa-spinner fa-spin"></i> Actualizando estado...
                </div>
            </div>
        `);
    }
}

function hideLoadingIndicator() {
    $('#loadingIndicator').remove();
}

// Show notifications
function showNotification(type, message) {
    const alertClass = type === 'success' ? 'alert-success' : 'alert-danger';
    
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
.device-card {
    margin-bottom: 1rem;
}

.info-item {
    margin-bottom: 0.5rem;
    font-size: 0.9em;
}

.card-outline {
    border-width: 2px;
}

.device-view {
    margin-top: 1rem;
}

.small-box .inner h3 {
    font-weight: bold;
}

.btn-group-sm .btn {
    padding: 0.25rem 0.5rem;
    font-size: 0.875rem;
}

@keyframes pulse-online {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.text-success.fa-circle {
    animation: pulse-online 2s infinite;
}
</style>