<?php
/**
 * Página de historial de configuraciones de dispositivos
 * server/pages/devices/history.php
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
$pageTitle = "Historial de Configuraciones";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener parámetros de filtro
$device_filter = $_GET['device'] ?? '';
$user_filter = $_GET['user'] ?? '';
$change_type_filter = $_GET['change_type'] ?? '';
$date_from = $_GET['date_from'] ?? '';
$date_to = $_GET['date_to'] ?? '';
$page = max(1, (int)($_GET['page'] ?? 1));
$per_page = 25;
$offset = ($page - 1) * $per_page;

// Construir consulta de historial con filtros
$where_conditions = [];
$params = [];

if ($device_filter) {
    $where_conditions[] = "(d.device_id LIKE ? OR d.name LIKE ?)";
    $params[] = "%$device_filter%";
    $params[] = "%$device_filter%";
}

if ($user_filter) {
    $where_conditions[] = "u.username LIKE ?";
    $params[] = "%$user_filter%";
}

if ($change_type_filter) {
    $where_conditions[] = "dch.change_type = ?";
    $params[] = $change_type_filter;
}

if ($date_from) {
    $where_conditions[] = "DATE(dch.created_at) >= ?";
    $params[] = $date_from;
}

if ($date_to) {
    $where_conditions[] = "DATE(dch.created_at) <= ?";
    $params[] = $date_to;
}

$where_clause = !empty($where_conditions) ? 'WHERE ' . implode(' AND ', $where_conditions) : '';

// Obtener historial con filtros aplicados
$history_query = "
    SELECT dch.*, 
           d.device_id, 
           d.name as device_name,
           d.device_type,
           u.username as changed_by_name,
           u.name as changed_by_full_name
    FROM device_config_history dch
    LEFT JOIN devices d ON dch.device_id = d.device_id
    LEFT JOIN users u ON dch.changed_by = u.id
    $where_clause
    ORDER BY dch.created_at DESC
    LIMIT $per_page OFFSET $offset
";

$history = db_fetch_all($history_query, $params);

// Obtener total de registros para paginación
$count_query = "
    SELECT COUNT(*) as total
    FROM device_config_history dch
    LEFT JOIN devices d ON dch.device_id = d.device_id
    LEFT JOIN users u ON dch.changed_by = u.id
    $where_clause
";

$total_records = db_fetch_one($count_query, $params)['total'];
$total_pages = ceil($total_records / $per_page);

// Obtener listas para filtros
$devices = db_fetch_all("SELECT DISTINCT device_id, name FROM devices ORDER BY device_id");
$users = db_fetch_all("SELECT DISTINCT username, name FROM users ORDER BY username");

// Función para obtener el icono según el tipo de cambio
function getChangeTypeIcon($change_type) {
    switch ($change_type) {
        case 'manual':
            return ['icon' => 'fa-edit', 'class' => 'text-primary', 'label' => 'Manual'];
        case 'profile':
            return ['icon' => 'fa-layer-group', 'class' => 'text-info', 'label' => 'Perfil'];
        case 'rollback':
            return ['icon' => 'fa-undo', 'class' => 'text-warning', 'label' => 'Rollback'];
        case 'reset':
            return ['icon' => 'fa-refresh', 'class' => 'text-secondary', 'label' => 'Reset'];
        default:
            return ['icon' => 'fa-question', 'class' => 'text-muted', 'label' => ucfirst($change_type)];
    }
}

// Función para obtener el estado de aplicación
function getApplicationStatus($applied_successfully, $applied_at, $error_message) {
    if ($applied_successfully === null) {
        return [
            'icon' => 'fa-clock',
            'class' => 'warning',
            'text' => 'Pendiente',
            'description' => 'Esperando aplicación en dispositivo'
        ];
    } elseif ($applied_successfully) {
        return [
            'icon' => 'fa-check-circle',
            'class' => 'success',
            'text' => 'Aplicado',
            'description' => 'Aplicado el ' . date('d/m/Y H:i', strtotime($applied_at))
        ];
    } else {
        return [
            'icon' => 'fa-times-circle',
            'class' => 'danger',
            'text' => 'Error',
            'description' => 'Error: ' . ($error_message ?: 'Error desconocido')
        ];
    }
}

// Función para calcular tiempo relativo
function timeAgo($datetime) {
    $time = time() - strtotime($datetime);
    
    if ($time < 60) return 'Hace menos de 1 minuto';
    if ($time < 3600) return 'Hace ' . floor($time/60) . ' minutos';
    if ($time < 86400) return 'Hace ' . floor($time/3600) . ' horas';
    if ($time < 2592000) return 'Hace ' . floor($time/86400) . ' días';
    
    return date('d/m/Y', strtotime($datetime));
}
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Historial de Configuraciones</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Historial</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            
            <!-- Estadísticas rápidas -->
            <div class="row">
                <?php
                $stats = [
                    'total' => $total_records,
                    'pending' => 0,
                    'applied' => 0,
                    'errors' => 0
                ];
                
                foreach ($history as $record) {
                    if ($record['applied_successfully'] === null) $stats['pending']++;
                    elseif ($record['applied_successfully']) $stats['applied']++;
                    else $stats['errors']++;
                }
                ?>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-info">
                        <div class="inner">
                            <h3><?php echo $stats['total']; ?></h3>
                            <p>Total Cambios</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-history"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-success">
                        <div class="inner">
                            <h3><?php echo $stats['applied']; ?></h3>
                            <p>Aplicados</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-check"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-warning">
                        <div class="inner">
                            <h3><?php echo $stats['pending']; ?></h3>
                            <p>Pendientes</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-clock"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-danger">
                        <div class="inner">
                            <h3><?php echo $stats['errors']; ?></h3>
                            <p>Con Errores</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-exclamation-triangle"></i>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Filtros -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Filtros de Búsqueda</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-tool" data-card-widget="collapse">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <form method="GET" action="" id="filterForm">
                        <div class="row">
                            <div class="col-md-3">
                                <div class="form-group">
                                    <label>Dispositivo:</label>
                                    <select class="form-control select2" name="device">
                                        <option value="">Todos los dispositivos</option>
                                        <?php foreach ($devices as $device): ?>
                                            <option value="<?php echo htmlspecialchars($device['device_id']); ?>" 
                                                    <?php echo $device_filter === $device['device_id'] ? 'selected' : ''; ?>>
                                                <?php echo htmlspecialchars($device['name'] ?: $device['device_id']); ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="form-group">
                                    <label>Usuario:</label>
                                    <select class="form-control select2" name="user">
                                        <option value="">Todos los usuarios</option>
                                        <?php foreach ($users as $user): ?>
                                            <option value="<?php echo htmlspecialchars($user['username']); ?>" 
                                                    <?php echo $user_filter === $user['username'] ? 'selected' : ''; ?>>
                                                <?php echo htmlspecialchars($user['name'] . ' (' . $user['username'] . ')'); ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="form-group">
                                    <label>Tipo de Cambio:</label>
                                    <select class="form-control" name="change_type">
                                        <option value="">Todos los tipos</option>
                                        <option value="manual" <?php echo $change_type_filter === 'manual' ? 'selected' : ''; ?>>Manual</option>
                                        <option value="profile" <?php echo $change_type_filter === 'profile' ? 'selected' : ''; ?>>Perfil</option>
                                        <option value="rollback" <?php echo $change_type_filter === 'rollback' ? 'selected' : ''; ?>>Rollback</option>
                                        <option value="reset" <?php echo $change_type_filter === 'reset' ? 'selected' : ''; ?>>Reset</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="col-md-3">
                                <div class="form-group">
                                    <label>Rango de Fechas:</label>
                                    <div class="input-group">
                                        <input type="date" class="form-control" name="date_from" 
                                               value="<?php echo htmlspecialchars($date_from); ?>">
                                        <div class="input-group-append input-group-prepend">
                                            <span class="input-group-text">a</span>
                                        </div>
                                        <input type="date" class="form-control" name="date_to" 
                                               value="<?php echo htmlspecialchars($date_to); ?>">
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-12">
                                <button type="submit" class="btn btn-primary">
                                    <i class="fas fa-search"></i> Buscar
                                </button>
                                <a href="history.php" class="btn btn-secondary">
                                    <i class="fas fa-times"></i> Limpiar Filtros
                                </a>
                                <button type="button" class="btn btn-info" onclick="exportHistory()">
                                    <i class="fas fa-download"></i> Exportar
                                </button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>

            <!-- Historial -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">
                        Historial de Cambios 
                        <?php if ($total_records > 0): ?>
                            <span class="badge badge-info"><?php echo $total_records; ?> registros</span>
                        <?php endif; ?>
                    </h3>
                </div>
                
                <div class="card-body p-0">
                    <?php if (empty($history)): ?>
                        <div class="text-center p-4">
                            <i class="fas fa-history fa-3x text-muted mb-3"></i>
                            <h4 class="text-muted">No se encontraron registros</h4>
                            <p class="text-muted">
                                <?php if (!empty($where_conditions)): ?>
                                    Intente modificar los filtros de búsqueda.
                                <?php else: ?>
                                    No hay historial de configuraciones disponible.
                                <?php endif; ?>
                            </p>
                        </div>
                    <?php else: ?>
                        <div class="timeline timeline-inverse">
                            <?php 
                            $current_date = '';
                            foreach ($history as $record): 
                                $record_date = date('Y-m-d', strtotime($record['created_at']));
                                
                                // Mostrar separador de fecha
                                if ($current_date !== $record_date):
                                    $current_date = $record_date;
                            ?>
                                <div class="time-label">
                                    <span class="bg-primary">
                                        <?php echo date('d M Y', strtotime($record['created_at'])); ?>
                                    </span>
                                </div>
                            <?php endif; ?>
                            
                            <?php 
                                $change_type_info = getChangeTypeIcon($record['change_type']);
                                $app_status = getApplicationStatus($record['applied_successfully'], $record['applied_at'], $record['error_message']);
                            ?>
                            
                            <div>
                                <i class="fas <?php echo $change_type_info['icon']; ?> bg-<?php echo $app_status['class']; ?>"></i>
                                
                                <div class="timeline-item">
                                    <span class="time">
                                        <i class="fas fa-clock"></i> <?php echo date('H:i:s', strtotime($record['created_at'])); ?>
                                    </span>
                                    
                                    <h3 class="timeline-header">
                                        <i class="fas <?php echo $change_type_info['icon']; ?> <?php echo $change_type_info['class']; ?>"></i>
                                        <strong><?php echo $change_type_info['label']; ?></strong>
                                        en 
                                        <a href="view.php?device_id=<?php echo $record['device_id']; ?>">
                                            <?php echo htmlspecialchars($record['device_name'] ?: $record['device_id']); ?>
                                        </a>
                                        por 
                                        <strong><?php echo htmlspecialchars($record['changed_by_full_name'] ?: $record['changed_by_name'] ?: 'Sistema'); ?></strong>
                                    </h3>
                                    
                                    <div class="timeline-body">
                                        <div class="row">
                                            <div class="col-md-8">
                                                <?php if ($record['changes_summary']): ?>
                                                    <p><strong>Resumen:</strong> <?php echo htmlspecialchars($record['changes_summary']); ?></p>
                                                <?php endif; ?>
                                                
                                                <div class="mb-2">
                                                    <span class="badge badge-<?php echo $app_status['class']; ?>">
                                                        <i class="fas <?php echo $app_status['icon']; ?>"></i>
                                                        <?php echo $app_status['text']; ?>
                                                    </span>
                                                    <small class="text-muted ml-2">
                                                        <?php echo $app_status['description']; ?>
                                                    </small>
                                                </div>
                                                
                                                <div class="device-info">
                                                    <small class="text-muted">
                                                        <strong>Dispositivo:</strong> <?php echo htmlspecialchars($record['device_id']); ?> 
                                                        (<?php echo htmlspecialchars($record['device_type']); ?>)
                                                    </small>
                                                </div>
                                            </div>
                                            
                                            <div class="col-md-4 text-right">
                                                <div class="btn-group btn-group-sm">
                                                    <button type="button" class="btn btn-outline-info" 
                                                            onclick="viewConfigDetails(<?php echo $record['id']; ?>)">
                                                        <i class="fas fa-eye"></i> Ver
                                                    </button>
                                                    
                                                    <?php if ($record['applied_successfully']): ?>
                                                        <button type="button" class="btn btn-outline-warning" 
                                                                onclick="rollbackToConfig(<?php echo $record['id']; ?>, '<?php echo htmlspecialchars($record['device_id']); ?>')">
                                                            <i class="fas fa-undo"></i> Rollback
                                                        </button>
                                                    <?php elseif ($record['applied_successfully'] === false): ?>
                                                        <button type="button" class="btn btn-outline-primary" 
                                                                onclick="retryConfig(<?php echo $record['id']; ?>, '<?php echo htmlspecialchars($record['device_id']); ?>')">
                                                            <i class="fas fa-redo"></i> Reintentar
                                                        </button>
                                                    <?php endif; ?>
                                                    
                                                    <button type="button" class="btn btn-outline-secondary" 
                                                            onclick="exportSingleConfig(<?php echo $record['id']; ?>)">
                                                        <i class="fas fa-download"></i>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <?php endforeach; ?>
                            
                            <div>
                                <i class="fas fa-flag bg-gray"></i>
                            </div>
                        </div>
                    <?php endif; ?>
                </div>
                
                <?php if ($total_pages > 1): ?>
                <div class="card-footer">
                    <nav aria-label="Navegación del historial">
                        <ul class="pagination justify-content-center mb-0">
                            <?php if ($page > 1): ?>
                                <li class="page-item">
                                    <a class="page-link" href="?<?php echo http_build_query(array_merge($_GET, ['page' => $page - 1])); ?>">
                                        <i class="fas fa-chevron-left"></i> Anterior
                                    </a>
                                </li>
                            <?php endif; ?>
                            
                            <?php
                            $start_page = max(1, $page - 2);
                            $end_page = min($total_pages, $page + 2);
                            
                            for ($i = $start_page; $i <= $end_page; $i++):
                            ?>
                                <li class="page-item <?php echo $i === $page ? 'active' : ''; ?>">
                                    <a class="page-link" href="?<?php echo http_build_query(array_merge($_GET, ['page' => $i])); ?>">
                                        <?php echo $i; ?>
                                    </a>
                                </li>
                            <?php endfor; ?>
                            
                            <?php if ($page < $total_pages): ?>
                                <li class="page-item">
                                    <a class="page-link" href="?<?php echo http_build_query(array_merge($_GET, ['page' => $page + 1])); ?>">
                                        Siguiente <i class="fas fa-chevron-right"></i>
                                    </a>
                                </li>
                            <?php endif; ?>
                        </ul>
                    </nav>
                    
                    <div class="text-center mt-2">
                        <small class="text-muted">
                            Página <?php echo $page; ?> de <?php echo $total_pages; ?> 
                            (<?php echo $total_records; ?> registros totales)
                        </small>
                    </div>
                </div>
                <?php endif; ?>
            </div>
        </div>
    </section>
</div>

<!-- Modal para detalles de configuración -->
<div class="modal fade" id="configDetailsModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-xl" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Detalles de Configuración</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="configDetailsContent">
                    <!-- Contenido se carga dinámicamente -->
                </div>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Inicializar Select2 para filtros
    $('.select2').select2({
        theme: 'bootstrap4',
        placeholder: 'Seleccionar...',
        allowClear: true
    });
});

// Ver detalles de configuración
function viewConfigDetails(historyId) {
    $('#configDetailsModal').modal('show');
    $('#configDetailsContent').html(`
        <div class="text-center">
            <i class="fas fa-spinner fa-spin fa-2x"></i>
            <h4>Cargando detalles...</h4>
        </div>
    `);
    
    $.ajax({
        url: 'get_config_details.php',
        method: 'GET',
        data: { history_id: historyId },
        success: function(response) {
            $('#configDetailsContent').html(response);
        },
        error: function() {
            $('#configDetailsContent').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error al cargar los detalles de configuración.
                </div>
            `);
        }
    });
}

// Rollback a configuración
function rollbackToConfig(historyId, deviceId) {
    if (confirm('¿Está seguro de que desea realizar rollback a esta configuración?')) {
        $.ajax({
            url: 'rollback_config.php',
            method: 'POST',
            data: { 
                history_id: historyId,
                device_id: deviceId
            },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Rollback realizado exitosamente');
                    setTimeout(() => location.reload(), 2000);
                } else {
                    showNotification('error', response.message || 'Error al realizar rollback');
                }
            },
            error: function() {
                showNotification('error', 'Error de conexión al realizar rollback');
            }
        });
    }
}

// Reintentar configuración
function retryConfig(historyId, deviceId) {
    if (confirm('¿Reintentar aplicar esta configuración?')) {
        $.ajax({
            url: 'retry_config_history.php',
            method: 'POST',
            data: { 
                history_id: historyId,
                device_id: deviceId
            },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Configuración reenviada correctamente');
                    setTimeout(() => location.reload(), 2000);
                } else {
                    showNotification('error', response.message || 'Error al reintentar configuración');
                }
            },
            error: function() {
                showNotification('error', 'Error de conexión al reintentar configuración');
            }
        });
    }
}

// Exportar historial completo
function exportHistory() {
    const form = document.getElementById('filterForm');
    const formData = new FormData(form);
    formData.append('export', '1');
    
    const params = new URLSearchParams();
    for (let [key, value] of formData.entries()) {
        if (value) params.append(key, value);
    }
    
    window.open('export_history.php?' + params.toString(), '_blank');
}

// Exportar configuración específica
function exportSingleConfig(historyId) {
    window.open('export_config_history.php?history_id=' + historyId, '_blank');
}

// Mostrar notificaciones
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
.timeline {
    position: relative;
    margin: 0 0 30px 0;
    padding: 0;
    list-style: none;
}

.timeline:before {
    content: '';
    position: absolute;
    top: 0;
    bottom: 0;
    width: 4px;
    background: #dee2e6;
    left: 31px;
    margin: 0;
    border-radius: 2px;
}

.timeline .time-label span {
    font-weight: 600;
    color: #fff;
    font-size: 12px;
    padding: 5px 10px;
    display: inline-block;
    border-radius: 4px;
}

.timeline .timeline-item {
    background: #fff;
    border-radius: 3px;
    width: calc(100% - 50px);
    margin-left: 50px;
    margin-top: 10px;
    color: #444;
    padding: 0;
    position: relative;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
}

.timeline .timeline-item > .time {
    color: #999;
    float: right;
    padding: 10px;
    font-size: 12px;
}

.timeline .timeline-item > .timeline-header {
    margin: 0;
    color: #555;
    border-bottom: 1px solid #f4f4f4;
    padding: 10px;
    font-size: 16px;
    line-height: 1.1;
}

.timeline .timeline-item > .timeline-body {
    padding: 10px;
}

.timeline > div > .fa,
.timeline > div > .fas {
    position: absolute;
    left: 18px;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    text-align: center;
    line-height: 30px;
    font-size: 15px;
    color: #fff;
}

.device-info {
    margin-top: 0.5rem;
    padding-top: 0.5rem;
    border-top: 1px solid #f0f0f0;
}

.select2-container--bootstrap4 .select2-selection {
    border-color: #ced4da;
}

.small-box .inner h3 {
    font-weight: bold;
}

.btn-group-sm .btn {
    padding: 0.25rem 0.5rem;
    font-size: 0.875rem;
}

.pagination .page-link {
    color: #007bff;
}

.pagination .page-item.active .page-link {
    background-color: #007bff;
    border-color: #007bff;
}
</style>