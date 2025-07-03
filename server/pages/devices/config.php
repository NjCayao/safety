<?php
/**
 * Página de configuración de dispositivos
 * server/pages/devices/config.php
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

// Verificar permisos (solo admin y supervisor)
if (!isset($_SESSION['user_role']) || !in_array($_SESSION['user_role'], ['admin', 'supervisor'])) {
    $_SESSION['error_message'] = 'No tiene permisos para acceder a esta sección.';
    header('Location: ../dashboard/index.php');
    exit();
}

// Definir el título de la página
$pageTitle = "Configuración de Dispositivos";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener todos los dispositivos con su estado de configuración
$devices = DeviceConfigManager::getAllDevicesConfigStatus();

// Función para obtener el icono y clase según el estado
function getStatusIcon($status) {
    switch ($status) {
        case 'synchronized':
            return ['icon' => 'fas fa-check-circle', 'class' => 'text-success', 'text' => 'Sincronizado'];
        case 'pending':
            return ['icon' => 'fas fa-clock', 'class' => 'text-warning', 'text' => 'Pendiente'];
        case 'sync_error':
            return ['icon' => 'fas fa-exclamation-triangle', 'class' => 'text-danger', 'text' => 'Error Sync'];
        case 'offline':
            return ['icon' => 'fas fa-power-off', 'class' => 'text-secondary', 'text' => 'Offline'];
        case 'never_configured':
            return ['icon' => 'fas fa-question-circle', 'class' => 'text-info', 'text' => 'Sin Configurar'];
        default:
            return ['icon' => 'fas fa-question', 'class' => 'text-muted', 'text' => 'Desconocido'];
    }
}

// Función para calcular tiempo relativo
function timeAgo($datetime) {
    if (!$datetime) return 'Nunca';
    
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
                    <h1>Configuración de Dispositivos</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Configuración</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            <!-- Resumen de estados -->
            <div class="row">
                <?php
                $stats = [
                    'total' => count($devices),
                    'synchronized' => 0,
                    'pending' => 0,
                    'offline' => 0,
                    'errors' => 0
                ];
                
                foreach ($devices as $device) {
                    switch ($device['config_status']) {
                        case 'synchronized':
                            $stats['synchronized']++;
                            break;
                        case 'pending':
                            $stats['pending']++;
                            break;
                        case 'offline':
                            $stats['offline']++;
                            break;
                        case 'sync_error':
                        case 'never_configured':
                            $stats['errors']++;
                            break;
                    }
                }
                ?>
                
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
                            <h3><?php echo $stats['synchronized']; ?></h3>
                            <p>Sincronizados</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-check-circle"></i>
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
                            <h3><?php echo $stats['offline'] + $stats['errors']; ?></h3>
                            <p>Offline/Errores</p>
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
                    <h3 class="card-title">Filtros</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-tool" data-card-widget="collapse">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Estado de Configuración:</label>
                                <select class="form-control" id="filterStatus">
                                    <option value="">Todos</option>
                                    <option value="synchronized">Sincronizados</option>
                                    <option value="pending">Pendientes</option>
                                    <option value="sync_error">Con Errores</option>
                                    <option value="offline">Offline</option>
                                    <option value="never_configured">Sin Configurar</option>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="form-group">
                                <label>Tipo de Dispositivo:</label>
                                <select class="form-control" id="filterType">
                                    <option value="">Todos</option>
                                    <?php
                                    $device_types = array_unique(array_column($devices, 'device_type'));
                                    foreach ($device_types as $type) {
                                        if ($type) echo "<option value=\"{$type}\">{$type}</option>";
                                    }
                                    ?>
                                </select>
                            </div>
                        </div>
                        <div class="col-md-4">
                            <div class="form-group">
                                <label>Buscar:</label>
                                <input type="text" class="form-control" id="searchBox" placeholder="Nombre o ID del dispositivo">
                            </div>
                        </div>
                        <div class="col-md-2">
                            <div class="form-group">
                                <label>&nbsp;</label>
                                <button type="button" class="btn btn-primary btn-block" onclick="refreshDeviceList()">
                                    <i class="fas fa-sync"></i> Actualizar
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Lista de dispositivos -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Dispositivos Registrados</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-success btn-sm" onclick="applyBulkConfiguration()">
                            <i class="fas fa-cogs"></i> Configuración Masiva
                        </button>
                    </div>
                </div>
                
                <div class="card-body table-responsive p-0">
                    <table class="table table-hover text-nowrap" id="devicesTable">
                        <thead>
                            <tr>
                                <th>
                                    <div class="custom-control custom-checkbox">
                                        <input class="custom-control-input" type="checkbox" id="selectAll">
                                        <label for="selectAll" class="custom-control-label"></label>
                                    </div>
                                </th>
                                <th>Dispositivo</th>
                                <th>Tipo</th>
                                <th>Máquina Asignada</th>
                                <th>Estado Config.</th>
                                <th>Versión</th>
                                <th>Última Actualización</th>
                                <th>Acciones</th>
                            </tr>
                        </thead>
                        <tbody>
                            <?php foreach ($devices as $device): ?>
                                <?php $status = getStatusIcon($device['config_status']); ?>
                                <tr data-status="<?php echo $device['config_status']; ?>" 
                                    data-type="<?php echo htmlspecialchars($device['device_type']); ?>"
                                    data-name="<?php echo htmlspecialchars($device['device_name']); ?>">
                                    <td>
                                        <div class="custom-control custom-checkbox">
                                            <input class="custom-control-input device-checkbox" 
                                                   type="checkbox" 
                                                   id="device_<?php echo $device['device_id']; ?>"
                                                   value="<?php echo $device['device_id']; ?>">
                                            <label for="device_<?php echo $device['device_id']; ?>" class="custom-control-label"></label>
                                        </div>
                                    </td>
                                    <td>
                                        <div>
                                            <strong><?php echo htmlspecialchars($device['device_name'] ?: $device['device_id']); ?></strong>
                                            <br>
                                            <small class="text-muted"><?php echo htmlspecialchars($device['device_id']); ?></small>
                                        </div>
                                    </td>
                                    <td><?php echo htmlspecialchars($device['device_type']); ?></td>
                                    <td>
                                        <?php if ($device['machine_id']): ?>
                                            <a href="../machines/view.php?id=<?php echo $device['machine_id']; ?>">
                                                <?php echo htmlspecialchars($device['machine_name'] ?? $device['machine_id']); ?>
                                            </a>
                                        <?php else: ?>
                                            <span class="text-muted">Sin asignar</span>
                                        <?php endif; ?>
                                    </td>
                                    <td>
                                        <i class="<?php echo $status['icon']; ?> <?php echo $status['class']; ?>"></i>
                                        <?php echo $status['text']; ?>
                                        <?php if ($device['config_pending']): ?>
                                            <br><small class="text-warning">
                                                <i class="fas fa-exclamation-circle"></i> Cambios pendientes
                                            </small>
                                        <?php endif; ?>
                                    </td>
                                    <td>
                                        <span class="badge badge-info">v<?php echo $device['config_version']; ?></span>
                                    </td>
                                    <td>
                                        <?php if ($device['config_applied']): ?>
                                            <span title="<?php echo date('d/m/Y H:i:s', strtotime($device['config_applied'])); ?>">
                                                <?php echo timeAgo($device['config_applied']); ?>
                                            </span>
                                        <?php else: ?>
                                            <span class="text-muted">Nunca</span>
                                        <?php endif; ?>
                                        
                                        <?php if ($device['last_change_summary']): ?>
                                            <br><small class="text-muted">
                                                <?php echo htmlspecialchars($device['last_change_summary']); ?>
                                            </small>
                                        <?php endif; ?>
                                    </td>
                                    <td>
                                        <div class="btn-group">
                                            <a href="configure.php?id=<?php echo $device['device_id']; ?>" 
                                               class="btn btn-primary btn-sm" 
                                               title="Configurar">
                                                <i class="fas fa-cog"></i>
                                            </a>
                                            
                                            <button type="button" 
                                                    class="btn btn-info btn-sm" 
                                                    onclick="viewConfigHistory('<?php echo $device['device_id']; ?>')"
                                                    title="Ver Historial">
                                                <i class="fas fa-history"></i>
                                            </button>
                                            
                                            <?php if ($device['config_status'] === 'sync_error'): ?>
                                            <button type="button" 
                                                    class="btn btn-warning btn-sm" 
                                                    onclick="retryConfiguration('<?php echo $device['device_id']; ?>')"
                                                    title="Reintentar Configuración">
                                                <i class="fas fa-redo"></i>
                                            </button>
                                            <?php endif; ?>
                                            
                                            <button type="button" 
                                                    class="btn btn-secondary btn-sm dropdown-toggle" 
                                                    data-toggle="dropdown"
                                                    title="Más Opciones">
                                                <i class="fas fa-ellipsis-v"></i>
                                            </button>
                                            <div class="dropdown-menu">
                                                <a class="dropdown-item" href="#" onclick="duplicateConfig('<?php echo $device['device_id']; ?>')">
                                                    <i class="fas fa-copy"></i> Duplicar Configuración
                                                </a>
                                                <a class="dropdown-item" href="#" onclick="resetToDefault('<?php echo $device['device_id']; ?>')">
                                                    <i class="fas fa-undo"></i> Restaurar por Defecto
                                                </a>
                                                <div class="dropdown-divider"></div>
                                                <a class="dropdown-item" href="#" onclick="exportConfig('<?php echo $device['device_id']; ?>')">
                                                    <i class="fas fa-download"></i> Exportar Configuración
                                                </a>
                                            </div>
                                        </div>
                                    </td>
                                </tr>
                            <?php endforeach; ?>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </section>
</div>

<!-- Modal para configuración masiva -->
<div class="modal fade" id="bulkConfigModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-lg" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Configuración Masiva</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div class="form-group">
                    <label>Aplicar Perfil a Dispositivos Seleccionados:</label>
                    <select class="form-control" id="bulkProfileSelect">
                        <option value="">Seleccione un perfil...</option>
                        <?php
                        $profiles = DeviceConfigManager::getConfigProfiles();
                        foreach ($profiles as $profile) {
                            echo "<option value=\"{$profile['id']}\">{$profile['name']} - {$profile['description']}</option>";
                        }
                        ?>
                    </select>
                </div>
                <div id="selectedDevicesList"></div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <button type="button" class="btn btn-primary" onclick="executeBulkConfiguration()">
                    <i class="fas fa-cogs"></i> Aplicar Configuración
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Modal para historial de configuración -->
<div class="modal fade" id="configHistoryModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-xl" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Historial de Configuración</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="configHistoryContent">
                    <div class="text-center">
                        <i class="fas fa-spinner fa-spin"></i> Cargando historial...
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Inicializar DataTable
    $('#devicesTable').DataTable({
        "paging": true,
        "lengthChange": true,
        "searching": false, // Usamos búsqueda personalizada
        "ordering": true,
        "info": true,
        "autoWidth": false,
        "responsive": true,
        "pageLength": 25,
        "order": [[1, "asc"]],
        "columnDefs": [
            { "orderable": false, "targets": [0, 7] } // Checkbox y acciones no ordenables
        ],
        "language": {
            "url": "<?php echo ASSETS_URL; ?>/plugins/datatables/Spanish.json"
        }
    });

    // Configurar filtros personalizados
    setupCustomFilters();
    
    // Auto-refresh cada 30 segundos
    setInterval(function() {
        if ($('#autoRefresh').is(':checked')) {
            refreshDeviceList();
        }
    }, 30000);
    
    // Agregar checkbox de auto-refresh
    $('.card-tools').first().prepend(`
        <div class="custom-control custom-switch mr-2 d-inline-block">
            <input type="checkbox" class="custom-control-input" id="autoRefresh" checked>
            <label class="custom-control-label" for="autoRefresh">Auto-actualizar</label>
        </div>
    `);
});

// Configurar filtros personalizados
function setupCustomFilters() {
    const table = $('#devicesTable').DataTable();
    
    // Filtro por estado
    $('#filterStatus').on('change', function() {
        const status = $(this).val();
        if (status) {
            table.column(4).search(status, true, false).draw();
        } else {
            table.column(4).search('').draw();
        }
    });
    
    // Filtro por tipo
    $('#filterType').on('change', function() {
        const type = $(this).val();
        if (type) {
            table.column(2).search('^' + type + '$', true, false).draw();
        } else {
            table.column(2).search('').draw();
        }
    });
    
    // Búsqueda personalizada
    $('#searchBox').on('keyup', function() {
        table.search($(this).val()).draw();
    });
}

// Seleccionar/deseleccionar todos los dispositivos
$('#selectAll').on('change', function() {
    $('.device-checkbox').prop('checked', $(this).is(':checked'));
});

// Actualizar lista de dispositivos
function refreshDeviceList() {
    // Mostrar indicador de carga
    const refreshBtn = $('button:contains("Actualizar")');
    const originalHtml = refreshBtn.html();
    refreshBtn.html('<i class="fas fa-spinner fa-spin"></i> Actualizando...');
    refreshBtn.prop('disabled', true);
    
    // Recargar página (en una implementación real, usarías AJAX)
    setTimeout(function() {
        location.reload();
    }, 1000);
}

// Ver historial de configuración
function viewConfigHistory(deviceId) {
    $('#configHistoryModal').modal('show');
    
    // Cargar historial via AJAX
    $.ajax({
        url: 'get_config_history.php',
        method: 'GET',
        data: { device_id: deviceId },
        success: function(response) {
            $('#configHistoryContent').html(response);
        },
        error: function() {
            $('#configHistoryContent').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error al cargar el historial de configuración.
                </div>
            `);
        }
    });
}

// Reintentar configuración
function retryConfiguration(deviceId) {
    if (confirm('¿Está seguro de que desea reintentar la configuración para este dispositivo?')) {
        $.ajax({
            url: 'retry_config.php',
            method: 'POST',
            data: { device_id: deviceId },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Configuración reenviada correctamente');
                    setTimeout(refreshDeviceList, 2000);
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

// Configuración masiva
function applyBulkConfiguration() {
    const selectedDevices = $('.device-checkbox:checked');
    
    if (selectedDevices.length === 0) {
        showNotification('warning', 'Debe seleccionar al menos un dispositivo');
        return;
    }
    
    // Crear lista de dispositivos seleccionados
    let deviceList = '<ul>';
    selectedDevices.each(function() {
        const row = $(this).closest('tr');
        const deviceName = row.find('td:eq(1) strong').text();
        deviceList += `<li>${deviceName}</li>`;
    });
    deviceList += '</ul>';
    
    $('#selectedDevicesList').html(`
        <div class="alert alert-info">
            <strong>Dispositivos seleccionados (${selectedDevices.length}):</strong>
            ${deviceList}
        </div>
    `);
    
    $('#bulkConfigModal').modal('show');
}

// Ejecutar configuración masiva
function executeBulkConfiguration() {
    const profileId = $('#bulkProfileSelect').val();
    const selectedDevices = $('.device-checkbox:checked').map(function() {
        return $(this).val();
    }).get();
    
    if (!profileId) {
        showNotification('warning', 'Debe seleccionar un perfil de configuración');
        return;
    }
    
    if (selectedDevices.length === 0) {
        showNotification('warning', 'No hay dispositivos seleccionados');
        return;
    }
    
    // Mostrar progreso
    const modalBody = $('#bulkConfigModal .modal-body');
    modalBody.html(`
        <div class="text-center">
            <i class="fas fa-spinner fa-spin fa-2x"></i>
            <h4>Aplicando configuración...</h4>
            <div class="progress mt-3">
                <div class="progress-bar" role="progressbar" style="width: 0%"></div>
            </div>
        </div>
    `);
    
    // Aplicar configuración a cada dispositivo
    applyBulkConfigRecursive(selectedDevices, profileId, 0);
}

// Aplicar configuración masiva recursivamente
function applyBulkConfigRecursive(devices, profileId, index) {
    if (index >= devices.length) {
        // Completado
        $('#bulkConfigModal .modal-body').html(`
            <div class="alert alert-success text-center">
                <i class="fas fa-check-circle fa-2x"></i>
                <h4>Configuración aplicada exitosamente</h4>
                <p>Se ha aplicado la configuración a ${devices.length} dispositivos.</p>
            </div>
        `);
        
        setTimeout(function() {
            $('#bulkConfigModal').modal('hide');
            refreshDeviceList();
        }, 2000);
        return;
    }
    
    const deviceId = devices[index];
    const progress = ((index + 1) / devices.length) * 100;
    
    // Actualizar barra de progreso
    $('.progress-bar').css('width', progress + '%').text(Math.round(progress) + '%');
    
    // Aplicar configuración
    $.ajax({
        url: 'apply_profile.php',
        method: 'POST',
        data: {
            device_id: deviceId,
            profile_id: profileId
        },
        success: function(response) {
            // Continuar con el siguiente dispositivo
            applyBulkConfigRecursive(devices, profileId, index + 1);
        },
        error: function() {
            // Manejar error pero continuar
            applyBulkConfigRecursive(devices, profileId, index + 1);
        }
    });
}

// Duplicar configuración
function duplicateConfig(sourceDeviceId) {
    const targetDevices = $('.device-checkbox:checked').not(`#device_${sourceDeviceId}`);
    
    if (targetDevices.length === 0) {
        showNotification('warning', 'Seleccione los dispositivos de destino');
        return;
    }
    
    if (confirm(`¿Copiar la configuración de este dispositivo a ${targetDevices.length} dispositivos seleccionados?`)) {
        const targetIds = targetDevices.map(function() { return $(this).val(); }).get();
        
        $.ajax({
            url: 'duplicate_config.php',
            method: 'POST',
            data: {
                source_device_id: sourceDeviceId,
                target_device_ids: targetIds
            },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Configuración duplicada correctamente');
                    setTimeout(refreshDeviceList, 2000);
                } else {
                    showNotification('error', response.message || 'Error al duplicar configuración');
                }
            },
            error: function() {
                showNotification('error', 'Error de conexión al duplicar configuración');
            }
        });
    }
}

// Restaurar configuración por defecto
function resetToDefault(deviceId) {
    if (confirm('¿Está seguro de que desea restaurar la configuración por defecto? Se perderán todos los ajustes personalizados.')) {
        $.ajax({
            url: 'reset_config.php',
            method: 'POST',
            data: { device_id: deviceId },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Configuración restaurada por defecto');
                    setTimeout(refreshDeviceList, 2000);
                } else {
                    showNotification('error', response.message || 'Error al restaurar configuración');
                }
            },
            error: function() {
                showNotification('error', 'Error de conexión al restaurar configuración');
            }
        });
    }
}

// Exportar configuración
function exportConfig(deviceId) {
    window.open(`export_config.php?device_id=${deviceId}`, '_blank');
}

// Mostrar notificaciones
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
    
    // Auto-remove después de 5 segundos
    setTimeout(function() {
        notification.alert('close');
    }, 5000);
}
</script>

<style>
.table th {
    border-top: none;
    font-weight: 600;
}

.device-checkbox {
    cursor: pointer;
}

.btn-group .dropdown-menu {
    min-width: 200px;
}

.config-status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 5px;
}

.status-synchronized { background-color: #28a745; }
.status-pending { background-color: #ffc107; }
.status-error { background-color: #dc3545; }
.status-offline { background-color: #6c757d; }

/* Animación para filas actualizadas */
@keyframes highlightUpdate {
    from { background-color: rgba(255, 193, 7, 0.3); }
    to { background-color: transparent; }
}

.row-updated {
    animation: highlightUpdate 2s ease-out;
}
</style>