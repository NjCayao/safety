<?php
/**
 * Script auxiliar: Obtener historial de configuración
 * server/pages/devices/get_config_history.php
 */

// Verificar sesión
session_start();
if (!isset($_SESSION['user_id'])) {
    http_response_code(401);
    exit('No autorizado');
}

// Incluir dependencias
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/device_config.php';

// Obtener device_id
$device_id = $_GET['device_id'] ?? null;

if (!$device_id) {
    echo '<div class="alert alert-danger">ID de dispositivo no especificado</div>';
    exit;
}

// Verificar que el dispositivo existe
$device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$device_id]);
if (!$device) {
    echo '<div class="alert alert-danger">Dispositivo no encontrado</div>';
    exit;
}

// Obtener historial de configuración
$history = db_fetch_all("
    SELECT dch.*, u.username as changed_by_name
    FROM device_config_history dch
    LEFT JOIN users u ON dch.changed_by = u.id
    WHERE dch.device_id = ?
    ORDER BY dch.created_at DESC
    LIMIT 50
", [$device_id]);

if (empty($history)): ?>
    <div class="alert alert-info">
        <i class="fas fa-info-circle"></i>
        No hay historial de configuración para este dispositivo.
    </div>
<?php else: ?>
    <div class="timeline">
        <?php foreach ($history as $change): ?>
            <div class="time-label">
                <span class="bg-<?php echo $change['applied_successfully'] === null ? 'warning' : ($change['applied_successfully'] ? 'success' : 'danger'); ?>">
                    <?php echo date('d M Y', strtotime($change['created_at'])); ?>
                </span>
            </div>
            
            <div>
                <i class="fas fa-<?php echo $change['applied_successfully'] === null ? 'clock' : ($change['applied_successfully'] ? 'check' : 'times'); ?> bg-<?php echo $change['applied_successfully'] === null ? 'warning' : ($change['applied_successfully'] ? 'success' : 'danger'); ?>"></i>
                
                <div class="timeline-item">
                    <span class="time">
                        <i class="fas fa-clock"></i> <?php echo date('H:i:s', strtotime($change['created_at'])); ?>
                    </span>
                    
                    <h3 class="timeline-header">
                        <span class="badge badge-<?php echo $change['change_type'] === 'manual' ? 'primary' : ($change['change_type'] === 'profile' ? 'info' : 'warning'); ?>">
                            <?php 
                            switch($change['change_type']) {
                                case 'manual': echo 'Cambio Manual'; break;
                                case 'profile': echo 'Perfil Aplicado'; break;
                                case 'rollback': echo 'Rollback'; break;
                                default: echo ucfirst($change['change_type']);
                            }
                            ?>
                        </span>
                        por <strong><?php echo htmlspecialchars($change['changed_by_name'] ?? 'Sistema'); ?></strong>
                    </h3>
                    
                    <div class="timeline-body">
                        <?php if ($change['changes_summary']): ?>
                            <p><strong>Resumen:</strong> <?php echo htmlspecialchars($change['changes_summary']); ?></p>
                        <?php endif; ?>
                        
                        <?php if ($change['applied_successfully'] === null): ?>
                            <div class="alert alert-warning alert-sm">
                                <i class="fas fa-clock"></i> Pendiente de aplicar
                            </div>
                        <?php elseif ($change['applied_successfully']): ?>
                            <div class="alert alert-success alert-sm">
                                <i class="fas fa-check"></i> 
                                Aplicado exitosamente el <?php echo date('d/m/Y H:i', strtotime($change['applied_at'])); ?>
                            </div>
                        <?php else: ?>
                            <div class="alert alert-danger alert-sm">
                                <i class="fas fa-times"></i> Error al aplicar
                                <?php if ($change['error_message']): ?>
                                    <br><small><?php echo htmlspecialchars($change['error_message']); ?></small>
                                <?php endif; ?>
                            </div>
                        <?php endif; ?>
                        
                        <!-- Botones de acción -->
                        <div class="mt-2">
                            <button type="button" class="btn btn-xs btn-outline-info" 
                                    onclick="viewConfigDetails(<?php echo $change['id']; ?>)">
                                <i class="fas fa-eye"></i> Ver Detalles
                            </button>
                            
                            <?php if ($change['applied_successfully']): ?>
                                <button type="button" class="btn btn-xs btn-outline-warning" 
                                        onclick="rollbackToConfig(<?php echo $change['id']; ?>)">
                                    <i class="fas fa-undo"></i> Rollback
                                </button>
                            <?php elseif ($change['applied_successfully'] === false): ?>
                                <button type="button" class="btn btn-xs btn-outline-primary" 
                                        onclick="retryConfig(<?php echo $change['id']; ?>)">
                                    <i class="fas fa-redo"></i> Reintentar
                                </button>
                            <?php endif; ?>
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

<!-- Modal para detalles de configuración -->
<div class="modal fade" id="configDetailsModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-lg" role="document">
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

<script>
// Ver detalles de configuración
function viewConfigDetails(historyId) {
    $('#configDetailsModal').modal('show');
    
    $.ajax({
        url: 'get_config_details.php',
        method: 'GET',
        data: { history_id: historyId },
        success: function(response) {
            $('#configDetailsContent').html(response);
        },
        error: function() {
            $('#configDetailsContent').html('<div class="alert alert-danger">Error al cargar detalles</div>');
        }
    });
}

// Rollback a configuración anterior
function rollbackToConfig(historyId) {
    if (confirm('¿Está seguro de que desea realizar rollback a esta configuración?')) {
        $.ajax({
            url: 'rollback_config.php',
            method: 'POST',
            data: { 
                history_id: historyId,
                device_id: '<?php echo $device_id; ?>'
            },
            success: function(response) {
                if (response.success) {
                    location.reload(); // Recargar historial
                } else {
                    alert('Error: ' + (response.message || 'Error al realizar rollback'));
                }
            },
            error: function() {
                alert('Error de conexión al realizar rollback');
            }
        });
    }
}

// Reintentar configuración
function retryConfig(historyId) {
    if (confirm('¿Reintentar aplicar esta configuración?')) {
        $.ajax({
            url: 'retry_config_history.php',
            method: 'POST',
            data: { 
                history_id: historyId,
                device_id: '<?php echo $device_id; ?>'
            },
            success: function(response) {
                if (response.success) {
                    location.reload(); // Recargar historial
                } else {
                    alert('Error: ' + (response.message || 'Error al reintentar configuración'));
                }
            },
            error: function() {
                alert('Error de conexión al reintentar configuración');
            }
        });
    }
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

.timeline > div > .fa {
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

.alert-sm {
    padding: 0.375rem 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.875rem;
}
</style>