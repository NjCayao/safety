<?php
/**
 * Script auxiliar: Obtener detalles de configuración específica
 * server/pages/devices/get_config_details.php
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

// Obtener history_id
$history_id = $_GET['history_id'] ?? null;

if (!$history_id) {
    echo '<div class="alert alert-danger">ID de historial no especificado</div>';
    exit;
}

// Obtener detalles del historial de configuración
$history_record = db_fetch_one("
    SELECT dch.*, u.username as changed_by_name, d.device_id, d.name as device_name
    FROM device_config_history dch
    LEFT JOIN users u ON dch.changed_by = u.id
    LEFT JOIN devices d ON dch.device_id = d.device_id
    WHERE dch.id = ?
", [$history_id]);

if (!$history_record): ?>
    <div class="alert alert-danger">
        <i class="fas fa-exclamation-triangle"></i>
        Registro de historial no encontrado.
    </div>
<?php else: ?>
    <div class="row">
        <!-- Información general -->
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h5><i class="fas fa-info-circle"></i> Información General</h5>
                </div>
                <div class="card-body">
                    <dl class="row">
                        <dt class="col-sm-3">Dispositivo:</dt>
                        <dd class="col-sm-9">
                            <strong><?php echo htmlspecialchars($history_record['device_name'] ?: $history_record['device_id']); ?></strong>
                            <br><small class="text-muted"><?php echo htmlspecialchars($history_record['device_id']); ?></small>
                        </dd>
                        
                        <dt class="col-sm-3">Fecha/Hora:</dt>
                        <dd class="col-sm-9"><?php echo date('d/m/Y H:i:s', strtotime($history_record['created_at'])); ?></dd>
                        
                        <dt class="col-sm-3">Modificado por:</dt>
                        <dd class="col-sm-9"><?php echo htmlspecialchars($history_record['changed_by_name'] ?: 'Sistema'); ?></dd>
                        
                        <dt class="col-sm-3">Tipo de cambio:</dt>
                        <dd class="col-sm-9">
                            <span class="badge badge-<?php echo $history_record['change_type'] === 'manual' ? 'primary' : ($history_record['change_type'] === 'profile' ? 'info' : 'warning'); ?>">
                                <?php 
                                switch($history_record['change_type']) {
                                    case 'manual': echo 'Cambio Manual'; break;
                                    case 'profile': echo 'Perfil Aplicado'; break;
                                    case 'rollback': echo 'Rollback'; break;
                                    case 'reset': echo 'Reset a Defecto'; break;
                                    default: echo ucfirst($history_record['change_type']);
                                }
                                ?>
                            </span>
                        </dd>
                        
                        <dt class="col-sm-3">Estado:</dt>
                        <dd class="col-sm-9">
                            <?php if ($history_record['applied_successfully'] === null): ?>
                                <span class="badge badge-warning">
                                    <i class="fas fa-clock"></i> Pendiente de aplicar
                                </span>
                            <?php elseif ($history_record['applied_successfully']): ?>
                                <span class="badge badge-success">
                                    <i class="fas fa-check"></i> Aplicado exitosamente
                                </span>
                                <br><small class="text-muted">
                                    Aplicado el <?php echo date('d/m/Y H:i', strtotime($history_record['applied_at'])); ?>
                                </small>
                            <?php else: ?>
                                <span class="badge badge-danger">
                                    <i class="fas fa-times"></i> Error al aplicar
                                </span>
                                <?php if ($history_record['error_message']): ?>
                                    <br><small class="text-danger">
                                        <?php echo htmlspecialchars($history_record['error_message']); ?>
                                    </small>
                                <?php endif; ?>
                            <?php endif; ?>
                        </dd>
                        
                        <?php if ($history_record['changes_summary']): ?>
                        <dt class="col-sm-3">Resumen:</dt>
                        <dd class="col-sm-9"><?php echo htmlspecialchars($history_record['changes_summary']); ?></dd>
                        <?php endif; ?>
                    </dl>
                </div>
            </div>
        </div>
    </div>

    <!-- Comparación de configuraciones -->
    <div class="row mt-3">
        <div class="col-md-12">
            <ul class="nav nav-tabs" id="configTabs" role="tablist">
                <?php if ($history_record['config_before']): ?>
                <li class="nav-item">
                    <a class="nav-link active" id="before-tab" data-toggle="tab" href="#before" role="tab">
                        <i class="fas fa-history"></i> Configuración Anterior
                    </a>
                </li>
                <?php endif; ?>
                
                <li class="nav-item">
                    <a class="nav-link <?php echo !$history_record['config_before'] ? 'active' : ''; ?>" id="after-tab" data-toggle="tab" href="#after" role="tab">
                        <i class="fas fa-cog"></i> Nueva Configuración
                    </a>
                </li>
                
                <?php if ($history_record['config_before']): ?>
                <li class="nav-item">
                    <a class="nav-link" id="diff-tab" data-toggle="tab" href="#diff" role="tab">
                        <i class="fas fa-exchange-alt"></i> Comparación
                    </a>
                </li>
                <?php endif; ?>
            </ul>
            
            <div class="tab-content" id="configTabContent">
                <?php if ($history_record['config_before']): ?>
                <!-- Configuración anterior -->
                <div class="tab-pane fade show active" id="before" role="tabpanel">
                    <div class="card">
                        <div class="card-header">
                            <h6><i class="fas fa-history"></i> Configuración Anterior</h6>
                        </div>
                        <div class="card-body">
                            <?php
                            $config_before = json_decode($history_record['config_before'], true);
                            if ($config_before):
                                echo generateConfigDisplay($config_before);
                            else:
                                echo '<p class="text-muted">No hay configuración anterior disponible</p>';
                            endif;
                            ?>
                        </div>
                    </div>
                </div>
                <?php endif; ?>
                
                <!-- Nueva configuración -->
                <div class="tab-pane fade <?php echo !$history_record['config_before'] ? 'show active' : ''; ?>" id="after" role="tabpanel">
                    <div class="card">
                        <div class="card-header">
                            <h6><i class="fas fa-cog"></i> Nueva Configuración</h6>
                        </div>
                        <div class="card-body">
                            <?php
                            $config_after = json_decode($history_record['config_after'], true);
                            if ($config_after):
                                echo generateConfigDisplay($config_after);
                            else:
                                echo '<p class="text-muted">No hay configuración disponible</p>';
                            endif;
                            ?>
                        </div>
                    </div>
                </div>
                
                <?php if ($history_record['config_before']): ?>
                <!-- Comparación -->
                <div class="tab-pane fade" id="diff" role="tabpanel">
                    <div class="card">
                        <div class="card-header">
                            <h6><i class="fas fa-exchange-alt"></i> Comparación de Cambios</h6>
                        </div>
                        <div class="card-body">
                            <?php
                            $config_before = json_decode($history_record['config_before'], true);
                            $config_after = json_decode($history_record['config_after'], true);
                            echo generateConfigComparison($config_before ?: [], $config_after ?: []);
                            ?>
                        </div>
                    </div>
                </div>
                <?php endif; ?>
            </div>
        </div>
    </div>

    <!-- Acciones disponibles -->
    <div class="row mt-3">
        <div class="col-md-12">
            <div class="card">
                <div class="card-header">
                    <h6><i class="fas fa-tools"></i> Acciones Disponibles</h6>
                </div>
                <div class="card-body">
                    <div class="btn-group" role="group">
                        <?php if ($history_record['applied_successfully']): ?>
                            <button type="button" class="btn btn-warning" onclick="rollbackToThis(<?php echo $history_id; ?>)">
                                <i class="fas fa-undo"></i> Hacer Rollback a esta Configuración
                            </button>
                        <?php elseif ($history_record['applied_successfully'] === false): ?>
                            <button type="button" class="btn btn-primary" onclick="retryConfiguration(<?php echo $history_id; ?>)">
                                <i class="fas fa-redo"></i> Reintentar Aplicación
                            </button>
                        <?php endif; ?>
                        
                        <button type="button" class="btn btn-info" onclick="exportConfiguration(<?php echo $history_id; ?>)">
                            <i class="fas fa-download"></i> Exportar Configuración
                        </button>
                        
                        <button type="button" class="btn btn-secondary" onclick="copyToClipboard()">
                            <i class="fas fa-copy"></i> Copiar JSON
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
<?php endif; ?>

<script>
function rollbackToThis(historyId) {
    if (confirm('¿Está seguro de que desea hacer rollback a esta configuración?')) {
        // Cerrar modal actual y ejecutar rollback desde la página padre
        $('#configDetailsModal').modal('hide');
        if (typeof rollbackToConfig === 'function') {
            rollbackToConfig(historyId);
        }
    }
}

function retryConfiguration(historyId) {
    if (confirm('¿Reintentar aplicar esta configuración?')) {
        // Cerrar modal y ejecutar retry desde la página padre
        $('#configDetailsModal').modal('hide');
        if (typeof retryConfig === 'function') {
            retryConfig(historyId);
        }
    }
}

function exportConfiguration(historyId) {
    window.open('export_config_history.php?history_id=' + historyId, '_blank');
}

function copyToClipboard() {
    // Obtener configuración actual en formato JSON
    const configAfter = <?php echo json_encode($history_record['config_after'] ?? '{}'); ?>;
    
    // Crear elemento temporal para copiar
    const temp = document.createElement('textarea');
    temp.value = JSON.stringify(JSON.parse(configAfter), null, 2);
    document.body.appendChild(temp);
    temp.select();
    document.execCommand('copy');
    document.body.removeChild(temp);
    
    // Mostrar notificación
    showNotification('success', 'Configuración copiada al portapapeles');
}

// Función para mostrar notificaciones (debe estar definida en la página padre)
function showNotification(type, message) {
    if (typeof parent.showNotification === 'function') {
        parent.showNotification(type, message);
    } else {
        alert(message);
    }
}
</script>

<?php
/**
 * Genera display HTML para una configuración
 */
function generateConfigDisplay($config, $level = 0) {
    if (!is_array($config)) {
        return '<p class="text-muted">Configuración inválida</p>';
    }
    
    $html = '<div class="config-display">';
    
    foreach ($config as $key => $value) {
        $indent = str_repeat('  ', $level);
        $html .= '<div class="config-item" style="margin-left: ' . ($level * 20) . 'px;">';
        
        if (is_array($value)) {
            $html .= '<strong class="text-primary">' . htmlspecialchars($key) . ':</strong>';
            $html .= generateConfigDisplay($value, $level + 1);
        } else {
            $html .= '<span class="text-secondary">' . htmlspecialchars($key) . ':</span> ';
            
            // Formatear valor según tipo
            if (is_bool($value)) {
                $badge_class = $value ? 'badge-success' : 'badge-secondary';
                $text = $value ? 'true' : 'false';
                $html .= '<span class="badge ' . $badge_class . '">' . $text . '</span>';
            } elseif (is_numeric($value)) {
                $html .= '<span class="badge badge-info">' . htmlspecialchars($value) . '</span>';
            } else {
                $html .= '<span class="text-dark">' . htmlspecialchars($value) . '</span>';
            }
        }
        
        $html .= '</div>';
    }
    
    $html .= '</div>';
    return $html;
}

/**
 * Genera comparación entre dos configuraciones
 */
function generateConfigComparison($config_before, $config_after) {
    $changes = findConfigChanges($config_before, $config_after);
    
    if (empty($changes)) {
        return '<p class="text-muted">No se detectaron cambios</p>';
    }
    
    $html = '<div class="config-comparison">';
    
    foreach ($changes as $change) {
        $html .= '<div class="change-item mb-2 p-2 border-left border-' . $change['type'] . '">';
        $html .= '<strong>' . htmlspecialchars($change['path']) . '</strong><br>';
        
        switch ($change['type']) {
            case 'success': // changed
                $html .= '<small class="text-muted">Cambiado:</small><br>';
                $html .= '<span class="text-danger">- ' . formatValue($change['old_value']) . '</span><br>';
                $html .= '<span class="text-success">+ ' . formatValue($change['new_value']) . '</span>';
                break;
                
            case 'info': // added
                $html .= '<small class="text-muted">Agregado:</small><br>';
                $html .= '<span class="text-success">+ ' . formatValue($change['new_value']) . '</span>';
                break;
                
            case 'warning': // removed
                $html .= '<small class="text-muted">Eliminado:</small><br>';
                $html .= '<span class="text-danger">- ' . formatValue($change['old_value']) . '</span>';
                break;
        }
        
        $html .= '</div>';
    }
    
    $html .= '</div>';
    return $html;
}

/**
 * Encuentra cambios entre dos configuraciones
 */
function findConfigChanges($old_config, $new_config, $path = '') {
    $changes = [];
    
    // Verificar cambios y agregados
    foreach ($new_config as $key => $new_value) {
        $current_path = $path ? $path . '.' . $key : $key;
        
        if (!array_key_exists($key, $old_config)) {
            // Agregado
            $changes[] = [
                'type' => 'info',
                'path' => $current_path,
                'old_value' => null,
                'new_value' => $new_value
            ];
        } elseif (is_array($new_value) && is_array($old_config[$key])) {
            // Recursivo para arrays
            $sub_changes = findConfigChanges($old_config[$key], $new_value, $current_path);
            $changes = array_merge($changes, $sub_changes);
        } elseif ($old_config[$key] !== $new_value) {
            // Cambiado
            $changes[] = [
                'type' => 'success',
                'path' => $current_path,
                'old_value' => $old_config[$key],
                'new_value' => $new_value
            ];
        }
    }
    
    // Verificar eliminados
    foreach ($old_config as $key => $old_value) {
        $current_path = $path ? $path . '.' . $key : $key;
        
        if (!array_key_exists($key, $new_config)) {
            $changes[] = [
                'type' => 'warning',
                'path' => $current_path,
                'old_value' => $old_value,
                'new_value' => null
            ];
        }
    }
    
    return $changes;
}

/**
 * Formatea un valor para display
 */
function formatValue($value) {
    if ($value === null) {
        return '<em>null</em>';
    } elseif (is_bool($value)) {
        return $value ? 'true' : 'false';
    } elseif (is_array($value)) {
        return '[Array con ' . count($value) . ' elementos]';
    } else {
        return htmlspecialchars($value);
    }
}
?>

<style>
.config-display {
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 0.25rem;
    padding: 1rem;
    max-height: 400px;
    overflow-y: auto;
}

.config-item {
    margin-bottom: 0.25rem;
    line-height: 1.4;
}

.config-comparison .change-item {
    background-color: #f8f9fa;
    border-radius: 0.25rem;
}

.config-comparison .border-success {
    border-left: 4px solid #28a745 !important;
}

.config-comparison .border-info {
    border-left: 4px solid #17a2b8 !important;
}

.config-comparison .border-warning {
    border-left: 4px solid #ffc107 !important;
}

.nav-tabs .nav-link {
    font-size: 0.9em;
}

.tab-content {
    border: 1px solid #dee2e6;
    border-top: none;
    background-color: white;
}
</style>