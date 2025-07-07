<?php
/**
 * Página de aplicación masiva de perfiles
 * server/pages/devices/apply_profile_bulk.php
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

// Obtener perfil seleccionado (si viene de profiles.php)
$selected_profile_id = $_GET['profile_id'] ?? null;

// Obtener todos los perfiles disponibles
$profiles = db_fetch_all("
    SELECT id, name, description, device_type, is_default,
           (SELECT COUNT(*) FROM devices d WHERE JSON_EXTRACT(d.config_json, '$.profile_id') = dcp.id) as current_usage
    FROM device_config_profiles dcp
    ORDER BY is_default DESC, name ASC
");

// Obtener todos los dispositivos disponibles
$devices = db_fetch_all("
    SELECT d.*, 
           m.name as machine_name,
           m.location as machine_location,
           JSON_EXTRACT(d.config_json, '$.profile_id') as current_profile_id,
           dcp.name as current_profile_name
    FROM devices d
    LEFT JOIN machines m ON d.machine_id = m.id
    LEFT JOIN device_config_profiles dcp ON JSON_EXTRACT(d.config_json, '$.profile_id') = dcp.id
    ORDER BY d.status DESC, d.device_id ASC
");

// Procesar aplicación masiva
$success_message = '';
$error_message = '';
$application_results = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['action']) && $_POST['action'] === 'apply_bulk') {
    $result = handleBulkApplication();
    if ($result['success']) {
        $success_message = $result['message'];
        $application_results = $result['results'];
    } else {
        $error_message = $result['error'];
    }
}

function handleBulkApplication() {
    $profile_id = (int)$_POST['profile_id'];
    $device_ids = $_POST['device_ids'] ?? [];
    $force_apply = isset($_POST['force_apply']);
    
    if (!$profile_id) {
        return ['success' => false, 'error' => 'Debe seleccionar un perfil'];
    }
    
    if (empty($device_ids)) {
        return ['success' => false, 'error' => 'Debe seleccionar al menos un dispositivo'];
    }
    
    // Verificar que el perfil existe
    $profile = db_fetch_one("SELECT * FROM device_config_profiles WHERE id = ?", [$profile_id]);
    if (!$profile) {
        return ['success' => false, 'error' => 'Perfil no encontrado'];
    }
    
    $results = [];
    $success_count = 0;
    $error_count = 0;
    $skipped_count = 0;
    
    foreach ($device_ids as $device_id) {
        try {
            // Verificar que el dispositivo existe
            $device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$device_id]);
            if (!$device) {
                $results[] = [
                    'device_id' => $device_id,
                    'status' => 'error',
                    'message' => 'Dispositivo no encontrado'
                ];
                $error_count++;
                continue;
            }
            
            // Verificar compatibilidad de tipo de dispositivo
            if ($profile['device_type'] && $profile['device_type'] !== $device['device_type']) {
                if (!$force_apply) {
                    $results[] = [
                        'device_id' => $device_id,
                        'device_name' => $device['name'],
                        'status' => 'skipped',
                        'message' => "Tipo incompatible ({$device['device_type']} vs {$profile['device_type']})"
                    ];
                    $skipped_count++;
                    continue;
                }
            }
            
            // Verificar si ya tiene este perfil aplicado
            $current_profile_id = json_decode($device['config_json'], true)['profile_id'] ?? null;
            if ($current_profile_id == $profile_id && !$force_apply) {
                $results[] = [
                    'device_id' => $device_id,
                    'device_name' => $device['name'],
                    'status' => 'skipped',
                    'message' => 'Ya tiene este perfil aplicado'
                ];
                $skipped_count++;
                continue;
            }
            
            // Aplicar perfil
            $apply_result = DeviceConfigManager::applyConfigProfile(
                $device_id, 
                $profile_id, 
                $_SESSION['user_id']
            );
            
            if ($apply_result['success']) {
                $results[] = [
                    'device_id' => $device_id,
                    'device_name' => $device['name'],
                    'status' => 'success',
                    'message' => 'Perfil aplicado correctamente',
                    'history_id' => $apply_result['history_id']
                ];
                $success_count++;
            } else {
                $results[] = [
                    'device_id' => $device_id,
                    'device_name' => $device['name'],
                    'status' => 'error',
                    'message' => $apply_result['error']
                ];
                $error_count++;
            }
            
        } catch (Exception $e) {
            $results[] = [
                'device_id' => $device_id,
                'status' => 'error',
                'message' => 'Error: ' . $e->getMessage()
            ];
            $error_count++;
        }
    }
    
    // Registrar operación en logs
    db_insert('system_logs', [
        'log_type' => 'info',
        'message' => "Aplicación masiva del perfil '{$profile['name']}' por " . ($_SESSION['username'] ?? 'usuario'),
        'details' => json_encode([
            'profile_id' => $profile_id,
            'profile_name' => $profile['name'],
            'total_devices' => count($device_ids),
            'success_count' => $success_count,
            'error_count' => $error_count,
            'skipped_count' => $skipped_count,
            'user' => $_SESSION['username'] ?? 'N/A'
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);
    
    // Preparar mensaje de resultado
    $total = count($device_ids);
    if ($success_count === $total) {
        $message = "Perfil aplicado exitosamente a todos los {$total} dispositivos";
    } elseif ($success_count > 0) {
        $message = "Aplicación completada: {$success_count} exitosos, {$error_count} errores, {$skipped_count} omitidos";
    } else {
        return [
            'success' => false, 
            'error' => "No se pudo aplicar el perfil a ningún dispositivo. {$error_count} errores, {$skipped_count} omitidos."
        ];
    }
    
    return [
        'success' => true,
        'message' => $message,
        'results' => $results,
        'summary' => [
            'total' => $total,
            'success' => $success_count,
            'errors' => $error_count,
            'skipped' => $skipped_count
        ]
    ];
}

// Definir el título de la página
$pageTitle = "Aplicación Masiva de Perfiles";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Aplicación Masiva de Perfiles</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item"><a href="profiles.php">Perfiles</a></li>
                        <li class="breadcrumb-item active">Aplicación Masiva</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            
            <!-- Mensajes -->
            <?php if ($success_message): ?>
                <div class="alert alert-success alert-dismissible fade show">
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                    <i class="fas fa-check-circle"></i> <?php echo htmlspecialchars($success_message); ?>
                </div>
            <?php endif; ?>
            
            <?php if ($error_message): ?>
                <div class="alert alert-danger alert-dismissible fade show">
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                    <i class="fas fa-exclamation-triangle"></i> <?php echo htmlspecialchars($error_message); ?>
                </div>
            <?php endif; ?>

            <!-- Información sobre aplicación masiva -->
            <div class="card card-info">
                <div class="card-header">
                    <h3 class="card-title">¿Qué es la aplicación masiva de perfiles?</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-tool" data-card-widget="collapse">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <p>La <strong>aplicación masiva</strong> permite configurar múltiples dispositivos simultáneamente usando un perfil predefinido, ahorrando tiempo y garantizando consistencia.</p>
                    
                    <div class="row">
                        <div class="col-md-4">
                            <h6><i class="fas fa-lightbulb text-warning"></i> Casos de uso:</h6>
                            <ul class="small">
                                <li>Cambio de turno (diurno → nocturno)</li>
                                <li>Implementar nuevas políticas de seguridad</li>
                                <li>Optimización de rendimiento masiva</li>
                                <li>Configuración inicial de nuevos dispositivos</li>
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <h6><i class="fas fa-shield-alt text-success"></i> Verificaciones automáticas:</h6>
                            <ul class="small">
                                <li>Compatibilidad de tipo de dispositivo</li>
                                <li>Estado de conectividad</li>
                                <li>Validación de configuración</li>
                                <li>Registro completo en historial</li>
                            </ul>
                        </div>
                        <div class="col-md-4">
                            <h6><i class="fas fa-chart-line text-info"></i> Seguimiento:</h6>
                            <ul class="small">
                                <li>Progreso en tiempo real</li>
                                <li>Reporte detallado de resultados</li>
                                <li>Rollback individual si es necesario</li>
                                <li>Notificaciones de estado</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Resultados de aplicación (si existen) -->
            <?php if (!empty($application_results)): ?>
                <div class="card">
                    <div class="card-header">
                        <h3 class="card-title">Resultados de la Aplicación</h3>
                    </div>
                    <div class="card-body">
                        <div class="row mb-3">
                            <div class="col-md-3">
                                <div class="info-box bg-success">
                                    <span class="info-box-icon"><i class="fas fa-check"></i></span>
                                    <div class="info-box-content">
                                        <span class="info-box-text">Exitosos</span>
                                        <span class="info-box-number"><?php echo array_sum(array_map(function($r) { return $r['status'] === 'success' ? 1 : 0; }, $application_results)); ?></span>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="info-box bg-danger">
                                    <span class="info-box-icon"><i class="fas fa-times"></i></span>
                                    <div class="info-box-content">
                                        <span class="info-box-text">Errores</span>
                                        <span class="info-box-number"><?php echo array_sum(array_map(function($r) { return $r['status'] === 'error' ? 1 : 0; }, $application_results)); ?></span>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="info-box bg-warning">
                                    <span class="info-box-icon"><i class="fas fa-exclamation"></i></span>
                                    <div class="info-box-content">
                                        <span class="info-box-text">Omitidos</span>
                                        <span class="info-box-number"><?php echo array_sum(array_map(function($r) { return $r['status'] === 'skipped' ? 1 : 0; }, $application_results)); ?></span>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <div class="info-box bg-info">
                                    <span class="info-box-icon"><i class="fas fa-microchip"></i></span>
                                    <div class="info-box-content">
                                        <span class="info-box-text">Total</span>
                                        <span class="info-box-number"><?php echo count($application_results); ?></span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="table-responsive">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Dispositivo</th>
                                        <th>Estado</th>
                                        <th>Mensaje</th>
                                        <th>Acciones</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <?php foreach ($application_results as $result): ?>
                                        <tr>
                                            <td>
                                                <strong><?php echo htmlspecialchars($result['device_name'] ?: $result['device_id']); ?></strong>
                                                <br><small class="text-muted"><?php echo htmlspecialchars($result['device_id']); ?></small>
                                            </td>
                                            <td>
                                                <?php
                                                $badge_class = [
                                                    'success' => 'badge-success',
                                                    'error' => 'badge-danger', 
                                                    'skipped' => 'badge-warning'
                                                ][$result['status']] ?? 'badge-secondary';
                                                
                                                $icon = [
                                                    'success' => 'fa-check',
                                                    'error' => 'fa-times',
                                                    'skipped' => 'fa-exclamation'
                                                ][$result['status']] ?? 'fa-question';
                                                ?>
                                                <span class="badge <?php echo $badge_class; ?>">
                                                    <i class="fas <?php echo $icon; ?>"></i>
                                                    <?php echo ucfirst($result['status'] === 'skipped' ? 'Omitido' : ($result['status'] === 'success' ? 'Exitoso' : 'Error')); ?>
                                                </span>
                                            </td>
                                            <td><?php echo htmlspecialchars($result['message']); ?></td>
                                            <td>
                                                <?php if ($result['status'] === 'success' && isset($result['history_id'])): ?>
                                                    <button type="button" class="btn btn-xs btn-outline-info" 
                                                            onclick="viewConfigHistory(<?php echo $result['history_id']; ?>)">
                                                        <i class="fas fa-history"></i>
                                                    </button>
                                                <?php elseif ($result['status'] === 'error'): ?>
                                                    <button type="button" class="btn btn-xs btn-outline-warning" 
                                                            onclick="retryDevice('<?php echo $result['device_id']; ?>')">
                                                        <i class="fas fa-redo"></i>
                                                    </button>
                                                <?php endif; ?>
                                            </td>
                                        </tr>
                                    <?php endforeach; ?>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            <?php endif; ?>

            <!-- Formulario principal -->
            <form method="POST" id="bulkApplicationForm">
                <input type="hidden" name="action" value="apply_bulk">
                
                <div class="row">
                    <!-- Selección de perfil -->
                    <div class="col-md-4">
                        <div class="card card-primary">
                            <div class="card-header">
                                <h3 class="card-title">1. Seleccionar Perfil</h3>
                            </div>
                            <div class="card-body">
                                <div class="form-group">
                                    <label>Perfil de Configuración:</label>
                                    <select class="form-control" name="profile_id" id="profileSelect" required onchange="updateProfileInfo()">
                                        <option value="">Seleccione un perfil...</option>
                                        <?php foreach ($profiles as $profile): ?>
                                            <option value="<?php echo $profile['id']; ?>" 
                                                    data-description="<?php echo htmlspecialchars($profile['description']); ?>"
                                                    data-device-type="<?php echo htmlspecialchars($profile['device_type']); ?>"
                                                    data-usage="<?php echo $profile['current_usage']; ?>"
                                                    <?php echo ($selected_profile_id == $profile['id']) ? 'selected' : ''; ?>>
                                                <?php echo htmlspecialchars($profile['name']); ?>
                                                <?php if ($profile['is_default']): ?>
                                                    (Por defecto)
                                                <?php endif; ?>
                                                <?php if ($profile['device_type']): ?>
                                                    - <?php echo htmlspecialchars($profile['device_type']); ?>
                                                <?php endif; ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                                
                                <div id="profileInfo" style="display: none;">
                                    <div class="alert alert-info">
                                        <h6><i class="fas fa-info-circle"></i> Información del Perfil</h6>
                                        <div id="profileDescription"></div>
                                        <div id="profileDetails"></div>
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <div class="custom-control custom-checkbox">
                                        <input type="checkbox" class="custom-control-input" id="force_apply" name="force_apply">
                                        <label class="custom-control-label" for="force_apply">
                                            Forzar aplicación
                                        </label>
                                    </div>
                                    <small class="form-text text-muted">
                                        Aplicar incluso si el tipo de dispositivo no coincide o ya tiene este perfil
                                    </small>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Selección de dispositivos -->
                    <div class="col-md-8">
                        <div class="card card-success">
                            <div class="card-header">
                                <h3 class="card-title">2. Seleccionar Dispositivos</h3>
                                <div class="card-tools">
                                    <span id="selectedCount" class="badge badge-light">0 seleccionados</span>
                                </div>
                            </div>
                            <div class="card-body">
                                <!-- Filtros de dispositivos -->
                                <div class="row mb-3">
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>Filtrar por Estado:</label>
                                            <select class="form-control form-control-sm" id="statusFilter">
                                                <option value="">Todos</option>
                                                <option value="online">En línea</option>
                                                <option value="offline">Fuera de línea</option>
                                                <option value="error">Con errores</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>Filtrar por Tipo:</label>
                                            <select class="form-control form-control-sm" id="typeFilter">
                                                <option value="">Todos</option>
                                                <option value="Raspberry Pi">Raspberry Pi</option>
                                                <option value="Edge Computer">Edge Computer</option>
                                                <option value="Industrial PC">Industrial PC</option>
                                            </select>
                                        </div>
                                    </div>
                                    <div class="col-md-4">
                                        <div class="form-group">
                                            <label>Acciones rápidas:</label>
                                            <div>
                                                <button type="button" class="btn btn-sm btn-outline-primary" onclick="selectAll()">
                                                    Seleccionar Todos
                                                </button>
                                                <button type="button" class="btn btn-sm btn-outline-secondary" onclick="selectNone()">
                                                    Deseleccionar
                                                </button>
                                                <button type="button" class="btn btn-sm btn-outline-info" onclick="selectCompatible()">
                                                    Solo Compatibles
                                                </button>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <!-- Lista de dispositivos -->
                                <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
                                    <table class="table table-sm table-hover">
                                        <thead class="thead-light sticky-top">
                                            <tr>
                                                <th width="40">
                                                    <div class="custom-control custom-checkbox">
                                                        <input type="checkbox" class="custom-control-input" id="selectAllDevices" onchange="toggleAllDevices()">
                                                        <label class="custom-control-label" for="selectAllDevices"></label>
                                                    </div>
                                                </th>
                                                <th>Dispositivo</th>
                                                <th>Tipo</th>
                                                <th>Estado</th>
                                                <th>Máquina</th>
                                                <th>Perfil Actual</th>
                                                <th>Compatibilidad</th>
                                            </tr>
                                        </thead>
                                        <tbody id="devicesTableBody">
                                            <?php foreach ($devices as $device): ?>
                                                <tr class="device-row" 
                                                    data-device-id="<?php echo $device['device_id']; ?>"
                                                    data-device-type="<?php echo $device['device_type']; ?>"
                                                    data-status="<?php echo $device['status']; ?>"
                                                    data-current-profile="<?php echo $device['current_profile_id'] ?: '0'; ?>">
                                                    <td>
                                                        <div class="custom-control custom-checkbox">
                                                            <input type="checkbox" class="custom-control-input device-checkbox" 
                                                                   id="device_<?php echo $device['device_id']; ?>"
                                                                   name="device_ids[]" 
                                                                   value="<?php echo $device['device_id']; ?>"
                                                                   onchange="updateSelectedCount()">
                                                            <label class="custom-control-label" for="device_<?php echo $device['device_id']; ?>"></label>
                                                        </div>
                                                    </td>
                                                    <td>
                                                        <div>
                                                            <strong><?php echo htmlspecialchars($device['name'] ?: $device['device_id']); ?></strong>
                                                            <br><small class="text-muted"><?php echo htmlspecialchars($device['device_id']); ?></small>
                                                        </div>
                                                    </td>
                                                    <td>
                                                        <span class="badge badge-light"><?php echo htmlspecialchars($device['device_type']); ?></span>
                                                    </td>
                                                    <td>
                                                        <?php
                                                        $status_class = [
                                                            'online' => 'success',
                                                            'offline' => 'danger',
                                                            'error' => 'danger',
                                                            'syncing' => 'warning'
                                                        ][$device['status']] ?? 'secondary';
                                                        ?>
                                                        <span class="badge badge-<?php echo $status_class; ?>">
                                                            <?php echo ucfirst($device['status']); ?>
                                                        </span>
                                                    </td>
                                                    <td>
                                                        <small><?php echo htmlspecialchars($device['machine_name'] ?: 'No asignada'); ?></small>
                                                    </td>
                                                    <td>
                                                        <small><?php echo htmlspecialchars($device['current_profile_name'] ?: 'Sin perfil'); ?></small>
                                                    </td>
                                                    <td>
                                                        <span class="compatibility-indicator" data-device-type="<?php echo $device['device_type']; ?>">
                                                            <i class="fas fa-question text-muted"></i>
                                                        </span>
                                                    </td>
                                                </tr>
                                            <?php endforeach; ?>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Botón de aplicación -->
                <div class="card">
                    <div class="card-body text-center">
                        <button type="submit" class="btn btn-primary btn-lg" id="applyButton" disabled>
                            <i class="fas fa-magic"></i> Aplicar Perfil a Dispositivos Seleccionados
                        </button>
                        <br><small class="text-muted mt-2 d-block">
                            Asegúrese de haber seleccionado el perfil y dispositivos correctos antes de continuar
                        </small>
                    </div>
                </div>
            </form>
        </div>
    </section>
</div>

<!-- Modal de confirmación -->
<div class="modal fade" id="confirmationModal" tabindex="-1" role="dialog">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Confirmar Aplicación Masiva</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="confirmationContent">
                    <!-- Contenido se genera dinámicamente -->
                </div>
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                <button type="button" class="btn btn-primary" onclick="executeApplication()">
                    <i class="fas fa-magic"></i> Confirmar Aplicación
                </button>
            </div>
        </div>
    </div>
</div>

<!-- Modal de progreso -->
<div class="modal fade" id="progressModal" tabindex="-1" role="dialog" data-backdrop="static">
    <div class="modal-dialog" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Aplicando Configuración...</h5>
            </div>
            <div class="modal-body text-center">
                <div class="progress mb-3">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                         role="progressbar" style="width: 0%"></div>
                </div>
                <div id="progressText">Preparando aplicación...</div>
                <div id="progressDetails" class="mt-3 small text-muted"></div>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Configurar filtros
    setupDeviceFilters();
    
    // Actualizar información del perfil si hay uno preseleccionado
    if ($('#profileSelect').val()) {
        updateProfileInfo();
    }
    
    // Configurar validación del formulario
    setupFormValidation();
});

// Actualizar información del perfil seleccionado
function updateProfileInfo() {
    const select = document.getElementById('profileSelect');
    const option = select.options[select.selectedIndex];
    
    if (!option.value) {
        $('#profileInfo').hide();
        updateCompatibilityIndicators(null);
        return;
    }
    
    const description = option.getAttribute('data-description');
    const deviceType = option.getAttribute('data-device-type');
    const usage = option.getAttribute('data-usage');
    
    let infoHtml = '';
    if (description) {
        infoHtml += `<p><strong>Descripción:</strong> ${description}</p>`;
    }
    
    infoHtml += `<ul class="mb-0">`;
    infoHtml += `<li><strong>Tipo compatible:</strong> ${deviceType || 'Universal (todos los tipos)'}</li>`;
    infoHtml += `<li><strong>Dispositivos actuales:</strong> ${usage} dispositivos usando este perfil</li>`;
    infoHtml += `</ul>`;
    
    $('#profileDescription').html(infoHtml);
    $('#profileInfo').show();
    
    // Actualizar indicadores de compatibilidad
    updateCompatibilityIndicators(deviceType);
    
    // Actualizar botón de aplicación
    updateApplyButton();
}

// Actualizar indicadores de compatibilidad
function updateCompatibilityIndicators(profileDeviceType) {
    $('.compatibility-indicator').each(function() {
        const deviceType = $(this).data('device-type');
        let html = '';
        
        if (!profileDeviceType) {
            // Perfil universal - compatible con todos
            html = '<i class="fas fa-check text-success" title="Compatible"></i>';
        } else if (deviceType === profileDeviceType) {
            // Tipos coinciden
            html = '<i class="fas fa-check text-success" title="Compatible"></i>';
        } else {
            // Tipos no coinciden
            html = '<i class="fas fa-exclamation-triangle text-warning" title="Tipo no coincide - requiere forzar"></i>';
        }
        
        $(this).html(html);
    });
}

// Configurar filtros de dispositivos
function setupDeviceFilters() {
    $('#statusFilter, #typeFilter').on('change', function() {
        filterDevices();
    });
}

// Filtrar dispositivos
function filterDevices() {
    const statusFilter = $('#statusFilter').val();
    const typeFilter = $('#typeFilter').val();
    
    $('.device-row').each(function() {
        const deviceStatus = $(this).data('status');
        const deviceType = $(this).data('device-type');
        
        let showRow = true;
        
        if (statusFilter && deviceStatus !== statusFilter) {
            showRow = false;
        }
        
        if (typeFilter && deviceType !== typeFilter) {
            showRow = false;
        }
        
        $(this).toggle(showRow);
    });
}

// Seleccionar todos los dispositivos visibles
function selectAll() {
    $('.device-row:visible .device-checkbox').prop('checked', true);
    updateSelectedCount();
}

// Deseleccionar todos
function selectNone() {
    $('.device-checkbox').prop('checked', false);
    updateSelectedCount();
}

// Seleccionar solo dispositivos compatibles
function selectCompatible() {
    const profileDeviceType = $('#profileSelect option:selected').data('device-type');
    
    $('.device-checkbox').prop('checked', false);
    
    if (!profileDeviceType) {
        // Perfil universal - todos son compatibles
        $('.device-row:visible .device-checkbox').prop('checked', true);
    } else {
        // Solo dispositivos del mismo tipo
        $(`.device-row:visible[data-device-type="${profileDeviceType}"] .device-checkbox`).prop('checked', true);
    }
    
    updateSelectedCount();
}

// Toggle todos los dispositivos visibles
function toggleAllDevices() {
    const checked = $('#selectAllDevices').is(':checked');
    $('.device-row:visible .device-checkbox').prop('checked', checked);
    updateSelectedCount();
}

// Actualizar contador de seleccionados
function updateSelectedCount() {
    const selectedCount = $('.device-checkbox:checked').length;
    $('#selectedCount').text(`${selectedCount} seleccionados`);
    updateApplyButton();
}

// Actualizar estado del botón de aplicación
function updateApplyButton() {
    const hasProfile = $('#profileSelect').val();
    const hasDevices = $('.device-checkbox:checked').length > 0;
    
    $('#applyButton').prop('disabled', !(hasProfile && hasDevices));
}

// Configurar validación del formulario
function setupFormValidation() {
    $('#bulkApplicationForm').on('submit', function(e) {
        e.preventDefault();
        
        const profileId = $('#profileSelect').val();
        const selectedDevices = $('.device-checkbox:checked').length;
        
        if (!profileId) {
            alert('Debe seleccionar un perfil');
            return false;
        }
        
        if (selectedDevices === 0) {
            alert('Debe seleccionar al menos un dispositivo');
            return false;
        }
        
        // Mostrar modal de confirmación
        showConfirmationModal();
    });
}

// Mostrar modal de confirmación
function showConfirmationModal() {
    const profileName = $('#profileSelect option:selected').text();
    const selectedDevices = $('.device-checkbox:checked').length;
    const selectedDeviceNames = [];
    
    $('.device-checkbox:checked').each(function() {
        const row = $(this).closest('tr');
        const deviceName = row.find('td:nth-child(2) strong').text();
        selectedDeviceNames.push(deviceName);
    });
    
    const deviceList = selectedDeviceNames.slice(0, 10).join(', ') + 
                      (selectedDeviceNames.length > 10 ? `... y ${selectedDeviceNames.length - 10} más` : '');
    
    const confirmHtml = `
        <div class="alert alert-warning">
            <h6><i class="fas fa-exclamation-triangle"></i> Confirmación requerida</h6>
            <p>Está a punto de aplicar el perfil <strong>"${profileName}"</strong> a <strong>${selectedDevices} dispositivos</strong>.</p>
        </div>
        
        <h6>Dispositivos afectados:</h6>
        <div class="border rounded p-2 mb-3" style="max-height: 150px; overflow-y: auto;">
            <small>${deviceList}</small>
        </div>
        
        <div class="alert alert-info">
            <small>
                <i class="fas fa-info-circle"></i>
                Esta acción sobrescribirá la configuración actual de todos los dispositivos seleccionados.
                Los cambios se registrarán en el historial para permitir rollback si es necesario.
            </small>
        </div>
    `;
    
    $('#confirmationContent').html(confirmHtml);
    $('#confirmationModal').modal('show');
}

// Ejecutar aplicación confirmada
function executeApplication() {
    $('#confirmationModal').modal('hide');
    $('#progressModal').modal('show');
    
    // Enviar formulario real
    setTimeout(() => {
        document.getElementById('bulkApplicationForm').submit();
    }, 500);
}

// Ver historial de configuración (para resultados exitosos)
function viewConfigHistory(historyId) {
    window.open(`get_config_details.php?history_id=${historyId}`, '_blank');
}

// Reintentar dispositivo con error
function retryDevice(deviceId) {
    const profileId = $('#profileSelect').val();
    
    if (!profileId) {
        alert('No hay perfil seleccionado para reintentar');
        return;
    }
    
    if (confirm(`¿Reintentar aplicar perfil al dispositivo ${deviceId}?`)) {
        // Crear formulario temporal para reintento individual
        const form = document.createElement('form');
        form.method = 'POST';
        form.innerHTML = `
            <input type="hidden" name="action" value="apply_bulk">
            <input type="hidden" name="profile_id" value="${profileId}">
            <input type="hidden" name="device_ids[]" value="${deviceId}">
            <input type="hidden" name="force_apply" value="1">
        `;
        document.body.appendChild(form);
        form.submit();
    }
}
</script>

<style>
.sticky-top {
    position: sticky;
    top: 0;
    z-index: 10;
    background: white;
}

.compatibility-indicator {
    font-size: 1.1em;
}

.device-row:hover {
    background-color: #f8f9fa;
}

.info-box {
    display: block;
    min-height: 90px;
    background: #fff;
    width: 100%;
    box-shadow: 0 1px 1px rgba(0,0,0,0.1);
    border-radius: 2px;
    margin-bottom: 15px;
}

.info-box .info-box-icon {
    border-top-left-radius: 2px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    border-bottom-left-radius: 2px;
    display: block;
    float: left;
    height: 90px;
    width: 90px;
    text-align: center;
    font-size: 45px;
    line-height: 90px;
    background: rgba(0,0,0,0.2);
}

.info-box .info-box-content {
    padding: 5px 10px;
    margin-left: 90px;
}

.info-box .info-box-number {
    display: block;
    font-weight: bold;
    font-size: 18px;
}

.info-box .info-box-text {
    display: block;
    font-size: 14px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.bg-success {
    background-color: #28a745 !important;
    color: white;
}

.bg-danger {
    background-color: #dc3545 !important;
    color: white;
}

.bg-warning {
    background-color: #ffc107 !important;
    color: white;
}

.bg-info {
    background-color: #17a2b8 !important;
    color: white;
}
</style>
