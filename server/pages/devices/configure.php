<?php
/**
 * Página de configuración individual de dispositivo
 * server/pages/devices/configure.php
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

// Obtener ID del dispositivo
$device_id = $_GET['id'] ?? null;

if (!$device_id) {
    $_SESSION['error_message'] = 'ID de dispositivo no especificado.';
    header('Location: config.php');
    exit();
}

// Obtener información del dispositivo
$device = db_fetch_one(
    "SELECT d.*, m.name as machine_name FROM devices d 
     LEFT JOIN machines m ON d.machine_id = m.id 
     WHERE d.device_id = ?",
    [$device_id]
);

if (!$device) {
    $_SESSION['error_message'] = 'Dispositivo no encontrado.';
    header('Location: config.php');
    exit();
}

// Obtener configuración actual
$config_data = DeviceConfigManager::getDeviceConfig($device_id);
$current_config = $config_data['config'];

// Obtener reglas de validación para JavaScript
$validation_rules = DeviceConfigManager::getValidationRules();

// Obtener perfiles disponibles
$profiles = DeviceConfigManager::getConfigProfiles($device['device_type']);

// Procesar formulario si se envía
$success_message = '';
$error_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action'])) {
        switch ($_POST['action']) {
            case 'save_config':
                $result = handleConfigSave();
                if ($result['success']) {
                    $success_message = $result['message'];
                    // Recargar configuración actualizada
                    $config_data = DeviceConfigManager::getDeviceConfig($device_id);
                    $current_config = $config_data['config'];
                } else {
                    $error_message = $result['error'];
                }
                break;
                
            case 'apply_profile':
                $result = handleProfileApplication();
                if ($result['success']) {
                    $success_message = $result['message'];
                    // Recargar configuración actualizada
                    $config_data = DeviceConfigManager::getDeviceConfig($device_id);
                    $current_config = $config_data['config'];
                } else {
                    $error_message = $result['error'];
                }
                break;
        }
    }
}

function handleConfigSave() {
    global $device_id;
    
    // Construir nueva configuración desde el formulario
    $new_config = [
        'camera' => [
            'fps' => (int)$_POST['camera_fps'],
            'width' => (int)$_POST['camera_width'],
            'height' => (int)$_POST['camera_height'],
            'brightness' => (int)$_POST['camera_brightness'],
            'contrast' => (int)$_POST['camera_contrast'],
            'saturation' => (int)$_POST['camera_saturation'],
            'buffer_size' => (int)$_POST['camera_buffer_size'],
            'use_threading' => isset($_POST['camera_use_threading']),
            'warmup_time' => (int)$_POST['camera_warmup_time']
        ],
        'fatigue' => [
            'eye_closed_threshold' => (float)$_POST['fatigue_eye_closed_threshold'],
            'ear_threshold' => (float)$_POST['fatigue_ear_threshold'],
            'ear_night_adjustment' => (float)$_POST['fatigue_ear_night_adjustment'],
            'frames_to_confirm' => (int)$_POST['fatigue_frames_to_confirm'],
            'calibration_period' => (int)$_POST['fatigue_calibration_period'],
            'alarm_cooldown' => (int)$_POST['fatigue_alarm_cooldown'],
            'multiple_fatigue_threshold' => (int)$_POST['fatigue_multiple_fatigue_threshold'],
            'night_mode_threshold' => (int)$_POST['fatigue_night_mode_threshold'],
            'enable_night_mode' => isset($_POST['fatigue_enable_night_mode'])
        ],
        'yawn' => [
            'mouth_threshold' => (float)$_POST['yawn_mouth_threshold'],
            'duration_threshold' => (float)$_POST['yawn_duration_threshold'],
            'frames_to_confirm' => (int)$_POST['yawn_frames_to_confirm'],
            'alert_cooldown' => (float)$_POST['yawn_alert_cooldown'],
            'max_yawns_before_alert' => (int)$_POST['yawn_max_yawns_before_alert'],
            'report_delay' => (float)$_POST['yawn_report_delay'],
            'enable_auto_calibration' => isset($_POST['yawn_enable_auto_calibration']),
            'calibration_frames' => (int)$_POST['yawn_calibration_frames'],
            'calibration_factor' => (float)$_POST['yawn_calibration_factor'],
            'enable_sounds' => isset($_POST['yawn_enable_sounds'])
        ],
        'distraction' => [
            'rotation_threshold_day' => (float)$_POST['distraction_rotation_threshold_day'],
            'rotation_threshold_night' => (float)$_POST['distraction_rotation_threshold_night'],
            'level1_time' => (int)$_POST['distraction_level1_time'],
            'level2_time' => (int)$_POST['distraction_level2_time'],
            'confidence_threshold' => (float)$_POST['distraction_confidence_threshold'],
            'audio_enabled' => isset($_POST['distraction_audio_enabled']),
            'level1_volume' => (float)$_POST['distraction_level1_volume'],
            'level2_volume' => (float)$_POST['distraction_level2_volume']
        ],
        'behavior' => [
            'confidence_threshold' => (float)$_POST['behavior_confidence_threshold'],
            'night_confidence_threshold' => (float)$_POST['behavior_night_confidence_threshold'],
            'phone_alert_threshold_1' => (int)$_POST['behavior_phone_alert_threshold_1'],
            'phone_alert_threshold_2' => (int)$_POST['behavior_phone_alert_threshold_2'],
            'cigarette_continuous_threshold' => (int)$_POST['behavior_cigarette_continuous_threshold'],
            'audio_enabled' => isset($_POST['behavior_audio_enabled'])
        ],
        'audio' => [
            'enabled' => isset($_POST['audio_enabled']),
            'volume' => (float)$_POST['audio_volume'],
            'frequency' => (int)$_POST['audio_frequency'],
            'channels' => (int)$_POST['audio_channels'],
            'buffer' => (int)$_POST['audio_buffer']
        ],
        'system' => [
            'enable_gui' => isset($_POST['system_enable_gui']),
            'log_level' => $_POST['system_log_level'],
            'debug_mode' => isset($_POST['system_debug_mode']),
            'performance_monitoring' => isset($_POST['system_performance_monitoring']),
            'auto_optimization' => isset($_POST['system_auto_optimization'])
        ],
        'sync' => [
            'enabled' => isset($_POST['sync_enabled']),
            'auto_sync_interval' => (int)$_POST['sync_auto_sync_interval'],
            'batch_size' => (int)$_POST['sync_batch_size'],
            'connection_timeout' => (int)$_POST['sync_connection_timeout'],
            'max_retries' => (int)$_POST['sync_max_retries']
        ]
    ];
    
    // Crear resumen de cambios
    $change_summary = $_POST['change_summary'] ?? 'Configuración manual actualizada';
    
    return DeviceConfigManager::updateDeviceConfig($device_id, $new_config, $_SESSION['user_id'], $change_summary);
}

function handleProfileApplication() {
    global $device_id;
    
    $profile_id = (int)$_POST['profile_id'];
    
    if (!$profile_id) {
        return ['success' => false, 'error' => 'ID de perfil no válido'];
    }
    
    return DeviceConfigManager::applyConfigProfile($device_id, $profile_id, $_SESSION['user_id']);
}

// Definir el título de la página
$pageTitle = "Configurar Dispositivo: " . ($device['name'] ?: $device_id);

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
                    <h1>Configurar Dispositivo</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item"><a href="config.php">Configuración</a></li>
                        <li class="breadcrumb-item active">Configurar</li>
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

            <div class="row">
                <!-- Información del dispositivo -->
                <div class="col-md-4">
                    <div class="card card-primary card-outline">
                        <div class="card-header">
                            <h3 class="card-title">Información del Dispositivo</h3>
                        </div>
                        <div class="card-body">
                            <dl class="row">
                                <dt class="col-sm-5">ID:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($device['device_id']); ?></dd>
                                
                                <dt class="col-sm-5">Nombre:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($device['name'] ?: 'Sin nombre'); ?></dd>
                                
                                <dt class="col-sm-5">Tipo:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($device['device_type']); ?></dd>
                                
                                <dt class="col-sm-5">Estado:</dt>
                                <dd class="col-sm-7">
                                    <?php
                                    $status_class = $device['status'] === 'online' ? 'success' : 'danger';
                                    echo "<span class=\"badge badge-{$status_class}\">" . ucfirst($device['status']) . "</span>";
                                    ?>
                                </dd>
                                
                                <dt class="col-sm-5">Máquina:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($device['machine_name'] ?: 'No asignada'); ?></dd>
                                
                                <dt class="col-sm-5">Config. Versión:</dt>
                                <dd class="col-sm-7">
                                    <span class="badge badge-info">v<?php echo $config_data['version']; ?></span>
                                    <?php if ($config_data['pending']): ?>
                                        <br><small class="text-warning">
                                            <i class="fas fa-clock"></i> Cambios pendientes
                                        </small>
                                    <?php endif; ?>
                                </dd>
                                
                                <dt class="col-sm-5">Última Aplicación:</dt>
                                <dd class="col-sm-7">
                                    <?php if ($config_data['last_applied']): ?>
                                        <?php echo date('d/m/Y H:i', strtotime($config_data['last_applied'])); ?>
                                    <?php else: ?>
                                        <span class="text-muted">Nunca</span>
                                    <?php endif; ?>
                                </dd>
                            </dl>
                        </div>
                    </div>

                    <!-- Perfiles de configuración -->
                    <div class="card card-secondary">
                        <div class="card-header">
                            <h3 class="card-title">Perfiles Predefinidos</h3>
                        </div>
                        <div class="card-body">
                            <form method="POST" id="profileForm">
                                <input type="hidden" name="action" value="apply_profile">
                                
                                <div class="form-group">
                                    <label>Seleccionar Perfil:</label>
                                    <select class="form-control" name="profile_id" id="profileSelect">
                                        <option value="">Seleccione un perfil...</option>
                                        <?php foreach ($profiles as $profile): ?>
                                            <option value="<?php echo $profile['id']; ?>" 
                                                    data-description="<?php echo htmlspecialchars($profile['description']); ?>">
                                                <?php echo htmlspecialchars($profile['name']); ?>
                                                <?php if ($profile['is_default']): ?>
                                                    (Por defecto)
                                                <?php endif; ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                                
                                <div id="profileDescription" class="alert alert-info" style="display: none;"></div>
                                
                                <button type="button" class="btn btn-secondary btn-block" onclick="applyProfile()">
                                    <i class="fas fa-magic"></i> Aplicar Perfil
                                </button>
                            </form>
                        </div>
                    </div>

                    <!-- Acciones rápidas -->
                    <div class="card card-warning">
                        <div class="card-header">
                            <h3 class="card-title">Acciones Rápidas</h3>
                        </div>
                        <div class="card-body">
                            <button type="button" class="btn btn-info btn-block mb-2" onclick="loadDefaultConfig()">
                                <i class="fas fa-undo"></i> Restaurar por Defecto
                            </button>
                            
                            <button type="button" class="btn btn-success btn-block mb-2" onclick="exportConfig()">
                                <i class="fas fa-download"></i> Exportar Configuración
                            </button>
                            
                            <button type="button" class="btn btn-warning btn-block" onclick="testConnection()">
                                <i class="fas fa-wifi"></i> Probar Conexión
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Formulario de configuración -->
                <div class="col-md-8">
                    <form method="POST" id="configForm">
                        <input type="hidden" name="action" value="save_config">
                        
                        <div class="card">
                            <div class="card-header p-2">
                                <ul class="nav nav-pills" id="configTabs">
                                    <li class="nav-item">
                                        <a class="nav-link active" href="#camera-tab" data-toggle="tab">
                                            <i class="fas fa-camera"></i> Cámara
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#fatigue-tab" data-toggle="tab">
                                            <i class="fas fa-bed"></i> Fatiga
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#yawn-tab" data-toggle="tab">
                                            <i class="fas fa-tired"></i> Bostezos
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#distraction-tab" data-toggle="tab">
                                            <i class="fas fa-eye-slash"></i> Distracción
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#behavior-tab" data-toggle="tab">
                                            <i class="fas fa-exclamation-circle"></i> Comportamientos
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#audio-tab" data-toggle="tab">
                                            <i class="fas fa-volume-up"></i> Audio
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#system-tab" data-toggle="tab">
                                            <i class="fas fa-cog"></i> Sistema
                                        </a>
                                    </li>
                                    <li class="nav-item">
                                        <a class="nav-link" href="#sync-tab" data-toggle="tab">
                                            <i class="fas fa-sync"></i> Sincronización
                                        </a>
                                    </li>
                                </ul>
                            </div>
                            
                            <div class="card-body">
                                <div class="tab-content">
                                    
                                    <!-- TAB: Cámara -->
                                    <div class="tab-pane active" id="camera-tab">
                                        <h4>Configuración de Cámara</h4>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <label>FPS (Frames por Segundo)</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="camera_fps" 
                                                           min="1" max="30" 
                                                           value="<?php echo $current_config['camera']['fps']; ?>"
                                                           oninput="updateSliderValue(this, 'camera_fps_value')">
                                                    <span id="camera_fps_value" class="badge badge-info"><?php echo $current_config['camera']['fps']; ?></span>
                                                    <small class="form-text text-muted">Menor FPS = menor uso de CPU</small>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Resolución</label>
                                                    <select class="form-control" name="camera_width" onchange="updateResolution(this)">
                                                        <option value="320" data-height="240" <?php echo $current_config['camera']['width'] == 320 ? 'selected' : ''; ?>>320x240 (QVGA)</option>
                                                        <option value="640" data-height="480" <?php echo $current_config['camera']['width'] == 640 ? 'selected' : ''; ?>>640x480 (VGA)</option>
                                                        <option value="800" data-height="600" <?php echo $current_config['camera']['width'] == 800 ? 'selected' : ''; ?>>800x600 (SVGA)</option>
                                                        <option value="1024" data-height="768" <?php echo $current_config['camera']['width'] == 1024 ? 'selected' : ''; ?>>1024x768 (XGA)</option>
                                                    </select>
                                                    <input type="hidden" name="camera_height" value="<?php echo $current_config['camera']['height']; ?>">
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Brillo</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="camera_brightness" 
                                                           min="-100" max="100" 
                                                           value="<?php echo $current_config['camera']['brightness']; ?>"
                                                           oninput="updateSliderValue(this, 'camera_brightness_value')">
                                                    <span id="camera_brightness_value" class="badge badge-info"><?php echo $current_config['camera']['brightness']; ?></span>
                                                </div>
                                            </div>
                                            
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <label>Contraste</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="camera_contrast" 
                                                           min="-100" max="100" 
                                                           value="<?php echo $current_config['camera']['contrast']; ?>"
                                                           oninput="updateSliderValue(this, 'camera_contrast_value')">
                                                    <span id="camera_contrast_value" class="badge badge-info"><?php echo $current_config['camera']['contrast']; ?></span>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Saturación</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="camera_saturation" 
                                                           min="-100" max="100" 
                                                           value="<?php echo $current_config['camera']['saturation']; ?>"
                                                           oninput="updateSliderValue(this, 'camera_saturation_value')">
                                                    <span id="camera_saturation_value" class="badge badge-info"><?php echo $current_config['camera']['saturation']; ?></span>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Tamaño de Buffer</label>
                                                    <select class="form-control" name="camera_buffer_size">
                                                        <option value="1" <?php echo $current_config['camera']['buffer_size'] == 1 ? 'selected' : ''; ?>>1 (Mínimo)</option>
                                                        <option value="2" <?php echo $current_config['camera']['buffer_size'] == 2 ? 'selected' : ''; ?>>2</option>
                                                        <option value="3" <?php echo $current_config['camera']['buffer_size'] == 3 ? 'selected' : ''; ?>>3</option>
                                                        <option value="5" <?php echo $current_config['camera']['buffer_size'] == 5 ? 'selected' : ''; ?>>5 (Máximo)</option>
                                                    </select>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="custom-control custom-switch">
                                                    <input type="checkbox" class="custom-control-input" 
                                                           id="camera_use_threading" name="camera_use_threading"
                                                           <?php echo $current_config['camera']['use_threading'] ? 'checked' : ''; ?>>
                                                    <label class="custom-control-label" for="camera_use_threading">
                                                        Usar Threading (recomendado para Pi)
                                                    </label>
                                                </div>
                                            </div>
                                            
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <label>Tiempo de Calentamiento (segundos)</label>
                                                    <input type="number" class="form-control" 
                                                           name="camera_warmup_time" 
                                                           min="0" max="10"
                                                           value="<?php echo $current_config['camera']['warmup_time']; ?>">
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <!-- TAB: Fatiga -->
                                    <div class="tab-pane" id="fatigue-tab">
                                        <h4>Detección de Fatiga</h4>
                                        <div class="row">
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <label>Umbral Ojos Cerrados (segundos)</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="fatigue_eye_closed_threshold" 
                                                           min="0.5" max="5" step="0.1"
                                                           value="<?php echo $current_config['fatigue']['eye_closed_threshold']; ?>"
                                                           oninput="updateSliderValue(this, 'fatigue_eye_closed_threshold_value')">
                                                    <span id="fatigue_eye_closed_threshold_value" class="badge badge-info"><?php echo $current_config['fatigue']['eye_closed_threshold']; ?>s</span>
                                                    <small class="form-text text-muted">Tiempo con ojos cerrados para detectar microsueño</small>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Umbral EAR (Eye Aspect Ratio)</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="fatigue_ear_threshold" 
                                                           min="0.1" max="0.5" step="0.01"
                                                           value="<?php echo $current_config['fatigue']['ear_threshold']; ?>"
                                                           oninput="updateSliderValue(this, 'fatigue_ear_threshold_value')">
                                                    <span id="fatigue_ear_threshold_value" class="badge badge-info"><?php echo $current_config['fatigue']['ear_threshold']; ?></span>
                                                    <small class="form-text text-muted">Menor valor = más sensible</small>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Frames para Confirmar</label>
                                                    <input type="number" class="form-control" 
                                                           name="fatigue_frames_to_confirm" 
                                                           min="1" max="10"
                                                           value="<?php echo $current_config['fatigue']['frames_to_confirm']; ?>">
                                                    <small class="form-text text-muted">Frames consecutivos para confirmar fatiga</small>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Período de Calibración (frames)</label>
                                                    <input type="number" class="form-control" 
                                                           name="fatigue_calibration_period" 
                                                           min="10" max="120"
                                                           value="<?php echo $current_config['fatigue']['calibration_period']; ?>">
                                                </div>
                                            </div>
                                            
                                            <div class="col-md-6">
                                                <div class="form-group">
                                                    <label>Tiempo de Espera entre Alarmas (segundos)</label>
                                                    <input type="number" class="form-control" 
                                                           name="fatigue_alarm_cooldown" 
                                                           min="1" max="30"
                                                           value="<?php echo $current_config['fatigue']['alarm_cooldown']; ?>">
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Episodios para Alerta Crítica</label>
                                                    <input type="number" class="form-control" 
                                                           name="fatigue_multiple_fatigue_threshold" 
                                                           min="1" max="10"
                                                           value="<?php echo $current_config['fatigue']['multiple_fatigue_threshold']; ?>">
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Umbral Modo Nocturno</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="fatigue_night_mode_threshold" 
                                                           min="10" max="100"
                                                           value="<?php echo $current_config['fatigue']['night_mode_threshold']; ?>"
                                                           oninput="updateSliderValue(this, 'fatigue_night_mode_threshold_value')">
                                                    <span id="fatigue_night_mode_threshold_value" class="badge badge-info"><?php echo $current_config['fatigue']['night_mode_threshold']; ?></span>
                                                </div>
                                                
                                                <div class="form-group">
                                                    <label>Ajuste EAR Nocturno</label>
                                                    <input type="range" class="form-control-range" 
                                                           name="fatigue_ear_night_adjustment" 
                                                           min="0.01" max="0.1" step="0.01"
                                                           value="<?php echo $current_config['fatigue']['ear_night_adjustment']; ?>"
                                                           oninput="updateSliderValue(this, 'fatigue_ear_night_adjustment_value')">
                                                    <span id="fatigue_ear_night_adjustment_value" class="badge badge-info"><?php echo $current_config['fatigue']['ear_night_adjustment']; ?></span>
                                                </div>
                                            </div>
                                        </div>
                                        
                                        <div class="custom-control custom-switch">
                                            <input type="checkbox" class="custom-control-input" 
                                                   id="fatigue_enable_night_mode" name="fatigue_enable_night_mode"
                                                   <?php echo $current_config['fatigue']['enable_night_mode'] ? 'checked' : ''; ?>>
                                            <label class="custom-control-label" for="fatigue_enable_night_mode">
                                                Habilitar Modo Nocturno Automático
                                            </label>
                                        </div>
                                    </div>

                                    <!-- Continúa con los demás tabs... -->
                                    
                                </div>
                            </div>
                            
                            <div class="card-footer">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <label>Resumen de Cambios (opcional)</label>
                                            <input type="text" class="form-control" name="change_summary" 
                                                   placeholder="Ej: Ajuste de sensibilidad para turno nocturno">
                                        </div>
                                    </div>
                                    <div class="col-md-6 text-right">
                                        <a href="config.php" class="btn btn-secondary mr-2">
                                            <i class="fas fa-times"></i> Cancelar
                                        </a>
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-save"></i> Guardar Configuración
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </section>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
// Reglas de validación desde PHP
const validationRules = <?php echo json_encode($validation_rules); ?>;

$(document).ready(function() {
    // Configurar validación del formulario
    setupFormValidation();
    
    // Configurar listeners para perfiles
    setupProfileHandlers();
    
    // Auto-save en localStorage cada minuto
    setInterval(autoSaveConfig, 60000);
});

// Actualizar valor de slider en tiempo real
function updateSliderValue(slider, spanId) {
    const value = parseFloat(slider.value);
    const span = document.getElementById(spanId);
    
    if (spanId.includes('threshold') && spanId.includes('value')) {
        span.textContent = value + (spanId.includes('time') ? 's' : '');
    } else {
        span.textContent = value;
    }
}

// Actualizar resolución cuando cambia el ancho
function updateResolution(select) {
    const height = select.options[select.selectedIndex].getAttribute('data-height');
    document.querySelector('input[name="camera_height"]').value = height;
}

// Configurar validación del formulario
function setupFormValidation() {
    $('#configForm').on('submit', function(e) {
        e.preventDefault();
        
        // Validar todos los campos
        const errors = validateAllFields();
        
        if (errors.length > 0) {
            showValidationErrors(errors);
            return false;
        }
        
        // Si no hay errores, enviar formulario
        this.submit();
    });
    
    // Validación en tiempo real
    $('input[type="range"], input[type="number"]').on('input change', function() {
        validateField(this);
    });
}

// Validar un campo específico
function validateField(field) {
    const name = field.name;
    const value = parseFloat(field.value);
    
    // Extraer sección y parámetro del nombre
    const parts = name.split('_');
    const section = parts[0];
    const param = parts.slice(1).join('_');
    
    if (!validationRules[section] || !validationRules[section][param]) {
        return true; // No hay reglas para este campo
    }
    
    const rule = validationRules[section][param];
    
    // Validar tipo
    if (rule.type === 'int' && !Number.isInteger(value)) {
        showFieldError(field, 'Debe ser un número entero');
        return false;
    }
    
    if (rule.type === 'float' && isNaN(value)) {
        showFieldError(field, 'Debe ser un número válido');
        return false;
    }
    
    // Validar rango
    if (rule.min !== undefined && value < rule.min) {
        showFieldError(field, `Valor mínimo: ${rule.min}`);
        return false;
    }
    
    if (rule.max !== undefined && value > rule.max) {
        showFieldError(field, `Valor máximo: ${rule.max}`);
        return false;
    }
    
    // Validar valores específicos
    if (rule.values && !rule.values.includes(value)) {
        showFieldError(field, `Valores válidos: ${rule.values.join(', ')}`);
        return false;
    }
    
    // Si llegamos aquí, el campo es válido
    clearFieldError(field);
    return true;
}

// Mostrar error en campo específico
function showFieldError(field, message) {
    const $field = $(field);
    $field.addClass('is-invalid');
    
    // Remover mensaje anterior si existe
    $field.siblings('.invalid-feedback').remove();
    
    // Agregar nuevo mensaje
    $field.after(`<div class="invalid-feedback">${message}</div>`);
}

// Limpiar error de campo
function clearFieldError(field) {
    const $field = $(field);
    $field.removeClass('is-invalid');
    $field.siblings('.invalid-feedback').remove();
}

// Validar todos los campos
function validateAllFields() {
    const errors = [];
    const fields = document.querySelectorAll('input[type="range"], input[type="number"]');
    
    fields.forEach(field => {
        if (!validateField(field)) {
            errors.push(`${field.name}: Campo inválido`);
        }
    });
    
    return errors;
}

// Mostrar errores de validación
function showValidationErrors(errors) {
    const errorHtml = `
        <div class="alert alert-danger alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert">
                <span aria-hidden="true">&times;</span>
            </button>
            <h5><i class="fas fa-exclamation-triangle"></i> Errores de Validación:</h5>
            <ul>
                ${errors.map(error => `<li>${error}</li>`).join('')}
            </ul>
        </div>
    `;
    
    $('.content-wrapper .container-fluid').prepend(errorHtml);
    
    // Scroll hacia arriba para mostrar errores
    $('html, body').animate({ scrollTop: 0 }, 500);
}

// Configurar handlers para perfiles
function setupProfileHandlers() {
    $('#profileSelect').on('change', function() {
        const description = $(this).find(':selected').data('description');
        if (description) {
            $('#profileDescription').html(`<strong>Descripción:</strong> ${description}`).show();
        } else {
            $('#profileDescription').hide();
        }
    });
}

// Aplicar perfil seleccionado
function applyProfile() {
    const profileId = $('#profileSelect').val();
    
    if (!profileId) {
        showNotification('warning', 'Seleccione un perfil');
        return;
    }
    
    if (confirm('¿Aplicar este perfil? Se sobrescribirá la configuración actual.')) {
        $('#profileForm').submit();
    }
}

// Cargar configuración por defecto
function loadDefaultConfig() {
    if (confirm('¿Restaurar configuración por defecto? Se perderán todos los cambios actuales.')) {
        $.ajax({
            url: 'load_default_config.php',
            method: 'POST',
            data: { device_id: '<?php echo $device_id; ?>' },
            success: function(response) {
                if (response.success) {
                    showNotification('success', 'Configuración por defecto cargada');
                    setTimeout(() => location.reload(), 1500);
                } else {
                    showNotification('error', response.message || 'Error al cargar configuración');
                }
            },
            error: function() {
                showNotification('error', 'Error de conexión');
            }
        });
    }
}

// Exportar configuración
function exportConfig() {
    window.open(`export_config.php?device_id=<?php echo $device_id; ?>`, '_blank');
}

// Probar conexión con dispositivo
function testConnection() {
    const btn = $('button:contains("Probar Conexión")');
    const originalHtml = btn.html();
    
    btn.html('<i class="fas fa-spinner fa-spin"></i> Probando...').prop('disabled', true);
    
    $.ajax({
        url: 'test_connection.php',
        method: 'POST',
        data: { device_id: '<?php echo $device_id; ?>' },
        success: function(response) {
            if (response.success) {
                showNotification('success', `Conexión exitosa. Última respuesta: ${response.last_seen}`);
            } else {
                showNotification('warning', response.message || 'Dispositivo no responde');
            }
        },
        error: function() {
            showNotification('error', 'Error al probar conexión');
        },
        complete: function() {
            btn.html(originalHtml).prop('disabled', false);
        }
    });
}

// Auto-guardar configuración en localStorage
function autoSaveConfig() {
    const formData = new FormData(document.getElementById('configForm'));
    const config = {};
    
    for (let [key, value] of formData.entries()) {
        config[key] = value;
    }
    
    localStorage.setItem(`device_config_${<?php echo json_encode($device_id); ?>}`, JSON.stringify(config));
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
    
    setTimeout(function() {
        notification.alert('close');
    }, 5000);
}
</script>

<style>
.form-control-range {
    width: 100%;
}

.tab-content {
    min-height: 400px;
}

.nav-pills .nav-link {
    color: #6c757d;
}

.nav-pills .nav-link.active {
    background-color: #007bff;
}

.is-invalid {
    border-color: #dc3545;
}

.invalid-feedback {
    display: block;
    width: 100%;
    margin-top: 0.25rem;
    font-size: 0.875em;
    color: #dc3545;
}

.badge {
    font-size: 0.9em;
}

/* Mejorar apariencia de sliders */
.form-control-range::-webkit-slider-thumb {
    background: #007bff;
}

.form-control-range::-moz-range-thumb {
    background: #007bff;
    border: none;
}
</style>

<script src="<?php echo BASE_URL; ?>/assets/js/device-config.js"></script>
<script>
// Pasar reglas de validación y configuración por defecto desde PHP
window.validationRules = <?php echo json_encode($validation_rules); ?>;
window.defaultConfig = <?php echo json_encode(DeviceConfigManager::getDefaultConfig()); ?>;
</script>