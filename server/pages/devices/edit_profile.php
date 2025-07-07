<?php
/**
 * Página de edición de perfil de configuración
 * server/pages/devices/edit_profile.php
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

// Obtener ID del perfil
$profile_id = $_GET['id'] ?? null;

if (!$profile_id) {
    $_SESSION['error_message'] = 'ID de perfil no especificado.';
    header('Location: profiles.php');
    exit();
}

// Obtener información del perfil
$profile = db_fetch_one(
    "SELECT dcp.*, u.username as created_by_name, u.name as created_by_full_name
     FROM device_config_profiles dcp 
     LEFT JOIN users u ON dcp.created_by = u.id 
     WHERE dcp.id = ?",
    [$profile_id]
);

if (!$profile) {
    $_SESSION['error_message'] = 'Perfil no encontrado.';
    header('Location: profiles.php');
    exit();
}

// Obtener configuración actual del perfil
$current_config = json_decode($profile['config_json'], true);
if (!$current_config) {
    $current_config = DeviceConfigManager::getDefaultConfig();
}

// Obtener reglas de validación para JavaScript
$validation_rules = DeviceConfigManager::getValidationRules();

// Obtener otros perfiles para comparación
$other_profiles = db_fetch_all(
    "SELECT id, name, device_type FROM device_config_profiles WHERE id != ? ORDER BY name",
    [$profile_id]
);

// Obtener dispositivos que usan este perfil
$devices_using = db_fetch_all(
    "SELECT d.device_id, d.name, d.device_type, d.status, d.last_access
     FROM devices d 
     WHERE JSON_EXTRACT(d.config_json, '$.profile_id') = ?
     ORDER BY d.device_id",
    [$profile_id]
);

// Procesar formulario si se envía
$success_message = '';
$error_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action'])) {
        switch ($_POST['action']) {
            case 'save_profile':
                $result = handleProfileSave();
                if ($result['success']) {
                    $success_message = $result['message'];
                    // Recargar configuración actualizada
                    $current_config = json_decode(
                        db_fetch_one("SELECT config_json FROM device_config_profiles WHERE id = ?", [$profile_id])['config_json'], 
                        true
                    );
                } else {
                    $error_message = $result['error'];
                }
                break;
                
            case 'apply_to_devices':
                $result = handleApplyToDevices();
                if ($result['success']) {
                    $success_message = $result['message'];
                } else {
                    $error_message = $result['error'];
                }
                break;
        }
    }
}

function handleProfileSave() {
    global $profile_id, $profile;
    
    // Obtener datos del formulario
    $name = trim($_POST['profile_name'] ?? '');
    $description = trim($_POST['profile_description'] ?? '');
    $device_type = $_POST['device_type'] ?? '';
    $is_default = isset($_POST['is_default']) ? 1 : 0;
    
    // Construir nueva configuración desde el formulario (misma lógica que configure.php)
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
    
    // Validar datos requeridos
    if (empty($name)) {
        return ['success' => false, 'error' => 'El nombre del perfil es requerido'];
    }
    
    // Verificar que no existe otro perfil con el mismo nombre (excepto el actual)
    $existing = db_fetch_one(
        "SELECT id FROM device_config_profiles WHERE name = ? AND device_type = ? AND id != ?",
        [$name, $device_type, $profile_id]
    );
    
    if ($existing) {
        return ['success' => false, 'error' => 'Ya existe otro perfil con ese nombre para este tipo de dispositivo'];
    }
    
    // Si se marca como default, quitar default de otros perfiles del mismo tipo
    if ($is_default && !$profile['is_default']) {
        db_update('device_config_profiles', 
            ['is_default' => 0], 
            'device_type = ? OR (device_type IS NULL AND ? IS NULL)',
            [$device_type, $device_type]
        );
    }
    
    // Actualizar perfil
    $update_data = [
        'name' => $name,
        'description' => $description,
        'device_type' => $device_type ?: null,
        'config_json' => json_encode($new_config),
        'is_default' => $is_default,
        'updated_at' => date('Y-m-d H:i:s')
    ];
    
    $result = db_update('device_config_profiles', $update_data, 'id = ?', [$profile_id]);
    
    if ($result !== false) {
        // Registrar en logs
        db_insert('system_logs', [
            'log_type' => 'info',
            'message' => "Perfil '{$name}' actualizado por " . ($_SESSION['username'] ?? 'usuario'),
            'details' => json_encode([
                'profile_id' => $profile_id,
                'old_name' => $profile['name'],
                'new_name' => $name,
                'user' => $_SESSION['username'] ?? 'N/A'
            ]),
            'timestamp' => date('Y-m-d H:i:s')
        ]);
        
        return ['success' => true, 'message' => "Perfil '$name' actualizado exitosamente"];
    } else {
        return ['success' => false, 'error' => 'Error al actualizar el perfil'];
    }
}

function handleApplyToDevices() {
    global $profile_id, $devices_using;
    
    if (empty($devices_using)) {
        return ['success' => false, 'error' => 'No hay dispositivos usando este perfil'];
    }
    
    $success_count = 0;
    $error_count = 0;
    $errors = [];
    
    foreach ($devices_using as $device) {
        try {
            // Aplicar perfil usando el manager
            $result = DeviceConfigManager::applyConfigProfile(
                $device['device_id'], 
                $profile_id, 
                $_SESSION['user_id']
            );
            
            if ($result['success']) {
                $success_count++;
            } else {
                $error_count++;
                $errors[] = "{$device['device_id']}: {$result['error']}";
            }
        } catch (Exception $e) {
            $error_count++;
            $errors[] = "{$device['device_id']}: {$e->getMessage()}";
        }
    }
    
    if ($success_count > 0 && $error_count == 0) {
        return ['success' => true, 'message' => "Configuración aplicada exitosamente a {$success_count} dispositivos"];
    } elseif ($success_count > 0) {
        return ['success' => true, 'message' => "Configuración aplicada a {$success_count} dispositivos. {$error_count} errores: " . implode(', ', $errors)];
    } else {
        return ['success' => false, 'error' => "No se pudo aplicar a ningún dispositivo. Errores: " . implode(', ', $errors)];
    }
}

// Definir el título de la página
$pageTitle = "Editar Perfil: " . $profile['name'];

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
                    <h1>Editar Perfil de Configuración</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item"><a href="profiles.php">Perfiles</a></li>
                        <li class="breadcrumb-item active">Editar</li>
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
                <!-- Información del perfil -->
                <div class="col-md-4">
                    <div class="card card-primary card-outline">
                        <div class="card-header">
                            <h3 class="card-title">Información del Perfil</h3>
                        </div>
                        <div class="card-body">
                            <dl class="row">
                                <dt class="col-sm-5">ID:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($profile['id']); ?></dd>
                                
                                <dt class="col-sm-5">Nombre:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($profile['name']); ?></dd>
                                
                                <dt class="col-sm-5">Tipo:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($profile['device_type'] ?: 'Universal'); ?></dd>
                                
                                <dt class="col-sm-5">Estado:</dt>
                                <dd class="col-sm-7">
                                    <?php if ($profile['is_default']): ?>
                                        <span class="badge badge-warning">
                                            <i class="fas fa-star"></i> Por Defecto
                                        </span>
                                    <?php else: ?>
                                        <span class="badge badge-secondary">Personalizado</span>
                                    <?php endif; ?>
                                </dd>
                                
                                <dt class="col-sm-5">Dispositivos:</dt>
                                <dd class="col-sm-7">
                                    <span class="badge badge-info"><?php echo count($devices_using); ?></span>
                                </dd>
                                
                                <dt class="col-sm-5">Creado por:</dt>
                                <dd class="col-sm-7"><?php echo htmlspecialchars($profile['created_by_full_name'] ?: $profile['created_by_name'] ?: 'Sistema'); ?></dd>
                                
                                <dt class="col-sm-5">Fecha:</dt>
                                <dd class="col-sm-7"><?php echo date('d/m/Y H:i', strtotime($profile['created_at'])); ?></dd>
                                
                                <?php if ($profile['updated_at']): ?>
                                <dt class="col-sm-5">Actualizado:</dt>
                                <dd class="col-sm-7"><?php echo date('d/m/Y H:i', strtotime($profile['updated_at'])); ?></dd>
                                <?php endif; ?>
                            </dl>
                        </div>
                    </div>

                    <!-- Dispositivos que usan este perfil -->
                    <div class="card card-info">
                        <div class="card-header">
                            <h3 class="card-title">Dispositivos Usando Este Perfil</h3>
                        </div>
                        <div class="card-body">
                            <?php if (empty($devices_using)): ?>
                                <p class="text-muted">Ningún dispositivo está usando este perfil actualmente.</p>
                            <?php else: ?>
                                <div class="list-group list-group-flush">
                                    <?php foreach ($devices_using as $device): ?>
                                        <div class="list-group-item p-2">
                                            <div class="d-flex justify-content-between align-items-center">
                                                <div>
                                                    <strong><?php echo htmlspecialchars($device['name'] ?: $device['device_id']); ?></strong>
                                                    <br><small class="text-muted"><?php echo htmlspecialchars($device['device_id']); ?></small>
                                                </div>
                                                <div>
                                                    <?php
                                                    $status_class = $device['status'] === 'online' ? 'success' : 'danger';
                                                    echo "<span class=\"badge badge-{$status_class}\">" . ucfirst($device['status']) . "</span>";
                                                    ?>
                                                </div>
                                            </div>
                                        </div>
                                    <?php endforeach; ?>
                                </div>
                                
                                <form method="POST" class="mt-3">
                                    <input type="hidden" name="action" value="apply_to_devices">
                                    <button type="submit" class="btn btn-info btn-block" 
                                            onclick="return confirm('¿Aplicar la configuración actualizada a todos estos dispositivos?')">
                                        <i class="fas fa-sync"></i> Aplicar Cambios a Dispositivos
                                    </button>
                                </form>
                            <?php endif; ?>
                        </div>
                    </div>

                    <!-- Acciones rápidas -->
                    <div class="card card-warning">
                        <div class="card-header">
                            <h3 class="card-title">Acciones Rápidas</h3>
                        </div>
                        <div class="card-body">
                            <button type="button" class="btn btn-success btn-block mb-2" onclick="loadFromProfile()">
                                <i class="fas fa-copy"></i> Copiar de Otro Perfil
                            </button>
                            
                            <button type="button" class="btn btn-info btn-block mb-2" onclick="exportProfile()">
                                <i class="fas fa-download"></i> Exportar Configuración
                            </button>
                            
                            <button type="button" class="btn btn-secondary btn-block" onclick="resetToDefault()">
                                <i class="fas fa-undo"></i> Restaurar por Defecto
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Formulario de configuración -->
                <div class="col-md-8">
                    <form method="POST" id="profileForm">
                        <input type="hidden" name="action" value="save_profile">
                        
                        <!-- Información básica del perfil -->
                        <div class="card">
                            <div class="card-header">
                                <h3 class="card-title">Información Básica</h3>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <label for="profile_name">Nombre del Perfil *</label>
                                            <input type="text" class="form-control" id="profile_name" name="profile_name" 
                                                   required value="<?php echo htmlspecialchars($profile['name']); ?>">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <label for="device_type">Tipo de Dispositivo</label>
                                            <select class="form-control" id="device_type" name="device_type">
                                                <option value="">Universal (todos los tipos)</option>
                                                <option value="Raspberry Pi" <?php echo $profile['device_type'] === 'Raspberry Pi' ? 'selected' : ''; ?>>Raspberry Pi</option>
                                                <option value="Edge Computer" <?php echo $profile['device_type'] === 'Edge Computer' ? 'selected' : ''; ?>>Edge Computer</option>
                                                <option value="Industrial PC" <?php echo $profile['device_type'] === 'Industrial PC' ? 'selected' : ''; ?>>Industrial PC</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="form-group">
                                    <label for="profile_description">Descripción</label>
                                    <textarea class="form-control" id="profile_description" name="profile_description" rows="2"><?php echo htmlspecialchars($profile['description'] ?? ''); ?></textarea>
                                </div>
                                
                                <div class="custom-control custom-checkbox">
                                    <input type="checkbox" class="custom-control-input" id="is_default" name="is_default"
                                           <?php echo $profile['is_default'] ? 'checked' : ''; ?>>
                                    <label class="custom-control-label" for="is_default">
                                        Establecer como perfil por defecto
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Configuración técnica (reutilizar de configure.php) -->
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
                                    
                                    <!-- TAB: Cámara (misma estructura que configure.php) -->
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
                                                <!-- Resto de configuración de cámara... -->
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

                                    <!-- TAB: Fatiga (estructura simplificada - los demás tabs serían similares) -->
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
                                                </div>
                                                
                                                <!-- Más campos de fatiga... -->
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Los demás tabs seguirían la misma estructura... -->
                                    <div class="tab-pane" id="yawn-tab">
                                        <h4>Detección de Bostezos</h4>
                                        <p class="text-muted">Configuración para detección de bostezos...</p>
                                        <!-- Campos de configuración de bostezos -->
                                    </div>
                                    
                                    <!-- Tabs adicionales... -->
                                    
                                </div>
                            </div>
                            
                            <div class="card-footer">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="form-group">
                                            <label>Resumen de Cambios (opcional)</label>
                                            <input type="text" class="form-control" name="change_summary" 
                                                   placeholder="Ej: Ajustes para turno nocturno, optimización de rendimiento">
                                        </div>
                                    </div>
                                    <div class="col-md-6 text-right">
                                        <a href="profiles.php" class="btn btn-secondary mr-2">
                                            <i class="fas fa-times"></i> Cancelar
                                        </a>
                                        <button type="submit" class="btn btn-primary">
                                            <i class="fas fa-save"></i> Guardar Perfil
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
    $('#profileForm').on('submit', function(e) {
        const profileName = $('#profile_name').val().trim();
        
        if (!profileName) {
            e.preventDefault();
            alert('El nombre del perfil es requerido');
            return false;
        }
    });
}

// Cargar configuración de otro perfil
function loadFromProfile() {
    const profiles = <?php echo json_encode($other_profiles); ?>;
    
    if (profiles.length === 0) {
        alert('No hay otros perfiles disponibles para copiar');
        return;
    }
    
    let profileOptions = profiles.map(p => 
        `<option value="${p.id}">${p.name} ${p.device_type ? '(' + p.device_type + ')' : ''}</option>`
    ).join('');
    
    const html = `
        <div class="form-group">
            <label>Seleccionar perfil para copiar configuración:</label>
            <select class="form-control" id="sourceProfile">
                <option value="">Seleccionar...</option>
                ${profileOptions}
            </select>
        </div>
    `;
    
    if (confirm('¿Cargar configuración de otro perfil? Se sobrescribirán los valores actuales.')) {
        // Aquí iría la lógica para cargar via AJAX
        alert('Funcionalidad en desarrollo');
    }
}

// Exportar configuración del perfil
function exportProfile() {
    window.open(`export_profile.php?id=<?php echo $profile_id; ?>`, '_blank');
}

// Restaurar configuración por defecto
function resetToDefault() {
    if (confirm('¿Restaurar configuración por defecto? Se perderán todos los ajustes personalizados.')) {
        // Aquí iría la lógica para restaurar valores por defecto
        alert('Funcionalidad en desarrollo');
    }
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

.badge {
    font-size: 0.9em;
}

.list-group-item {
    border: none;
    border-bottom: 1px solid #dee2e6;
}

.description-block {
    text-align: center;
}
</style>