<?php
/**
 * Página de gestión de perfiles de configuración
 * server/pages/devices/profiles.php
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
$pageTitle = "Perfiles de Configuración";

// Incluir header y sidebar
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Obtener todos los perfiles
$profiles = db_fetch_all("
    SELECT dcp.*, 
           u.username as created_by_name,
           u.name as created_by_full_name,
           COUNT(d.id) as devices_using
    FROM device_config_profiles dcp
    LEFT JOIN users u ON dcp.created_by = u.id
    LEFT JOIN devices d ON JSON_EXTRACT(d.config_json, '$.profile_id') = dcp.id
    GROUP BY dcp.id
    ORDER BY dcp.is_default DESC, dcp.device_type ASC, dcp.name ASC
");

// Obtener tipos de dispositivos únicos
$device_types = db_fetch_all("SELECT DISTINCT device_type FROM devices WHERE device_type IS NOT NULL ORDER BY device_type");

// Procesar acciones
$success_message = '';
$error_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    if (isset($_POST['action'])) {
        switch ($_POST['action']) {
            case 'create_profile':
                $result = handleCreateProfile();
                if ($result['success']) {
                    $success_message = $result['message'];
                    // Recargar perfiles
                    header('Location: profiles.php?success=' . urlencode($result['message']));
                    exit;
                } else {
                    $error_message = $result['error'];
                }
                break;
                
            case 'delete_profile':
                $result = handleDeleteProfile();
                if ($result['success']) {
                    $success_message = $result['message'];
                } else {
                    $error_message = $result['error'];
                }
                break;
                
            case 'set_default':
                $result = handleSetDefault();
                if ($result['success']) {
                    $success_message = $result['message'];
                } else {
                    $error_message = $result['error'];
                }
                break;
        }
    }
}

function handleCreateProfile() {
    global $device_types;
    
    $name = trim($_POST['profile_name'] ?? '');
    $description = trim($_POST['profile_description'] ?? '');
    $device_type = $_POST['device_type'] ?? '';
    $is_default = isset($_POST['is_default']) ? 1 : 0;
    $source_profile_id = $_POST['source_profile_id'] ?? null;
    
    if (empty($name)) {
        return ['success' => false, 'error' => 'El nombre del perfil es requerido'];
    }
    
    // Verificar que no existe otro perfil con el mismo nombre y tipo
    $existing = db_fetch_one(
        "SELECT id FROM device_config_profiles WHERE name = ? AND device_type = ?",
        [$name, $device_type]
    );
    
    if ($existing) {
        return ['success' => false, 'error' => 'Ya existe un perfil con ese nombre para este tipo de dispositivo'];
    }
    
    // Obtener configuración base
    if ($source_profile_id) {
        // Copiar de perfil existente
        $source_profile = db_fetch_one("SELECT config_json FROM device_config_profiles WHERE id = ?", [$source_profile_id]);
        if (!$source_profile) {
            return ['success' => false, 'error' => 'Perfil fuente no encontrado'];
        }
        $config_json = $source_profile['config_json'];
    } else {
        // Usar configuración por defecto del sistema
        $default_config = DeviceConfigManager::getDefaultConfig();
        $config_json = json_encode($default_config);
    }
    
    // Si se marca como default, quitar default de otros perfiles del mismo tipo
    if ($is_default) {
        db_update('device_config_profiles', 
            ['is_default' => 0], 
            'device_type = ? OR device_type IS NULL',
            [$device_type]
        );
    }
    
    // Crear perfil
    $profile_id = db_insert('device_config_profiles', [
        'name' => $name,
        'description' => $description,
        'device_type' => $device_type ?: null,
        'config_json' => $config_json,
        'is_default' => $is_default,
        'created_by' => $_SESSION['user_id'],
        'created_at' => date('Y-m-d H:i:s')
    ]);
    
    if ($profile_id) {
        return ['success' => true, 'message' => "Perfil '$name' creado exitosamente"];
    } else {
        return ['success' => false, 'error' => 'Error al crear el perfil'];
    }
}

function handleDeleteProfile() {
    $profile_id = (int)$_POST['profile_id'];
    
    // Verificar que el perfil existe
    $profile = db_fetch_one("SELECT * FROM device_config_profiles WHERE id = ?", [$profile_id]);
    if (!$profile) {
        return ['success' => false, 'error' => 'Perfil no encontrado'];
    }
    
    // Verificar que no hay dispositivos usando este perfil
    $devices_using = db_fetch_one(
        "SELECT COUNT(*) as count FROM devices WHERE JSON_EXTRACT(config_json, '$.profile_id') = ?",
        [$profile_id]
    )['count'];
    
    if ($devices_using > 0) {
        return ['success' => false, 'error' => "No se puede eliminar: $devices_using dispositivos están usando este perfil"];
    }
    
    // Eliminar perfil
    $result = db_delete('device_config_profiles', 'id = ?', [$profile_id]);
    
    if ($result) {
        return ['success' => true, 'message' => "Perfil '{$profile['name']}' eliminado exitosamente"];
    } else {
        return ['success' => false, 'error' => 'Error al eliminar el perfil'];
    }
}

function handleSetDefault() {
    $profile_id = (int)$_POST['profile_id'];
    
    $profile = db_fetch_one("SELECT * FROM device_config_profiles WHERE id = ?", [$profile_id]);
    if (!$profile) {
        return ['success' => false, 'error' => 'Perfil no encontrado'];
    }
    
    // Quitar default de otros perfiles del mismo tipo
    db_update('device_config_profiles', 
        ['is_default' => 0], 
        'device_type = ? OR (device_type IS NULL AND ? IS NULL)',
        [$profile['device_type'], $profile['device_type']]
    );
    
    // Establecer como default
    $result = db_update('device_config_profiles', 
        ['is_default' => 1], 
        'id = ?', 
        [$profile_id]
    );
    
    if ($result !== false) {
        return ['success' => true, 'message' => "Perfil '{$profile['name']}' establecido como predeterminado"];
    } else {
        return ['success' => false, 'error' => 'Error al establecer perfil predeterminado'];
    }
}

// Mostrar mensajes de GET (redirects)
if (isset($_GET['success'])) {
    $success_message = $_GET['success'];
}
if (isset($_GET['error'])) {
    $error_message = $_GET['error'];
}
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Perfiles de Configuración</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Perfiles</li>
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

            <!-- Información de perfiles -->
            <div class="row">
                <div class="col-md-12">
                    <div class="card card-info">
                        <div class="card-header">
                            <h3 class="card-title">¿Qué son los perfiles de configuración?</h3>
                            <div class="card-tools">
                                <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                    <i class="fas fa-minus"></i>
                                </button>
                            </div>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-8">
                                    <p>Los <strong>perfiles de configuración</strong> son plantillas predefinidas que permiten aplicar configuraciones consistentes y optimizadas a múltiples dispositivos de forma rápida y eficiente.</p>
                                    
                                    <h6><i class="fas fa-star text-warning"></i> Beneficios:</h6>
                                    <ul class="list-unstyled">
                                        <li><i class="fas fa-check text-success"></i> <strong>Estandarización:</strong> Configuraciones consistentes para operaciones similares</li>
                                        <li><i class="fas fa-check text-success"></i> <strong>Eficiencia:</strong> Aplicar configuraciones complejas con un solo clic</li>
                                        <li><i class="fas fa-check text-success"></i> <strong>Mejores prácticas:</strong> Configuraciones probadas y optimizadas</li>
                                        <li><i class="fas fa-check text-success"></i> <strong>Gestión centralizada:</strong> Actualizar múltiples dispositivos simultáneamente</li>
                                    </ul>
                                </div>
                                <div class="col-md-4">
                                    <h6><i class="fas fa-lightbulb text-warning"></i> Ejemplos de uso:</h6>
                                    <div class="small">
                                        <span class="badge badge-primary">Turno Diurno</span>
                                        <span class="badge badge-dark">Turno Nocturno</span>
                                        <span class="badge badge-warning">Zona Construcción</span>
                                        <span class="badge badge-success">Ahorro Energía</span>
                                        <span class="badge badge-info">Alta Precisión</span>
                                        <span class="badge badge-secondary">Mantenimiento</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Estadísticas rápidas -->
            <div class="row">
                <?php
                $stats = [
                    'total' => count($profiles),
                    'by_type' => [],
                    'default_profiles' => 0,
                    'total_devices_using' => 0
                ];
                
                foreach ($profiles as $profile) {
                    $type = $profile['device_type'] ?: 'Universal';
                    if (!isset($stats['by_type'][$type])) {
                        $stats['by_type'][$type] = 0;
                    }
                    $stats['by_type'][$type]++;
                    
                    if ($profile['is_default']) {
                        $stats['default_profiles']++;
                    }
                    
                    $stats['total_devices_using'] += $profile['devices_using'];
                }
                ?>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-info">
                        <div class="inner">
                            <h3><?php echo $stats['total']; ?></h3>
                            <p>Total Perfiles</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-layer-group"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-success">
                        <div class="inner">
                            <h3><?php echo $stats['default_profiles']; ?></h3>
                            <p>Perfiles por Defecto</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-star"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-warning">
                        <div class="inner">
                            <h3><?php echo count($stats['by_type']); ?></h3>
                            <p>Tipos de Dispositivo</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-microchip"></i>
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-3 col-6">
                    <div class="small-box bg-primary">
                        <div class="inner">
                            <h3><?php echo $stats['total_devices_using']; ?></h3>
                            <p>Dispositivos Usando Perfiles</p>
                        </div>
                        <div class="icon">
                            <i class="fas fa-wifi"></i>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Perfiles existentes -->
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Perfiles Existentes</h3>
                    <div class="card-tools">
                        <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#createProfileModal">
                            <i class="fas fa-plus"></i> Nuevo Perfil
                        </button>
                    </div>
                </div>
                
                <div class="card-body">
                    <?php if (empty($profiles)): ?>
                        <div class="text-center py-4">
                            <i class="fas fa-layer-group fa-3x text-muted mb-3"></i>
                            <h4 class="text-muted">No hay perfiles configurados</h4>
                            <p class="text-muted">Cree el primer perfil para comenzar a estandarizar configuraciones.</p>
                            <button type="button" class="btn btn-primary" data-toggle="modal" data-target="#createProfileModal">
                                <i class="fas fa-plus"></i> Crear Primer Perfil
                            </button>
                        </div>
                    <?php else: ?>
                        <div class="row">
                            <?php foreach ($profiles as $profile): ?>
                                <div class="col-md-6 col-lg-4">
                                    <div class="card card-outline <?php echo $profile['is_default'] ? 'card-warning' : 'card-secondary'; ?>">
                                        <div class="card-header">
                                            <h5 class="card-title">
                                                <?php if ($profile['is_default']): ?>
                                                    <i class="fas fa-star text-warning" title="Perfil por defecto"></i>
                                                <?php endif; ?>
                                                <?php echo htmlspecialchars($profile['name']); ?>
                                            </h5>
                                            <div class="card-tools">
                                                <div class="dropdown">
                                                    <button class="btn btn-tool" data-toggle="dropdown">
                                                        <i class="fas fa-ellipsis-v"></i>
                                                    </button>
                                                    <div class="dropdown-menu dropdown-menu-right">
                                                        <a class="dropdown-item" href="#" onclick="editProfile(<?php echo $profile['id']; ?>)">
                                                            <i class="fas fa-edit"></i> Editar
                                                        </a>
                                                        <a class="dropdown-item" href="#" onclick="duplicateProfile(<?php echo $profile['id']; ?>)">
                                                            <i class="fas fa-copy"></i> Duplicar
                                                        </a>
                                                        <a class="dropdown-item" href="#" onclick="exportProfile(<?php echo $profile['id']; ?>)">
                                                            <i class="fas fa-download"></i> Exportar
                                                        </a>
                                                        <div class="dropdown-divider"></div>
                                                        <?php if (!$profile['is_default']): ?>
                                                        <a class="dropdown-item" href="#" onclick="setAsDefault(<?php echo $profile['id']; ?>)">
                                                            <i class="fas fa-star"></i> Marcar como Defecto
                                                        </a>
                                                        <?php endif; ?>
                                                        <a class="dropdown-item text-danger" href="#" onclick="deleteProfile(<?php echo $profile['id']; ?>, '<?php echo htmlspecialchars($profile['name']); ?>')">
                                                            <i class="fas fa-trash"></i> Eliminar
                                                        </a>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="card-body">
                                            <p class="text-muted small">
                                                <?php echo htmlspecialchars($profile['description'] ?: 'Sin descripción'); ?>
                                            </p>
                                            
                                            <div class="row text-center">
                                                <div class="col-6">
                                                    <div class="description-block border-right">
                                                        <span class="description-percentage text-info">
                                                            <i class="fas fa-microchip"></i>
                                                        </span>
                                                        <h5 class="description-header"><?php echo htmlspecialchars($profile['device_type'] ?: 'Universal'); ?></h5>
                                                        <span class="description-text">TIPO</span>
                                                    </div>
                                                </div>
                                                <div class="col-6">
                                                    <div class="description-block">
                                                        <span class="description-percentage text-success">
                                                            <i class="fas fa-wifi"></i>
                                                        </span>
                                                        <h5 class="description-header"><?php echo $profile['devices_using']; ?></h5>
                                                        <span class="description-text">DISPOSITIVOS</span>
                                                    </div>
                                                </div>
                                            </div>
                                            
                                            <div class="mt-3">
                                                <small class="text-muted">
                                                    <i class="fas fa-user"></i> 
                                                    Creado por <?php echo htmlspecialchars($profile['created_by_full_name'] ?: $profile['created_by_name'] ?: 'Sistema'); ?>
                                                    <br>
                                                    <i class="fas fa-calendar"></i> 
                                                    <?php echo date('d/m/Y', strtotime($profile['created_at'])); ?>
                                                </small>
                                            </div>
                                        </div>
                                        <div class="card-footer">
                                            <button type="button" class="btn btn-primary btn-sm" onclick="viewProfile(<?php echo $profile['id']; ?>)">
                                                <i class="fas fa-eye"></i> Ver Configuración
                                            </button>
                                            <button type="button" class="btn btn-success btn-sm" onclick="applyProfileToDevices(<?php echo $profile['id']; ?>)">
                                                <i class="fas fa-magic"></i> Aplicar a Dispositivos
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            <?php endforeach; ?>
                        </div>
                    <?php endif; ?>
                </div>
            </div>
        </div>
    </section>
</div>

<!-- Modal para crear perfil -->
<div class="modal fade" id="createProfileModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-lg" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Crear Nuevo Perfil</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <form method="POST" action="">
                <input type="hidden" name="action" value="create_profile">
                <div class="modal-body">
                    <div class="form-group">
                        <label for="profile_name">Nombre del Perfil *</label>
                        <input type="text" class="form-control" id="profile_name" name="profile_name" required
                               placeholder="Ej: Turno Nocturno, Zona Construcción, etc.">
                    </div>
                    
                    <div class="form-group">
                        <label for="profile_description">Descripción</label>
                        <textarea class="form-control" id="profile_description" name="profile_description" rows="3"
                                  placeholder="Descripción del propósito y características de este perfil"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="device_type">Tipo de Dispositivo</label>
                        <select class="form-control" id="device_type" name="device_type">
                            <option value="">Universal (todos los tipos)</option>
                            <?php foreach ($device_types as $type): ?>
                                <option value="<?php echo htmlspecialchars($type['device_type']); ?>">
                                    <?php echo htmlspecialchars($type['device_type']); ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                        <small class="form-text text-muted">
                            Dejar en "Universal" para que sea compatible con todos los tipos de dispositivo
                        </small>
                    </div>
                    
                    <div class="form-group">
                        <label for="source_profile_id">Configuración Base</label>
                        <select class="form-control" id="source_profile_id" name="source_profile_id">
                            <option value="">Configuración por defecto del sistema</option>
                            <?php foreach ($profiles as $p): ?>
                                <option value="<?php echo $p['id']; ?>">
                                    Copiar de: <?php echo htmlspecialchars($p['name']); ?>
                                    <?php if ($p['device_type']): ?>
                                        (<?php echo htmlspecialchars($p['device_type']); ?>)
                                    <?php endif; ?>
                                </option>
                            <?php endforeach; ?>
                        </select>
                        <small class="form-text text-muted">
                            Seleccione un perfil existente para copiar su configuración, o use la configuración por defecto
                        </small>
                    </div>
                    
                    <div class="form-group">
                        <div class="custom-control custom-checkbox">
                            <input type="checkbox" class="custom-control-input" id="is_default" name="is_default">
                            <label class="custom-control-label" for="is_default">
                                Establecer como perfil por defecto
                            </label>
                        </div>
                        <small class="form-text text-muted">
                            El perfil por defecto se aplicará automáticamente a nuevos dispositivos de este tipo
                        </small>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Crear Perfil
                    </button>
                </div>
            </form>
        </div>
    </div>
</div>

<!-- Modal para ver configuración de perfil -->
<div class="modal fade" id="viewProfileModal" tabindex="-1" role="dialog">
    <div class="modal-dialog modal-xl" role="document">
        <div class="modal-content">
            <div class="modal-header">
                <h5 class="modal-title">Configuración del Perfil</h5>
                <button type="button" class="close" data-dismiss="modal">
                    <span aria-hidden="true">&times;</span>
                </button>
            </div>
            <div class="modal-body">
                <div id="profileConfigContent">
                    <!-- Contenido se carga dinámicamente -->
                </div>
            </div>
        </div>
    </div>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
// Ver configuración de perfil
function viewProfile(profileId) {
    $('#viewProfileModal').modal('show');
    $('#profileConfigContent').html(`
        <div class="text-center">
            <i class="fas fa-spinner fa-spin fa-2x"></i>
            <h4>Cargando configuración...</h4>
        </div>
    `);
    
    $.ajax({
        url: 'view_profile_config.php',
        method: 'GET',
        data: { profile_id: profileId },
        success: function(response) {
            $('#profileConfigContent').html(response);
        },
        error: function() {
            $('#profileConfigContent').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    Error al cargar la configuración del perfil.
                </div>
            `);
        }
    });
}

// Editar perfil
function editProfile(profileId) {
    window.location.href = `edit_profile.php?id=${profileId}`;
}

// Duplicar perfil
function duplicateProfile(profileId) {
    $('#source_profile_id').val(profileId);
    $('#createProfileModal').modal('show');
}

// Exportar perfil
function exportProfile(profileId) {
    window.open(`export_profile.php?id=${profileId}`, '_blank');
}

// Establecer como por defecto
function setAsDefault(profileId) {
    if (confirm('¿Establecer este perfil como predeterminado?')) {
        const form = document.createElement('form');
        form.method = 'POST';
        form.innerHTML = `
            <input type="hidden" name="action" value="set_default">
            <input type="hidden" name="profile_id" value="${profileId}">
        `;
        document.body.appendChild(form);
        form.submit();
    }
}

// Eliminar perfil
function deleteProfile(profileId, profileName) {
    if (confirm(`¿Está seguro de eliminar el perfil "${profileName}"?\n\nEsta acción no se puede deshacer.`)) {
        const form = document.createElement('form');
        form.method = 'POST';
        form.innerHTML = `
            <input type="hidden" name="action" value="delete_profile">
            <input type="hidden" name="profile_id" value="${profileId}">
        `;
        document.body.appendChild(form);
        form.submit();
    }
}

// Aplicar perfil a dispositivos
function applyProfileToDevices(profileId) {
    window.location.href = `apply_profile_bulk.php?profile_id=${profileId}`;
}
</script>

<style>
.description-block {
    padding: 0 10px;
}

.description-block .description-header {
    margin: 10px 0 7px 0;
    font-size: 16px;
    font-weight: bold;
}

.description-block .description-text {
    font-size: 13px;
    text-transform: uppercase;
    font-weight: bold;
    color: #999;
}

.card-outline.card-warning {
    border-top: 3px solid #ffc107;
}

.dropdown-menu {
    min-width: 160px;
}

.small-box .icon {
    color: rgba(255,255,255,0.15);
}
</style>