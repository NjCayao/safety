<?php
// pages/devices/edit.php - VERSIÓN CORREGIDA

// Asegurar que no hay salida antes de headers
ob_start();

// Primero iniciar sesión si no está iniciada
if (session_status() === PHP_SESSION_NONE) {
    session_start();
}

// Cargar configuración
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/auth.php';

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    ob_end_clean();
    header("Location: " . BASE_URL . "/login.php");
    exit();
}

// Obtener ID del dispositivo
$device_id = $_GET['id'] ?? null;

if (!$device_id) {
    ob_end_clean();
    header("Location: index.php?error=" . urlencode("ID de dispositivo no especificado"));
    exit();
}

// Obtener información del dispositivo
$device = db_fetch_one("SELECT * FROM devices WHERE id = ?", [$device_id]);

if (!$device) {
    ob_end_clean();
    header("Location: index.php?error=" . urlencode("Dispositivo no encontrado"));
    exit();
}

// Procesar formulario ANTES de mostrar contenido
$success_message = '';
$error_message = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $name = trim($_POST['name'] ?? '');
    $device_type = $_POST['device_type'] ?? '';
    $machine_id = $_POST['machine_id'] ?? null;
    $location = trim($_POST['location'] ?? '');
    $status = $_POST['status'] ?? 'offline';
    $reset_api_key = isset($_POST['reset_api_key']);
    
    // Validar datos requeridos
    if (empty($device_type)) {
        $error_message = "El tipo de dispositivo es requerido";
    } else {
        try {
            // Preparar datos para actualizar
            $updateData = [
                'name' => $name,
                'device_type' => $device_type,
                'machine_id' => $machine_id ?: null,
                'location' => $location,
                'status' => $status
            ];
            
            // Si se solicita resetear la API key
            $new_api_key = null;
            if ($reset_api_key) {
                $new_api_key = bin2hex(random_bytes(32));
                $updateData['api_key'] = password_hash($new_api_key, PASSWORD_DEFAULT);
            }
            
            // Actualizar dispositivo
            $result = db_update('devices', $updateData, 'id = ?', [$device_id]);
            
            if ($result !== false) {
                if ($reset_api_key && $new_api_key) {
                    // Guardar la nueva API key para mostrarla
                    $_SESSION['new_device_api_key'] = $new_api_key;
                    $_SESSION['new_device_id'] = $device['device_id'];
                    
                    ob_end_clean();
                    header("Location: view.php?id=$device_id&new=1");
                    exit();
                } else {
                    ob_end_clean();
                    header("Location: index.php?msg=" . urlencode("Dispositivo actualizado exitosamente"));
                    exit();
                }
            } else {
                $error_message = "Error al actualizar el dispositivo en la base de datos";
            }
        } catch (Exception $e) {
            $error_message = "Error al procesar la actualización: " . $e->getMessage();
        }
    }
}

// Obtener lista de máquinas para el select
try {
    $machines = db_fetch_all("SELECT id, name FROM machines WHERE status = 'active' ORDER BY name");
} catch (Exception $e) {
    $machines = [];
    if (empty($error_message)) {
        $error_message = "Error al cargar la lista de máquinas";
    }
}

// Definir el título de la página
$pageTitle = "Editar Dispositivo";

// Limpiar buffer y empezar salida
ob_end_clean();

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
                    <h1>Editar Dispositivo</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Dispositivos</a></li>
                        <li class="breadcrumb-item active">Editar</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            
            <?php if (!empty($success_message)): ?>
                <div class="alert alert-success alert-dismissible fade show" role="alert">
                    <i class="fas fa-check-circle"></i> <?php echo htmlspecialchars($success_message); ?>
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            <?php endif; ?>
            
            <?php if (!empty($error_message)): ?>
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    <i class="fas fa-exclamation-triangle"></i> <?php echo htmlspecialchars($error_message); ?>
                    <button type="button" class="close" data-dismiss="alert">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
            <?php endif; ?>

            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Información del Dispositivo</h3>
                </div>
                
                <form method="POST" action="" id="editDeviceForm">
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label>ID del Dispositivo</label>
                                    <input type="text" class="form-control" readonly 
                                           value="<?php echo htmlspecialchars($device['device_id']); ?>">
                                    <small class="form-text text-muted">El ID no puede ser modificado</small>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="name">Nombre del Dispositivo</label>
                                    <input type="text" class="form-control" id="name" name="name" 
                                           placeholder="Ej: Raspberry Pi Zona Norte" maxlength="255"
                                           value="<?php echo htmlspecialchars($_POST['name'] ?? $device['name'] ?? ''); ?>">
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="device_type">Tipo de Dispositivo *</label>
                                    <select class="form-control" id="device_type" name="device_type" required>
                                        <option value="">Seleccione un tipo</option>
                                        <?php
                                        $current_device_type = $_POST['device_type'] ?? $device['device_type'];
                                        $device_types = [
                                            'Raspberry Pi' => 'Raspberry Pi',
                                            'Edge Computer' => 'Edge Computer', 
                                            'Industrial PC' => 'Industrial PC',
                                            'Other' => 'Otro'
                                        ];
                                        
                                        foreach ($device_types as $value => $label):
                                        ?>
                                            <option value="<?php echo $value; ?>" 
                                                    <?php echo ($current_device_type === $value) ? 'selected' : ''; ?>>
                                                <?php echo $label; ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="machine_id">Máquina Asignada</label>
                                    <select class="form-control" id="machine_id" name="machine_id">
                                        <option value="">Sin asignar</option>
                                        <?php 
                                        $current_machine_id = $_POST['machine_id'] ?? $device['machine_id'];
                                        foreach ($machines as $machine): 
                                        ?>
                                            <option value="<?php echo $machine['id']; ?>" 
                                                    <?php echo ($current_machine_id == $machine['id']) ? 'selected' : ''; ?>>
                                                <?php echo htmlspecialchars($machine['name']); ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="location">Ubicación</label>
                                    <input type="text" class="form-control" id="location" name="location" 
                                           placeholder="Ej: Zona Norte, Área de Excavación" maxlength="255"
                                           value="<?php echo htmlspecialchars($_POST['location'] ?? $device['location'] ?? ''); ?>">
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="form-group">
                                    <label for="status">Estado</label>
                                    <select class="form-control" id="status" name="status">
                                        <?php
                                        $current_status = $_POST['status'] ?? $device['status'];
                                        $status_options = [
                                            'online' => 'En línea',
                                            'offline' => 'Fuera de línea',
                                            'syncing' => 'Sincronizando',
                                            'error' => 'Error'
                                        ];
                                        
                                        foreach ($status_options as $value => $label):
                                        ?>
                                            <option value="<?php echo $value; ?>" 
                                                    <?php echo ($current_status === $value) ? 'selected' : ''; ?>>
                                                <?php echo $label; ?>
                                            </option>
                                        <?php endforeach; ?>
                                    </select>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-12">
                                <div class="form-group">
                                    <div class="custom-control custom-checkbox">
                                        <input type="checkbox" class="custom-control-input" id="reset_api_key" name="reset_api_key">
                                        <label class="custom-control-label" for="reset_api_key">
                                            <strong>Generar nueva API Key</strong> (esto invalidará la clave actual)
                                        </label>
                                    </div>
                                    <small class="form-text text-muted">
                                        ⚠️ Si genera una nueva API Key, deberá reconfigurar el dispositivo cliente
                                    </small>
                                </div>
                            </div>
                        </div>
                        
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> 
                            <strong>Información de conexión:</strong>
                            <ul class="mb-0 mt-2">
                                <li><strong>Último acceso:</strong> 
                                    <?php echo $device['last_access'] ? date('d/m/Y H:i:s', strtotime($device['last_access'])) : 'Nunca'; ?>
                                </li>
                                <li><strong>Última sincronización:</strong> 
                                    <?php echo $device['last_sync'] ? date('d/m/Y H:i:s', strtotime($device['last_sync'])) : 'Nunca'; ?>
                                </li>
                                <li><strong>IP:</strong> 
                                    <?php echo htmlspecialchars($device['ip_address'] ?? 'No disponible'); ?>
                                </li>
                                <li><strong>Registrado:</strong> 
                                    <?php echo date('d/m/Y H:i:s', strtotime($device['created_at'])); ?>
                                </li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="card-footer">
                        <button type="submit" class="btn btn-primary" id="submitBtn">
                            <i class="fas fa-save"></i> Guardar Cambios
                        </button>
                        <a href="index.php" class="btn btn-secondary">
                            <i class="fas fa-times"></i> Cancelar
                        </a>
                        <a href="view.php?id=<?php echo $device_id; ?>" class="btn btn-info">
                            <i class="fas fa-eye"></i> Ver Detalles
                        </a>
                    </div>
                </form>
            </div>
        </div>
    </section>
</div>

<?php include '../../includes/footer.php'; ?>

<script>
$(document).ready(function() {
    // Confirmación para reset de API key
    $('#reset_api_key').change(function() {
        if ($(this).is(':checked')) {
            if (!confirm('¿Está seguro de generar una nueva API Key?\n\n' +
                        'ADVERTENCIA: Esto invalidará la clave actual y el dispositivo ' +
                        'deberá ser reconfigurado con la nueva clave.\n\n' +
                        '¿Desea continuar?')) {
                $(this).prop('checked', false);
            }
        }
    });
    
    // Validación del formulario
    $('#editDeviceForm').on('submit', function(e) {
        const deviceType = $('#device_type').val();
        
        // Deshabilitar botón de envío para evitar doble submit
        $('#submitBtn').prop('disabled', true);
        
        if (!deviceType) {
            e.preventDefault();
            $('#submitBtn').prop('disabled', false);
            alert('Por favor seleccione el tipo de dispositivo');
            $('#device_type').focus();
            return false;
        }
        
        // Mostrar indicador de carga
        $('#submitBtn').html('<i class="fas fa-spinner fa-spin"></i> Guardando...');
        
        // Si se va a resetear la API key, mostrar confirmación final
        if ($('#reset_api_key').is(':checked')) {
            if (!confirm('CONFIRMACIÓN FINAL:\n\n' +
                        'Se generará una nueva API Key. El dispositivo dejará de funcionar ' +
                        'hasta que configure la nueva clave.\n\n' +
                        '¿Proceder con el cambio?')) {
                e.preventDefault();
                $('#submitBtn').prop('disabled', false).html('<i class="fas fa-save"></i> Guardar Cambios');
                return false;
            }
        }
    });
    
    // Indicador visual para campos modificados
    $('input, select').on('change', function() {
        $(this).addClass('border-warning');
    });
});
</script>

<style>
.border-warning {
    border-color: #ffc107 !important;
}

.custom-control-label {
    cursor: pointer;
}

.alert ul {
    padding-left: 20px;
}
</style>