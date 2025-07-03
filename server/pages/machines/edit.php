<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Verificar si se proporcionó un ID de máquina
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$machineId = $_GET['id'];

// Obtener información de la máquina
$machine = db_fetch_one(
    "SELECT * FROM machines WHERE id = ?",
    [$machineId]
);

// Si no se encuentra la máquina, redirigir al listado
if (!$machine) {
    $_SESSION['error_message'] = "Máquina no encontrada.";
    header('Location: index.php');
    exit;
}

// Variables para manejo de errores y mensajes
$errors = [];
$successMessage = '';

// Procesar formulario si se envió
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener y validar datos del formulario
    $name = trim($_POST['name'] ?? '');
    $type = trim($_POST['type'] ?? '');
    $location = trim($_POST['location'] ?? '');
    $status = $_POST['status'] ?? 'active';
    $notes = trim($_POST['notes'] ?? '');
    
    // Validaciones
    if (empty($name)) {
        $errors[] = 'El nombre es obligatorio';
    }
    
    if (empty($type)) {
        $errors[] = 'El tipo de máquina es obligatorio';
    }
    
    // Si no hay errores, procesar los datos
    if (empty($errors)) {
        // Datos de la máquina para actualizar
        $machineData = [
            'name' => $name,
            'type' => $type,
            'location' => $location,
            'status' => $status,
            'notes' => $notes
        ];
        
        $result = db_update('machines', $machineData, 'id = ?', [$machineId]);
        
        if ($result !== false) {
            // Registrar en el log
            log_system_message(
                'info',
                'Máquina actualizada: ' . $name . ' (ID: ' . $machineId . ')',
                $machineId,
                'Usuario: ' . $_SESSION['username']
            );
            
            $successMessage = 'Máquina actualizada exitosamente.';
            
            // Actualizar la variable $machine para mostrar datos actualizados
            $machine = db_fetch_one(
                "SELECT * FROM machines WHERE id = ?",
                [$machineId]
            );
        } else {
            $errors[] = 'Error al actualizar la máquina en la base de datos';
        }
    }
}

// Obtener tipos de máquinas existentes para sugerencias
$existingTypes = db_fetch_all(
    "SELECT DISTINCT type FROM machines ORDER BY type"
);

// Definir título de la página
$pageTitle = 'Editar Máquina: ' . htmlspecialchars($machine['name']);

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Máquinas' => 'index.php',
    'Detalles' => 'view.php?id=' . $machineId,
    'Editar' => ''
];

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Mensajes de error o éxito -->
<?php if (!empty($errors)): ?>
    <div class="alert alert-danger">
        <h5><i class="icon fas fa-ban"></i> Se encontraron errores:</h5>
        <ul>
            <?php foreach ($errors as $error): ?>
                <li><?php echo $error; ?></li>
            <?php endforeach; ?>
        </ul>
    </div>
<?php endif; ?>

<?php if (!empty($successMessage)): ?>
    <div class="alert alert-success">
        <h5><i class="icon fas fa-check"></i> Éxito!</h5>
        <?php echo $successMessage; ?>
    </div>
<?php endif; ?>

<!-- Formulario de edición -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Información de la Máquina</h3>
    </div>
    <div class="card-body">
        <form action="edit.php?id=<?php echo $machineId; ?>" method="POST">
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="name">Nombre <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="name" name="name" value="<?php echo htmlspecialchars($machine['name']); ?>" required>
                        <small class="form-text text-muted">Nombre o identificador de la máquina</small>
                    </div>
                    <div class="form-group">
                        <label for="type">Tipo <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="type" name="type" value="<?php echo htmlspecialchars($machine['type']); ?>" list="type-suggestions" required>
                        <datalist id="type-suggestions">
                            <?php foreach ($existingTypes as $existingType): ?>
                                <option value="<?php echo htmlspecialchars($existingType['type']); ?>">
                            <?php endforeach; ?>
                        </datalist>
                        <small class="form-text text-muted">Tipo o categoría de la máquina (ej: Excavadora, Grúa, Cargadora)</small>
                    </div>
                    <div class="form-group">
                        <label for="location">Ubicación</label>
                        <input type="text" class="form-control" id="location" name="location" value="<?php echo htmlspecialchars($machine['location'] ?? ''); ?>">
                        <small class="form-text text-muted">Ubicación física de la máquina</small>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="status">Estado</label>
                        <select class="form-control" id="status" name="status">
                            <option value="active" <?php echo ($machine['status'] === 'active') ? 'selected' : ''; ?>>Activa</option>
                            <option value="maintenance" <?php echo ($machine['status'] === 'maintenance') ? 'selected' : ''; ?>>En Mantenimiento</option>
                            <option value="inactive" <?php echo ($machine['status'] === 'inactive') ? 'selected' : ''; ?>>Inactiva</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="notes">Notas</label>
                        <textarea class="form-control" id="notes" name="notes" rows="5"><?php echo htmlspecialchars($machine['notes'] ?? ''); ?></textarea>
                        <small class="form-text text-muted">Información adicional, especificaciones o comentarios sobre la máquina</small>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-12">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Guardar Cambios
                    </button>
                    <a href="view.php" class="btn btn-secondary">
                        <i class="fas fa-times"></i> Cancelar
                    </a>
                </div>
            </div>
        </form>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<div class="btn-group">
    <a href="view.php?id=' . $machineId . '" class="btn btn-secondary">
        <i class="fas fa-arrow-left"></i> Volver
    </a>
</div>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>