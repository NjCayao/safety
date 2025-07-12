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

// Verificar si se proporcionó un ID de operador
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$operatorId = $_GET['id'];

// Obtener información del operador
$operator = db_fetch_one(
    "SELECT * FROM operators WHERE id = ?",
    [$operatorId]
);

// Si no se encuentra el operador, redirigir al listado
if (!$operator) {
    $_SESSION['error_message'] = "Operador no encontrado.";
    header('Location: index.php');
    exit;
}

// Obtener máquinas activas disponibles
$availableMachines = db_fetch_all(
    "SELECT m.* FROM machines m 
     WHERE m.status = 'active' 
     AND m.id NOT IN (
         SELECT machine_id FROM operator_machine WHERE is_current = 1
     )
     ORDER BY m.name ASC"
);

// Obtener asignaciones actuales del operador
$currentAssignments = db_fetch_all(
    "SELECT om.*, m.name as machine_name, m.type as machine_type 
     FROM operator_machine om 
     JOIN machines m ON om.machine_id = m.id 
     WHERE om.operator_id = ? AND om.is_current = 1
     ORDER BY om.assigned_date DESC",
    [$operatorId]
);

// Variables para manejo de errores y mensajes
$errors = [];
$successMessage = '';

// Procesar formulario si se envió
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener y validar datos del formulario
    $machineId = $_POST['machine_id'] ?? '';
    
    // Validaciones
    if (empty($machineId)) {
        $errors[] = 'Debe seleccionar una máquina';
    } else {
        // Verificar si la máquina existe y está disponible
        $machine = db_fetch_one(
            "SELECT * FROM machines WHERE id = ? AND status = 'active'",
            [$machineId]
        );
        
        if (!$machine) {
            $errors[] = 'La máquina seleccionada no existe o no está activa';
        } else {
            // Verificar si la máquina ya está asignada a otro operador
            $existingAssignment = db_fetch_one(
                "SELECT * FROM operator_machine WHERE machine_id = ? AND is_current = 1",
                [$machineId]
            );
            
            if ($existingAssignment) {
                $errors[] = 'La máquina seleccionada ya está asignada a otro operador';
            }
        }
    }
    
    // Si no hay errores, procesar la asignación
    if (empty($errors)) {
        // Crear la asignación
        $assignmentData = [
            'operator_id' => $operatorId,
            'machine_id' => $machineId,
            'assigned_date' => date('Y-m-d H:i:s'),
            'is_current' => 1
        ];
        
        $result = db_insert('operator_machine', $assignmentData);
        
        if ($result !== false) {
            // Registrar en el log
            log_system_message(
                'info',
                'Máquina asignada al operador: ' . $operator['name'] . ' (ID: ' . $operatorId . ')',
                $machineId,
                'Usuario: ' . $_SESSION['username']
            );
            
            $successMessage = 'Máquina asignada exitosamente.';
            
            // Actualizar la lista de asignaciones y máquinas disponibles
            $currentAssignments = db_fetch_all(
                "SELECT om.*, m.name as machine_name, m.type as machine_type 
                 FROM operator_machine om 
                 JOIN machines m ON om.machine_id = m.id 
                 WHERE om.operator_id = ? AND om.is_current = 1
                 ORDER BY om.assigned_date DESC",
                [$operatorId]
            );
            
            $availableMachines = db_fetch_all(
                "SELECT m.* FROM machines m 
                 WHERE m.status = 'active' 
                 AND m.id NOT IN (
                     SELECT machine_id FROM operator_machine WHERE is_current = 1
                 )
                 ORDER BY m.name ASC"
            );
        } else {
            $errors[] = 'Error al asignar la máquina en la base de datos';
        }
    }
}

// Definir título de la página
$pageTitle = 'Asignar Máquina a: ' . htmlspecialchars($operator['name']);

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Operadores' => 'index.php',
    'Detalles' => 'view.php?id=' . $operatorId,
    'Asignar Máquina' => ''
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

<div class="row">
    <div class="col-md-5">
        <!-- Perfil del operador -->
        <div class="card card-primary card-outline">
            <div class="card-body box-profile">
                <div class="text-center">
                    <?php if (!empty($operator['photo_path'])): ?>
                        <img class="profile-user-img img-fluid img-circle" 
                             src="<?php echo htmlspecialchars($operator['photo_path']); ?>" 
                             alt="Foto del operador">
                    <?php else: ?>
                        <img class="profile-user-img img-fluid img-circle" 
                             src="../../assets/dist/img/user-default.jpg" 
                             alt="Foto del operador">
                    <?php endif; ?>
                </div>

                <h3 class="profile-username text-center"><?php echo htmlspecialchars($operator['name']); ?></h3>
                <p class="text-muted text-center"><?php echo htmlspecialchars($operator['position'] ?? 'Sin cargo asignado'); ?></p>

                <ul class="list-group list-group-unbordered mb-3">
                    <li class="list-group-item">
                        <b>ID</b> <a class="float-right"><?php echo htmlspecialchars($operator['id']); ?></a>
                    </li>
                    <li class="list-group-item">
                        <b>Estado</b> 
                        <span class="float-right">
                            <?php if ($operator['status'] == 'active'): ?>
                                <span class="badge badge-success">Activo</span>
                            <?php else: ?>
                                <span class="badge badge-danger">Inactivo</span>
                            <?php endif; ?>
                        </span>
                    </li>
                    <li class="list-group-item">
                        <b>Asignaciones Actuales</b> 
                        <span class="float-right badge badge-info"><?php echo count($currentAssignments); ?></span>
                    </li>
                </ul>
            </div>
        </div>

        <!-- Formulario de asignación -->
        <div class="card card-primary">
            <div class="card-header">
                <h3 class="card-title">Asignar Máquina</h3>
            </div>
            <div class="card-body">
                <?php if (empty($availableMachines)): ?>
                    <div class="alert alert-warning">
                        <i class="icon fas fa-exclamation-triangle"></i> No hay máquinas disponibles para asignar.
                    </div>
                    <p>Todas las máquinas activas ya están asignadas a otros operadores.</p>
                    <a href="../machines/create.php" class="btn btn-primary">
                        <i class="fas fa-plus"></i> Crear Nueva Máquina
                    </a>
                <?php else: ?>
                    <form action="assign.php?id=<?php echo $operatorId; ?>" method="POST">
                        <div class="form-group">
                            <label for="machine_id">Seleccione Máquina:</label>
                            <select class="form-control select2" id="machine_id" name="machine_id" required>
                                <option value="">-- Seleccionar Máquina --</option>
                                <?php foreach ($availableMachines as $machine): ?>
                                    <option value="<?php echo $machine['id']; ?>">
                                        <?php echo htmlspecialchars($machine['name']); ?> 
                                        (<?php echo htmlspecialchars($machine['type']); ?> - 
                                        <?php echo htmlspecialchars($machine['location'] ?? 'Sin ubicación'); ?>)
                                    </option>
                                <?php endforeach; ?>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary">
                            <i class="fas fa-link"></i> Asignar Máquina
                        </button>
                    </form>
                <?php endif; ?>
            </div>
        </div>
    </div>
    
    <div class="col-md-7">
        <!-- Asignaciones actuales -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Asignaciones Actuales</h3>
            </div>
            <div class="card-body">
                <?php if (empty($currentAssignments)): ?>
                    <div class="alert alert-info">
                        <i class="icon fas fa-info"></i> Este operador no tiene máquinas asignadas actualmente.
                    </div>
                <?php else: ?>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Máquina</th>
                                    <th>Tipo</th>
                                    <th>Fecha Asignación</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($currentAssignments as $assignment): ?>
                                    <tr>
                                        <td>
                                            <a href="../machines/view.php?id=<?php echo htmlspecialchars($assignment['machine_id']); ?>">
                                                <?php echo htmlspecialchars($assignment['machine_name']); ?>
                                            </a>
                                        </td>
                                        <td><?php echo htmlspecialchars($assignment['machine_type']); ?></td>
                                        <td><?php echo date('d/m/Y H:i', strtotime($assignment['assigned_date'])); ?></td>
                                        <td>
                                            <a href="../machines/unassign.php?id=<?php echo $assignment['id']; ?>&operator_id=<?php echo $operator['id']; ?>" class="btn btn-sm btn-danger" onclick="return confirm('¿Está seguro que desea desasignar esta máquina?')">
                                                <i class="fas fa-unlink"></i> Desasignar
                                            </a>
                                        </td>
                                    </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    </div>
                <?php endif; ?>
            </div>
        </div>

        <!-- Lista de máquinas disponibles -->
        <div class="card">
            <div class="card-header">
                <h3 class="card-title">Máquinas Disponibles</h3>
                <div class="card-tools">
                    <a href="../machines/create.php" class="btn btn-sm btn-primary">
                        <i class="fas fa-plus"></i> Nueva Máquina
                    </a>
                </div>
            </div>
            <div class="card-body">
                <?php if (empty($availableMachines)): ?>
                    <div class="alert alert-warning">
                        <i class="icon fas fa-exclamation-triangle"></i> No hay máquinas disponibles para asignar.
                    </div>
                <?php else: ?>
                    <div class="table-responsive">
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Nombre</th>
                                    <th>Tipo</th>
                                    <th>Ubicación</th>
                                    <th>Acciones</th>
                                </tr>
                            </thead>
                            <tbody>
                                <?php foreach ($availableMachines as $machine): ?>
                                    <tr>
                                        <td><?php echo htmlspecialchars($machine['id']); ?></td>
                                        <td><?php echo htmlspecialchars($machine['name']); ?></td>
                                        <td><?php echo htmlspecialchars($machine['type']); ?></td>
                                        <td><?php echo htmlspecialchars($machine['location'] ?? 'Sin ubicación'); ?></td>
                                        <td>
                                            <div class="btn-group">
                                                <a href="../machines/view.php?id=<?php echo $machine['id']; ?>" class="btn btn-sm btn-info">
                                                    <i class="fas fa-eye"></i>
                                                </a>
                                                <form action="assign.php?id=<?php echo $operatorId; ?>" method="POST" style="display: inline;">
                                                    <input type="hidden" name="machine_id" value="<?php echo $machine['id']; ?>">
                                                    <button type="submit" class="btn btn-sm btn-success">
                                                        <i class="fas fa-link"></i> Asignar
                                                    </button>
                                                </form>
                                            </div>
                                        </td>
                                    </tr>
                                <?php endforeach; ?>
                            </tbody>
                        </table>
                    </div>
                <?php endif; ?>
            </div>
        </div>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<div class="btn-group">
    <a href="view.php?id=' . $operatorId . '" class="btn btn-secondary">
        <i class="fas fa-arrow-left"></i> Volver
    </a>
    <a href="../machines/create.php" class="btn btn-primary">
        <i class="fas fa-plus"></i> Nueva Máquina
    </a>
</div>
';

// JavaScript específico para esta página
$extraJs = '
<script>
  $(function () {
    //Initialize Select2 Elements
    $(".select2").select2({
      placeholder: "Seleccionar máquina",
      allowClear: true
    });
  });
</script>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>