<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

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

// Definir título de la página
$pageTitle = 'Gestión de Operadores';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Operadores' => ''
];

// Obtener filtros
$status = isset($_GET['status']) ? $_GET['status'] : '';
$license_status = isset($_GET['license_status']) ? $_GET['license_status'] : '';
$search = isset($_GET['search']) ? trim($_GET['search']) : '';

// Configuración de paginación - OBTENER VALOR DE REGISTROS POR PÁGINA
$registros_por_pagina = isset($_GET['per_page']) ? intval($_GET['per_page']) : 20;
// Validar que sea un valor permitido
if (!in_array($registros_por_pagina, [10, 20, 50, 100])) {
    $registros_por_pagina = 20;
}

$pagina_actual = isset($_GET['page']) ? max(1, intval($_GET['page'])) : 1;
$offset = ($pagina_actual - 1) * $registros_por_pagina;

// Primero contar el total de registros para la paginación
$count_sql = "SELECT COUNT(*) as total FROM operators o WHERE 1=1";
$count_params = [];

// Aplicar los mismos filtros para el conteo
if ($status !== '') {
    $count_sql .= " AND o.status = ?";
    $count_params[] = $status;
}

if ($license_status !== '') {
    $count_sql .= " AND o.license_status = ?";
    $count_params[] = $license_status;
}

if ($search !== '') {
    $count_sql .= " AND (o.name LIKE ? OR o.id LIKE ? OR o.position LIKE ? OR o.dni_number LIKE ?)";
    $searchTerm = "%$search%";
    $count_params[] = $searchTerm;
    $count_params[] = $searchTerm;
    $count_params[] = $searchTerm;
    $count_params[] = $searchTerm;
}

// Obtener total de registros
$total_result = db_fetch_one($count_sql, $count_params);
$total_registros = $total_result['total'];
$total_paginas = ceil($total_registros / $registros_por_pagina);

// Construir la consulta SQL principal con paginación
$sql = "SELECT o.*, 
               (SELECT COUNT(*) FROM operator_machine om WHERE om.operator_id = o.id AND om.is_current = 1) as assigned_machines,
               (SELECT m.name FROM operator_machine om JOIN machines m ON om.machine_id = m.id WHERE om.operator_id = o.id AND om.is_current = 1 LIMIT 1) as current_machine
        FROM operators o
        WHERE 1=1";
$params = [];

// Aplicar filtros
if ($status !== '') {
    $sql .= " AND o.status = ?";
    $params[] = $status;
}

if ($license_status !== '') {
    $sql .= " AND o.license_status = ?";
    $params[] = $license_status;
}

if ($search !== '') {
    $sql .= " AND (o.name LIKE ? OR o.id LIKE ? OR o.position LIKE ? OR o.dni_number LIKE ?)";
    $searchTerm = "%$search%";
    $params[] = $searchTerm;
    $params[] = $searchTerm;
    $params[] = $searchTerm;
    $params[] = $searchTerm;
}

// Ordenar por fecha de registro descendente
$sql .= " ORDER BY o.registration_date DESC";

// Agregar límite y offset para paginación
$sql .= " LIMIT ? OFFSET ?";
$params[] = $registros_por_pagina;
$params[] = $offset;

// Ejecutar la consulta
$operators = db_fetch_all($sql, $params);

// Verificar estado de calibración para cada operador
$base_dir = dirname(dirname(dirname(__DIR__))); // Llegar a safety_system

// Agregar campo de calibración a cada operador
for ($i = 0; $i < count($operators); $i++) {
    if (!empty($operators[$i]['dni_number'])) {
        $calibration_file = $base_dir . DIRECTORY_SEPARATOR . 'operators' .
            DIRECTORY_SEPARATOR . 'baseline-json' .
            DIRECTORY_SEPARATOR . $operators[$i]['dni_number'] .
            DIRECTORY_SEPARATOR . 'master_baseline.json';

        $operators[$i]['has_calibration'] = file_exists($calibration_file);
    } else {
        $operators[$i]['has_calibration'] = false;
    }
}

// Mensajes de sesión
$successMessage = $_SESSION['success_message'] ?? '';
$errorMessage = $_SESSION['error_message'] ?? '';

// Limpiar mensajes de sesión después de usarlos
unset($_SESSION['success_message'], $_SESSION['error_message']);

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<?php
// Incluir funciones necesarias
require_once('../../includes/progress_functions.php');

// Configurar zona horaria a Perú
date_default_timezone_set('America/Lima');

// Verificar si hay una operación en curso
$in_progress = is_operation_in_progress('update_encodings');
$progress = get_progress('update_encodings');

// Obtener información del archivo encodings.pkl
$encodings_info = get_encodings_info();

// Mensaje de estado (solo si viene de iniciar la actualización)
$status_msg = '';
if (isset($_GET['msg']) && $_GET['msg'] == 'update_started') {
    $status_msg = '<div class="alert alert-info">La actualización ha Finalizado.</div>';
}
?>

<!-- Reconocimiento Facial -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Reconocimiento Facial</h3>
    </div>
    <div class="card-body">
        <!-- Estado del Reconocimiento Facial -->
        <h4>Estado del Reconocimiento Facial</h4>
        <?php if ($encodings_info['exists']): ?>
            <div class="alert alert-info">
                <p><strong>Última actualización:</strong> <?php echo $encodings_info['modified']; ?></p>
                <p><strong>Tamaño del archivo:</strong> <?php echo round($encodings_info['size'] / 1024, 2); ?> KB</p>
            </div>
        <?php else: ?>
            <div class="alert alert-warning">
                <p>No se ha generado el archivo de reconocimiento facial. Por favor, actualice el sistema.</p>
            </div>
        <?php endif; ?>

        <!-- Mensaje de estado (solo visible durante el inicio de la actualización) -->
        <div id="status-message">
            <?php echo $status_msg; ?>
        </div>

        <!-- Contenedor de progreso (visible solo durante la actualización) -->
        <div id="progress-container" style="<?php echo $in_progress ? '' : 'display:none;'; ?>">
            <div class="progress">
                <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-success"
                    role="progressbar" style="width: <?php echo $progress; ?>%"
                    aria-valuenow="<?php echo $progress; ?>" aria-valuemin="0" aria-valuemax="100">
                    <span id="progress-text"><?php echo $progress; ?>%</span>
                </div>
            </div>
            <p id="status-text" class="mt-2">Actualizando reconocimiento facial...</p>
        </div>

        <!-- Botón de actualización (visible cuando no hay una actualización en curso) -->
        <form id="update-form" action="actions/update_encodings.php" method="post"
            style="<?php echo $in_progress ? 'display:none;' : ''; ?>">
            <button type="submit" class="btn btn-warning">
                <i class="fas fa-sync"></i> Actualizar Reconocimiento Facial
            </button>
        </form>
    </div>
</div>

<!-- JavaScript para actualizar la barra de progreso -->
<script>
    $(document).ready(function() {
        // Si hay una operación en curso, inicia el monitoreo
        <?php if ($in_progress): ?>
            startProgressMonitoring();
        <?php endif; ?>

        function startProgressMonitoring() {
            // Mostrar el contenedor de progreso y ocultar el formulario
            $('#progress-container').show();
            $('#update-form').hide();

            // Función para actualizar el progreso
            function updateProgress() {
                $.ajax({
                    url: 'actions/update_encodings.php?check_progress=1',
                    dataType: 'json',
                    success: function(data) {
                        // Actualiza la barra de progreso
                        $('#progress-bar').css('width', data.progress + '%')
                            .attr('aria-valuenow', data.progress);
                        $('#progress-text').text(data.progress + '%');

                        // Si la operación ha terminado
                        if (data.progress >= 100) {
                            $('#status-text').text('¡Actualización completada!');

                            // Espera un momento y luego recarga la página
                            setTimeout(function() {
                                window.location.href = window.location.pathname; // Recarga sin parámetros GET
                            }, 1000);

                            // Detiene el monitoreo
                            clearInterval(progressInterval);
                        }
                    },
                    error: function() {
                        // En caso de error en la solicitud
                        $('#status-text').text('Error al actualizar el progreso.');
                        $('#progress-bar').removeClass('bg-success')
                            .addClass('bg-danger');
                    }
                });
            }

            // Actualiza inmediatamente
            updateProgress();

            // Configura una actualización periódica
            var progressInterval = setInterval(updateProgress, 1000);
        }

        // Cuando se envía el formulario
        $('#update-form').on('submit', function() {
            startProgressMonitoring();
        });
    });
</script>
<br>

<!-- Filtros y búsqueda -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Filtros</h3>
    </div>
    <div class="card-body">
        <form action="index.php" method="GET" class="form-horizontal">
            <div class="row">
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Estado del Operador:</label>
                        <select name="status" class="form-control">
                            <option value="">Todos</option>
                            <option value="active" <?php echo $status === 'active' ? 'selected' : ''; ?>>Activos</option>
                            <option value="inactive" <?php echo $status === 'inactive' ? 'selected' : ''; ?>>Inactivos</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Estado de Licencia:</label>
                        <select name="license_status" class="form-control">
                            <option value="">Todos</option>
                            <option value="active" <?php echo $license_status === 'active' ? 'selected' : ''; ?>>Licencia Activa</option>
                            <option value="expired" <?php echo $license_status === 'expired' ? 'selected' : ''; ?>>Licencia Vencida</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="form-group">
                        <label>Mostrar:</label>
                        <select name="per_page" class="form-control">
                            <option value="10" <?php echo ($registros_por_pagina == 10) ? 'selected' : ''; ?>>10 por página</option>
                            <option value="20" <?php echo ($registros_por_pagina == 20) ? 'selected' : ''; ?>>20 por página</option>
                            <option value="50" <?php echo ($registros_por_pagina == 50) ? 'selected' : ''; ?>>50 por página</option>
                            <option value="100" <?php echo ($registros_por_pagina == 100) ? 'selected' : ''; ?>>100 por página</option>
                        </select>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="form-group">
                        <label>Buscar:</label>
                        <input type="text" name="search" class="form-control" placeholder="Nombre, ID, DNI o Posición" value="<?php echo htmlspecialchars($search); ?>">
                    </div>
                </div>
                <div class="col-md-1">
                    <div class="form-group">
                        <label>&nbsp;</label>
                        <div>
                            <button type="submit" class="btn btn-primary btn-block">
                                <i class="fas fa-search"></i> Filtrar
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </form>
    </div>
</div>

<!-- Listado de operadores -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Operadores Registrados</h3>
        <div class="card-tools">
            <a href="create.php" class="btn btn-primary btn-sm">
                <i class="fas fa-user-plus"></i> Nuevo Operador
            </a>
        </div>
    </div>
    <div class="card-body table-responsive p-0">
        <table class="table table-hover text-nowrap">
            <thead>
                <tr>
                    <th>Foto</th>
                    <th>ID</th>
                    <th>Nombre</th>
                    <th>DNI/CE</th>
                    <th>Posición</th>
                    <th>Estado</th>
                    <th>Licencia</th>
                    <th>Calibración</th>
                    <th>Máquina Asignada</th>
                    <th>Acciones</th>
                </tr>
            </thead>
            <tbody>
                <?php if (empty($operators)): ?>
                    <tr>
                        <td colspan="10" class="text-center">No se encontraron operadores</td>
                    </tr>
                <?php else: ?>
                    <?php foreach ($operators as $operator): ?>
                        <tr>
                            <td class="text-center">
                                <?php if (!empty($operator['photo_path'])): ?>
                                    <img src="<?php echo htmlspecialchars($operator['photo_path']); ?>"
                                        alt="Foto de <?php echo htmlspecialchars($operator['name']); ?>"
                                        class="img-circle elevation-2"
                                        width="40" height="40">
                                <?php else: ?>
                                    <div style="width: 40px; height: 40px; margin: 0 auto; background-color: #6c757d; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white;">
                                        <?php echo substr($operator['name'], 0, 1); ?>
                                    </div>
                                <?php endif; ?>
                            </td>
                            <td><?php echo htmlspecialchars($operator['id']); ?></td>
                            <td><?php echo htmlspecialchars($operator['name']); ?></td>
                            <td><?php echo htmlspecialchars($operator['dni_number'] ?? 'No registrado'); ?></td>
                            <td><?php echo htmlspecialchars($operator['position'] ?? 'No especificada'); ?></td>
                            <td>
                                <?php if ($operator['status'] == 'active'): ?>
                                    <span class="badge badge-success">Activo</span>
                                <?php else: ?>
                                    <span class="badge badge-danger">Inactivo</span>
                                <?php endif; ?>
                            </td>
                            <td>
                                <?php if (empty($operator['license_number'])): ?>
                                    <span class="badge badge-secondary">Sin licencia</span>
                                <?php elseif ($operator['license_status'] == 'active'): ?>
                                    <span class="badge badge-success">Activa</span>
                                    <?php if (!empty($operator['license_expiry'])): ?>
                                        <small class="d-block">Vence: <?php echo date('d/m/Y', strtotime($operator['license_expiry'])); ?></small>
                                    <?php endif; ?>
                                <?php else: ?>
                                    <span class="badge badge-danger">Vencida</span>
                                    <?php if (!empty($operator['license_expiry'])): ?>
                                        <small class="d-block">Venció: <?php echo date('d/m/Y', strtotime($operator['license_expiry'])); ?></small>
                                    <?php endif; ?>
                                <?php endif; ?>
                            </td>
                            <td>
                                <?php if ($operator['has_calibration']): ?>
                                    <span class="badge badge-success" title="Operador calibrado">
                                        <i class="fas fa-check-circle"></i> Calibrado
                                    </span>
                                <?php else: ?>
                                    <span class="badge badge-warning" title="Sin calibración">
                                        <i class="fas fa-exclamation-triangle"></i> Sin calibrar
                                    </span>
                                <?php endif; ?>
                            </td>
                            <td>
                                <?php if ($operator['assigned_machines'] > 0): ?>
                                    <span class="badge badge-info"><?php echo htmlspecialchars($operator['current_machine']); ?></span>
                                <?php else: ?>
                                    <span class="badge badge-secondary">Sin asignar</span>
                                <?php endif; ?>
                            </td>
                            <td>
                                <div class="btn-group">
                                    <a href="view.php?id=<?php echo $operator['id']; ?>" class="btn btn-info btn-sm">
                                        <i class="fas fa-eye"></i>
                                    </a>
                                    <a href="edit.php?id=<?php echo $operator['id']; ?>" class="btn btn-warning btn-sm">
                                        <i class="fas fa-edit"></i>
                                    </a>
                                    <?php if ($operator['assigned_machines'] == 0): ?>
                                        <a href="assign.php?id=<?php echo $operator['id']; ?>" class="btn btn-primary btn-sm">
                                            <i class="fas fa-link"></i>
                                        </a>
                                    <?php endif; ?>
                                    <button type="button" class="btn btn-danger btn-sm" data-toggle="modal" data-target="#deleteModal<?php echo $operator['id']; ?>">
                                        <i class="fas fa-trash"></i>
                                    </button>
                                </div>

                                <!-- Modal de confirmación para eliminar -->
                                <div class="modal fade" id="deleteModal<?php echo $operator['id']; ?>" tabindex="-1" role="dialog" aria-labelledby="deleteModalLabel" aria-hidden="true">
                                    <div class="modal-dialog" role="document">
                                        <div class="modal-content">
                                            <div class="modal-header">
                                                <h5 class="modal-title" id="deleteModalLabel">Confirmar Eliminación</h5>
                                                <button type="button" class="close" data-dismiss="modal" aria-label="Close">
                                                    <span aria-hidden="true">&times;</span>
                                                </button>
                                            </div>
                                            <div class="modal-body">
                                                ¿Está seguro que desea eliminar al operador <strong><?php echo htmlspecialchars($operator['name']); ?></strong>?
                                                <?php if ($operator['assigned_machines'] > 0): ?>
                                                    <div class="alert alert-warning mt-3">
                                                        <i class="fas fa-exclamation-triangle"></i> Este operador tiene máquinas asignadas. Al eliminarlo, se eliminarán también todas sus asignaciones.
                                                    </div>
                                                <?php endif; ?>
                                            </div>
                                            <div class="modal-footer">
                                                <button type="button" class="btn btn-secondary" data-dismiss="modal">Cancelar</button>
                                                <a href="delete.php?id=<?php echo $operator['id']; ?>" class="btn btn-danger">Eliminar</a>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                <?php endif; ?>
            </tbody>
        </table>
    </div>
    
    <!-- Paginación -->
    <?php if ($total_paginas > 1): ?>
    <div class="card-footer clearfix">
        <div class="float-left">
            <p class="mb-0">
                Mostrando <?php echo $offset + 1; ?> a 
                <?php echo min($offset + $registros_por_pagina, $total_registros); ?> 
                de <?php echo $total_registros; ?> operadores
            </p>
        </div>
        <ul class="pagination pagination-sm m-0 float-right">
            <!-- Botón Primera página -->
            <?php if ($pagina_actual > 1): ?>
                <li class="page-item">
                    <a class="page-link" href="?page=1&status=<?php echo $status; ?>&license_status=<?php echo $license_status; ?>&search=<?php echo urlencode($search); ?>&per_page=<?php echo $registros_por_pagina; ?>">
                        <i class="fas fa-angle-double-left"></i>
                    </a>
                </li>
            <?php endif; ?>
            
            <!-- Botón Anterior -->
            <?php if ($pagina_actual > 1): ?>
                <li class="page-item">
                    <a class="page-link" href="?page=<?php echo $pagina_actual - 1; ?>&status=<?php echo $status; ?>&license_status=<?php echo $license_status; ?>&search=<?php echo urlencode($search); ?>&per_page=<?php echo $registros_por_pagina; ?>">
                        <i class="fas fa-chevron-left"></i>
                    </a>
                </li>
            <?php endif; ?>
            
            <!-- Números de página -->
            <?php
            $rango = 2; // Mostrar 2 páginas antes y después de la actual
            $desde = max(1, $pagina_actual - $rango);
            $hasta = min($total_paginas, $pagina_actual + $rango);
            
            if ($desde > 1) {
                echo '<li class="page-item disabled"><span class="page-link">...</span></li>';
            }
            
            for ($i = $desde; $i <= $hasta; $i++):
            ?>
                <li class="page-item <?php echo ($i == $pagina_actual) ? 'active' : ''; ?>">
                    <a class="page-link" href="?page=<?php echo $i; ?>&status=<?php echo $status; ?>&license_status=<?php echo $license_status; ?>&search=<?php echo urlencode($search); ?>&per_page=<?php echo $registros_por_pagina; ?>">
                        <?php echo $i; ?>
                    </a>
                </li>
            <?php 
            endfor;
            
            if ($hasta < $total_paginas) {
                echo '<li class="page-item disabled"><span class="page-link">...</span></li>';
            }
            ?>
            
            <!-- Botón Siguiente -->
            <?php if ($pagina_actual < $total_paginas): ?>
                <li class="page-item">
                    <a class="page-link" href="?page=<?php echo $pagina_actual + 1; ?>&status=<?php echo $status; ?>&license_status=<?php echo $license_status; ?>&search=<?php echo urlencode($search); ?>&per_page=<?php echo $registros_por_pagina; ?>">
                        <i class="fas fa-chevron-right"></i>
                    </a>
                </li>
            <?php endif; ?>
            
            <!-- Botón Última página -->
            <?php if ($pagina_actual < $total_paginas): ?>
                <li class="page-item">
                    <a class="page-link" href="?page=<?php echo $total_paginas; ?>&status=<?php echo $status; ?>&license_status=<?php echo $license_status; ?>&search=<?php echo urlencode($search); ?>&per_page=<?php echo $registros_por_pagina; ?>">
                        <i class="fas fa-angle-double-right"></i>
                    </a>
                </li>
            <?php endif; ?>
        </ul>
    </div>
    <?php endif; ?>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="create.php" class="btn btn-primary">
    <i class="fas fa-user-plus"></i> Nuevo Operador
</a>
';

// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>