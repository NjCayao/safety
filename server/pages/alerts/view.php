<?php
// Iniciar sesión
session_start();

// Verificar si el usuario está autenticado
if (!isset($_SESSION['user_id'])) {
    header('Location: ../../login.php');
    exit;
}

// Verificar si se proporcionó un ID de alerta
if (!isset($_GET['id']) || empty($_GET['id'])) {
    $_SESSION['error_message'] = 'ID de alerta no especificado.';
    header('Location: index.php');
    exit;
}

// Asignar rol de administrador al usuario "admin" si no está definido
if (!isset($_SESSION['user_role']) && $_SESSION['username'] == 'admin') {
    $_SESSION['user_role'] = 'admin';
}

// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../models/Alert.php';

// Crear instancia del modelo Alert y obtener datos
try {
    // Obtener ID de la alerta
    $alertId = (int)$_GET['id'];

    // Crear instancia del modelo
    $alertModel = new Alert();

    // Obtener detalles de la alerta
    $alert = $alertModel->getAlertById($alertId);

    // Si la alerta no existe, redirigir al listado
    if (!$alert) {
        $_SESSION['error_message'] = 'La alerta #' . $alertId . ' no existe o no se pudo recuperar.';
        header('Location: index.php');
        exit;
    }

    // Obtener las imágenes asociadas a la alerta
    $images = $alertModel->getAlertImages($alertId);

    // Obtener IDs de alertas anterior y siguiente para navegación
    $prevAlertId = $alertModel->getPreviousAlertId($alertId);
    $nextAlertId = $alertModel->getNextAlertId($alertId);
} catch (Exception $e) {
    $_SESSION['error_message'] = 'Error al recuperar datos de la alerta: ' . $e->getMessage();
    header('Location: index.php');
    exit;
}

// Procesar la acción de reconocer alerta
if (isset($_POST['acknowledge']) && !$alert['acknowledged'] && isset($_SESSION['user_id'])) {
    $actionsTaken = $_POST['actions_taken'] ?? '';
    $comments = $_POST['comments'] ?? '';

    try {
        if (method_exists($alertModel, 'markAsAcknowledgedWithDetails')) {
            $result = $alertModel->markAsAcknowledgedWithDetails($alertId, $_SESSION['username'], $actionsTaken, $comments);
        } else {
            $result = $alertModel->markAsAcknowledged($alertId, $_SESSION['username']);
        }

        if ($result) {
            $_SESSION['success_message'] = 'Alerta marcada como atendida exitosamente.';
            header("Location: view.php?id=$alertId");
            exit;
        } else {
            $_SESSION['error_message'] = 'Error al intentar marcar la alerta como atendida.';
        }
    } catch (Exception $e) {
        $_SESSION['error_message'] = 'Error al procesar la acción: ' . $e->getMessage();
    }
}

// Definir etiquetas para los tipos de alerta
$alertTypeLabels = [
    'fatigue' => 'Fatiga',
    'yawn' => 'Bostezo',
    'phone' => 'Uso de Teléfono',
    'smoking' => 'Fumando',
    'distraction' => 'Distracción',
    'unauthorized' => 'Operador No Autorizado',
    'behavior' => 'Comportamiento Inadecuado',
    'other' => 'Otro'
];

// Colores para los tipos de alerta
$alertTypeClasses = [
    'fatigue' => 'danger',
    'yawn' => 'warning',
    'phone' => 'info',
    'smoking' => 'secondary',
    'distraction' => 'primary',
    'unauthorized' => 'dark',
    'behavior' => 'info',
    'other' => 'light'
];

// Incluir header
$pageTitle = "Detalles de Alerta";
include_once '../../includes/header.php';
include_once '../../includes/sidebar.php';
?>

<!-- Content Wrapper. Contains page content -->
<div class="content-wrapper">
    <!-- Content Header (Page header) -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>Detalles de Alerta</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="../../index.php">Inicio</a></li>
                        <li class="breadcrumb-item"><a href="index.php">Alertas</a></li>
                        <li class="breadcrumb-item active">Detalles</li>
                    </ol>
                </div>
            </div>
        </div><!-- /.container-fluid -->
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h3 class="card-title">
                                Alerta #<?php echo $alert['id']; ?> -
                                <span class="badge badge-<?php echo $alertTypeClasses[$alert['alert_type']] ?? 'secondary'; ?>">
                                    <?php echo $alertTypeLabels[$alert['alert_type']] ?? $alert['alert_type']; ?>
                                </span>
                            </h3>
                            <div class="card-tools">
                                <?php if ($prevAlertId): ?>
                                    <a href="view.php?id=<?php echo $prevAlertId; ?>" class="btn btn-sm btn-default mr-2">
                                        <i class="fas fa-chevron-left"></i> Anterior
                                    </a>
                                <?php endif; ?>

                                <?php if ($nextAlertId): ?>
                                    <a href="view.php?id=<?php echo $nextAlertId; ?>" class="btn btn-sm btn-default">
                                        Siguiente <i class="fas fa-chevron-right"></i>
                                    </a>
                                <?php endif; ?>
                            </div>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="info-box">
                                        <span class="info-box-icon bg-info">
                                            <i class="fas fa-user"></i>
                                        </span>
                                        <div class="info-box-content">
                                            <span class="info-box-text">Operador</span>
                                            <span class="info-box-number">
                                                <?php if ($alert['operator_id']): ?>
                                                    <?php echo $alert['operator_name']; ?> (<?php echo $alert['operator_id']; ?>)
                                                    <?php if (!empty($alert['operator_dni'])): ?>
                                                        <br><small>DNI: <?php echo $alert['operator_dni']; ?></small>
                                                    <?php endif; ?>
                                                <?php else: ?>
                                                    No especificado
                                                <?php endif; ?>
                                            </span>
                                        </div>
                                    </div>

                                    <div class="info-box">
                                        <span class="info-box-icon bg-warning">
                                            <i class="fas fa-truck"></i>
                                        </span>
                                        <div class="info-box-content">
                                            <span class="info-box-text">Máquina</span>
                                            <span class="info-box-number">
                                                <?php if ($alert['machine_id']): ?>
                                                    <?php echo $alert['machine_name']; ?> (<?php echo $alert['machine_id']; ?>)
                                                <?php else: ?>
                                                    No especificada
                                                <?php endif; ?>
                                            </span>
                                        </div>
                                    </div>

                                    <div class="info-box">
                                        <span class="info-box-icon bg-success">
                                            <i class="fas fa-clock"></i>
                                        </span>
                                        <div class="info-box-content">
                                            <span class="info-box-text">Fecha y Hora</span>
                                            <span class="info-box-number">
                                                <?php echo date('d/m/Y H:i:s', strtotime($alert['timestamp'])); ?>
                                            </span>
                                        </div>
                                    </div>

                                    <div class="info-box">
                                        <span class="info-box-icon bg-<?php echo $alert['acknowledged'] ? 'success' : 'danger'; ?>">
                                            <i class="fas fa-<?php echo $alert['acknowledged'] ? 'check' : 'exclamation'; ?>"></i>
                                        </span>
                                        <div class="info-box-content">
                                            <span class="info-box-text">Estado</span>
                                            <span class="info-box-number">
                                                <?php if ($alert['acknowledged']): ?>
                                                    Atendida
                                                    <?php if ($alert['acknowledged_by']): ?>
                                                        por <?php echo $alert['acknowledged_by']; ?>
                                                    <?php endif; ?>
                                                    <?php if ($alert['acknowledgement_time']): ?>
                                                        <br><small>
                                                            <?php echo date('d/m/Y H:i:s', strtotime($alert['acknowledgement_time'])); ?>
                                                        </small>
                                                    <?php endif; ?>
                                                <?php else: ?>
                                                    Pendiente
                                                <?php endif; ?>
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                <div class="col-md-6">
                                    <div class="card">
                                        <div class="card-header">
                                            <h3 class="card-title">Detalles</h3>
                                        </div>
                                        <div class="card-body">
                                            <?php if (!empty($alert['details'])): ?>
                                                <p><?php echo nl2br(htmlspecialchars($alert['details'])); ?></p>
                                            <?php else: ?>
                                                <p class="text-muted">No hay detalles adicionales para esta alerta.</p>
                                            <?php endif; ?>
                                        </div>
                                    </div>

                                    <?php if ($alert['acknowledged'] && (!empty($alert['actions_taken']) || !empty($alert['ack_comments']))): ?>
                                        <div class="card">
                                            <div class="card-header">
                                                <h3 class="card-title">Detalles de Atención</h3>
                                            </div>
                                            <div class="card-body">
                                                <?php if (!empty($alert['actions_taken'])): ?>
                                                    <div class="form-group">
                                                        <label>Acciones Tomadas:</label>
                                                        <p><?php echo nl2br(htmlspecialchars($alert['actions_taken'])); ?></p>
                                                    </div>
                                                <?php endif; ?>

                                                <?php if (!empty($alert['ack_comments'])): ?>
                                                    <div class="form-group">
                                                        <label>Comentarios:</label>
                                                        <p><?php echo nl2br(htmlspecialchars($alert['ack_comments'])); ?></p>
                                                    </div>
                                                <?php endif; ?>
                                            </div>
                                        </div>
                                    <?php endif; ?>



                                    <!-- Imágenes sin procesar relacionadas con este operador -->
                                    <div class="card">
                                        <div class="card-header bg-warning">
                                            <h3 class="card-title">Imágenes en Procesamiento</h3>
                                            <div class="card-tools">
                                                <button type="button" id="refresh-unprocessed" class="btn btn-tool" data-toggle="tooltip" title="Actualizar">
                                                    <i class="fas fa-sync"></i>
                                                </button>
                                                <button type="button" class="btn btn-tool" data-card-widget="collapse">
                                                    <i class="fas fa-minus"></i>
                                                </button>
                                            </div>
                                        </div>
                                        <div class="card-body">
                                            <div id="unprocessed-loading" class="text-center" style="display: none;">
                                                <div class="spinner-border text-warning" role="status">
                                                    <span class="sr-only">Cargando...</span>
                                                </div>
                                                <p>Buscando imágenes...</p>
                                            </div>
                                            <div id="unprocessed-images-container">
                                                <div class="text-center p-3"><i class="fas fa-spinner fa-spin"></i> Buscando imágenes...</div>
                                            </div>
                                        </div>
                                    </div>


                                    <?php if (!empty($images)): ?>
                                        <div class="card">
                                            <div class="card-header">
                                                <h3 class="card-title">Imágenes</h3>
                                            </div>
                                            <div class="card-body">
                                                <div class="row">
                                                    <?php foreach ($images as $image): ?>
                                                        <div class="col-md-12 text-center">
                                                            <?php if (file_exists('../../' . $image['image_path'])): ?>
                                                                <a href="../../<?php echo $image['image_path']; ?>" data-toggle="lightbox">
                                                                    <img src="../../<?php echo $image['image_path']; ?>"
                                                                        class="img-fluid mb-2"
                                                                        alt="Imagen de Alerta">
                                                                </a>
                                                                <p class="text-muted">
                                                                    Capturada el <?php echo date('d/m/Y H:i:s', strtotime($image['created_at'])); ?>
                                                                </p>
                                                            <?php else: ?>
                                                                <div class="alert alert-warning">
                                                                    La imagen no está disponible. Ruta registrada: <?php echo $image['image_path']; ?>
                                                                </div>
                                                            <?php endif; ?>
                                                        </div>
                                                    <?php endforeach; ?>
                                                </div>
                                            </div>
                                        </div>
                                    <?php endif; ?>

                                    <?php if (!$alert['acknowledged'] && ($_SESSION['username'] == 'admin' || (isset($_SESSION['user_role']) && ($_SESSION['user_role'] == 'admin' || $_SESSION['user_role'] == 'supervisor')))): ?>
                                        <div class="card">
                                            <div class="card-header">
                                                <h3 class="card-title">Atender Alerta</h3>
                                            </div>
                                            <div class="card-body">
                                                <form method="POST" action="">
                                                    <div class="form-group">
                                                        <label for="actions_taken">Acciones Tomadas:</label>
                                                        <textarea class="form-control" id="actions_taken" name="actions_taken" rows="3" placeholder="Describa las acciones tomadas para atender esta alerta..."></textarea>
                                                    </div>

                                                    <div class="form-group">
                                                        <label for="comments">Comentarios Adicionales:</label>
                                                        <textarea class="form-control" id="comments" name="comments" rows="2" placeholder="Comentarios adicionales..."></textarea>
                                                    </div>

                                                    <button type="submit" name="acknowledge" class="btn btn-success btn-block">
                                                        <i class="fas fa-check-circle"></i> Marcar como Atendida
                                                    </button>
                                                </form>
                                            </div>
                                        </div>
                                    <?php endif; ?>
                                </div>
                            </div>
                        </div>
                        <div class="card-footer">
                            <a href="index.php" class="btn btn-default">
                                <i class="fas fa-arrow-left"></i> Volver al Listado
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
</div>

<!-- Footer -->
<?php include_once '../../includes/footer.php'; ?>

<!-- Ekko Lightbox -->
<script src="<?php echo ASSETS_URL; ?>/plugins/ekko-lightbox/ekko-lightbox.min.js"></script>
<script>
    $(document).on('click', '[data-toggle="lightbox"]', function(event) {
        event.preventDefault();
        $(this).ekkoLightbox();
    });
</script>

</body>

</html>