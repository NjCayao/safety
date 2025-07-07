<?php
// Incluir configuración
require_once 'config/config.php';

// Iniciar sesión si no está iniciada
if (session_status() == PHP_SESSION_NONE) {
    session_start();
}

// Incluir header
$pageTitle = 'Acceso No Autorizado';
include 'includes/header.php';
?>

<!-- Content Wrapper -->
<div class="content-wrapper">
    <!-- Content Header -->
    <section class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1>401 Acceso No Autorizado</h1>
                </div>
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="<?php echo BASE_URL; ?>/index.php">Inicio</a></li>
                        <li class="breadcrumb-item active">Error 401</li>
                    </ol>
                </div>
            </div>
        </div>
    </section>

    <!-- Main content -->
    <section class="content">
        <div class="error-page">
            <h2 class="headline text-warning">401</h2>
            <div class="error-content">
                <h3><i class="fas fa-exclamation-triangle text-warning"></i> Acceso denegado.</h3>
                <p>
                    No tiene permisos suficientes para acceder a esta página.
                    Si considera que esto es un error, por favor contacte al administrador del sistema.
                </p>
                <p>
                    <a href="<?php echo BASE_URL; ?>/index.php" class="btn btn-primary">Volver al Inicio</a>
                </p>
            </div>
        </div>
    </section>
</div>

<?php include 'includes/footer.php'; ?>