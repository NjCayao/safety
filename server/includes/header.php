<?php
// Iniciar sesión si no está iniciada
if (!defined('BASE_URL')) {
  session_start();
  
  // Determinar la ruta base desde cualquier ubicación
  $currentPath = dirname($_SERVER['SCRIPT_FILENAME']);
  $rootPath = $currentPath;
  
  // Buscar la carpeta 'server' subiendo niveles
  while (!file_exists($rootPath . '/config/config.php') && $rootPath !== '/') {
    $rootPath = dirname($rootPath);
  }
  
  require_once $rootPath . '/config/config.php';
}


// Verificar si hay usuario autenticado
$loggedIn = isset($_SESSION['user_id']);
?>
<!DOCTYPE html>
<html lang="es">

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title><?php echo SITE_TITLE; ?> | <?php echo $pageTitle ?? 'Dashboard'; ?></title>

  <!-- Google Font: Source Sans Pro -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Source+Sans+Pro:300,400,400i,700&display=fallback">
  <!-- Font Awesome -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/fontawesome-free/css/all.min.css">
  <!-- Ionicons -->
  <link rel="stylesheet" href="https://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css">
  <!-- Tempusdominus Bootstrap 4 -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/tempusdominus-bootstrap-4/css/tempusdominus-bootstrap-4.min.css">
  <!-- iCheck -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/icheck-bootstrap/icheck-bootstrap.min.css">
  <!-- JQVMap -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/jqvmap/jqvmap.min.css">
  <!-- Theme style -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/dist/css/adminlte.min.css">
  <!-- overlayScrollbars -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/overlayScrollbars/css/OverlayScrollbars.min.css">
  <!-- Daterange picker -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/daterangepicker/daterangepicker.css">
  <!-- summernote -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/summernote/summernote-bs4.min.css">

  <!-- Estilos adicionales específicos de la página -->
  <?php if (isset($extraCss)) echo $extraCss; ?>
</head>
<!-- 
<style> 
  .main-header {
    margin-top: -250px;
}

</style> -->

<body class="hold-transition sidebar-mini layout-fixed">
  <div class="wrapper">

    <!-- Preloader -->
    <div class="preloader flex-column justify-content-center align-items-center">
      <img class="animation__shake" src="<?php echo ASSETS_URL; ?>/dist/img/AdminLTELogo.png" alt="Logo" height="60" width="60">
    </div>

    <!-- Navbar -->
    <nav class="main-header navbar navbar-expand navbar-white navbar-light">
      <!-- Left navbar links -->
      <ul class="navbar-nav">
        <li class="nav-item">
          <a class="nav-link" data-widget="pushmenu" href="#" role="button"><i class="fas fa-bars"></i></a>
        </li>
        <li class="nav-item d-none d-sm-inline-block">
          <a href="index.php" class="nav-link">Inicio</a>
        </li>
      </ul>

      <!-- Right navbar links -->
      <ul class="navbar-nav ml-auto">
        <!-- Notifications Dropdown Menu -->
         
        <li class="nav-item dropdown">
          <a class="nav-link" data-toggle="dropdown" href="#">
            <i class="far fa-bell"></i>
            <span class="badge badge-warning navbar-badge" id="alerts-counter">0</span>
          </a>
          <div class="dropdown-menu dropdown-menu-lg dropdown-menu-right">
            <span class="dropdown-item dropdown-header">Alertas Recientes</span>
            <div class="dropdown-divider"></div>
            <div id="realtime-alerts-container">
              <div class="text-center p-3">Cargando alertas...</div>
            </div>
            <div class="dropdown-divider"></div>
            <a href="pages/alerts/index.php" class="dropdown-item dropdown-footer">Ver Todas las Alertas</a>
          </div>
        </li>


        <li class="nav-item dropdown">
          <a class="nav-link" data-toggle="dropdown" href="#">
            <i class="far fa-bell"></i>
            <span id="notification-counter" class="badge badge-warning navbar-badge">0</span>
          </a>
          <div class="dropdown-menu dropdown-menu-lg dropdown-menu-right">
            <span id="notification-header" class="dropdown-item dropdown-header">0 Notificaciones</span>
            <div class="dropdown-divider"></div>
            <div id="notification-container"></div>
            <div class="dropdown-divider"></div>
            <a href="pages/alerts/index.php" class="dropdown-item dropdown-footer">Ver todas</a>
          </div>
        </li>
        <li class="nav-item">
          <a class="nav-link" data-widget="fullscreen" href="#" role="button">
            <i class="fas fa-expand-arrows-alt"></i>
          </a>
        </li>
        <?php if ($loggedIn): ?>
          <li class="nav-item dropdown">
            <a class="nav-link" data-toggle="dropdown" href="#">
              <i class="far fa-user"></i> <?php echo $_SESSION['username'] ?? 'Usuario'; ?>
            </a>
            <div class="dropdown-menu dropdown-menu-lg dropdown-menu-right">
              <a href="pages/profile.php" class="dropdown-item">
                <i class="fas fa-user mr-2"></i> Perfil
              </a>
              <div class="dropdown-divider"></div>
              <a href="logout.php" class="dropdown-item">
                <i class="fas fa-sign-out-alt mr-2"></i> Cerrar Sesión
              </a>
            </div>
          </li>
        <?php endif; ?>
      </ul>
    </nav>
    <!-- /.navbar -->