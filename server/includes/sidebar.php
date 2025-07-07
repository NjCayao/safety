<?php
// Si no viene de header.php, incluir sesión y configuración
if (!defined('BASE_URL')) {
  session_start();
  require_once 'config/config.php';
}

// Incluir funciones auxiliares
require_once __DIR__ . '/functions_helpers.php';

// Obtener la página actual para resaltar el menú correspondiente
$currentPage = basename($_SERVER['PHP_SELF']);
$currentDir = basename(dirname($_SERVER['PHP_SELF']));

// Contar alertas pendientes (si la función está disponible)
$pendingAlerts = 0;
if (function_exists('db_fetch_one')) {
    $result = db_fetch_one("SELECT COUNT(*) as count FROM alerts WHERE acknowledged = 0");
    if ($result) {
        $pendingAlerts = $result['count'];
    }
}

// 🆕 NUEVO: Contar dispositivos offline y configuraciones pendientes
$offlineDevices = 0;
$pendingConfigs = 0;
if (function_exists('db_fetch_one')) {
    try {
        $result = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE status = 'offline'");
        if ($result) {
            $offlineDevices = $result['count'];
        }
        
        $result = db_fetch_one("SELECT COUNT(*) as count FROM devices WHERE config_pending = 1");
        if ($result) {
            $pendingConfigs = $result['count'];
        }
    } catch (Exception $e) {
        // Si no existe la tabla, valores por defecto
        $offlineDevices = 0;
        $pendingConfigs = 0;
    }
}
?>
<!-- Main Sidebar Container -->
<aside class="main-sidebar sidebar-dark-primary elevation-4">
  <!-- Brand Logo -->
  <a href="<?php echo BASE_URL; ?>/index.php" class="brand-link">
    <img src="<?php echo ASSETS_URL; ?>/dist/img/AdminLTELogo.png" alt="Logo" class="brand-image img-circle elevation-3" style="opacity: .8">
    <span class="brand-text font-weight-light">Safety System</span>
  </a>

  <!-- Sidebar -->
  <div class="sidebar">
    <!-- Sidebar user panel (optional) -->
    <?php if (isset($_SESSION['user_id'])): ?>
      <div class="user-panel mt-3 pb-3 mb-3 d-flex">
        <div class="image">
          <img src="<?php echo ASSETS_URL; ?>/dist/img/user2-160x160.jpg" class="img-circle elevation-2" alt="User Image">
        </div>
        <div class="info">
          <a href="#" class="d-block"><?php echo $_SESSION['name'] ?? 'Usuario'; ?></a>
          <span class="badge badge-info"><?php echo ucfirst($_SESSION['role'] ?? 'usuario'); ?></span>
        </div>
      </div>
    <?php endif; ?>

    <!-- Sidebar Menu -->
    <nav class="mt-2">
      <ul class="nav nav-pills nav-sidebar flex-column" data-widget="treeview" role="menu" data-accordion="false">
        <li class="nav-item">
          <a href="<?php echo BASE_URL; ?>/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'server') ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-tachometer-alt"></i>
            <p>Dashboard</p>
          </a>
        </li>

        <li class="nav-item <?php echo (strpos($currentDir, 'operators') !== false) ? 'menu-open' : ''; ?>">
          <a href="#" class="nav-link <?php echo (strpos($currentDir, 'operators') !== false) ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-users"></i>
            <p>
              Operadores
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>
          <ul class="nav nav-treeview">
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/operators/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'operators') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Lista de Operadores</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/operators/create.php" class="nav-link <?php echo ($currentPage == 'create.php' && $currentDir == 'operators') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Nuevo Operador</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/operators/assign.php" class="nav-link <?php echo ($currentPage == 'assign.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Asignar a Máquina</p>
              </a>
            </li>
          </ul>
        </li>

        <!-- 🆕 DISPOSITIVOS ACTUALIZADO - MÓDULO COMPLETO -->
        <li class="nav-item <?php echo (strpos($currentDir, 'devices') !== false) ? 'menu-open' : ''; ?>">
          <a href="#" class="nav-link <?php echo (strpos($currentDir, 'devices') !== false) ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-microchip"></i>
            <p>
              Dispositivos
              <?php if ($offlineDevices > 0): ?>
                <span class="badge badge-danger right"><?php echo $offlineDevices; ?></span>
              <?php elseif ($pendingConfigs > 0): ?>
                <span class="badge badge-warning right"><?php echo $pendingConfigs; ?></span>
              <?php endif; ?>
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>
          <ul class="nav nav-treeview">
            <!-- Lista de Dispositivos -->
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>Lista de Dispositivos</p>
              </a>
            </li>
            
            <!-- Nuevo Dispositivo -->
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/create.php" class="nav-link <?php echo ($currentPage == 'create.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>Nuevo Dispositivo</p>
              </a>
            </li>
            
            <!-- 🆕 NUEVO: Gestión de Configuraciones -->
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/config.php" class="nav-link <?php echo ($currentPage == 'config.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>
                  Gestión de Configuraciones
                  <?php if ($pendingConfigs > 0): ?>
                    <span class="badge badge-warning right"><?php echo $pendingConfigs; ?></span>
                  <?php endif; ?>
                </p>
              </a>
            </li>
            
            <!-- 🆕 NUEVO: Estado en Tiempo Real -->
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/status.php" class="nav-link <?php echo ($currentPage == 'status.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>
                  Estado en Tiempo Real
                  <?php if ($offlineDevices > 0): ?>
                    <span class="badge badge-danger right"><?php echo $offlineDevices; ?></span>
                  <?php endif; ?>
                </p>
              </a>
            </li>
            
            <!-- 🆕 NUEVO: Historial de Configuraciones -->
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/history.php" class="nav-link <?php echo ($currentPage == 'history.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>Historial de Configuraciones</p>
              </a>
            </li>
            
            <!-- 🆕 NUEVO: Perfiles de Configuración (Solo Admin/Supervisor) -->
            <?php if (isset($_SESSION['user_id']) && ($_SESSION['role'] == 'admin' || $_SESSION['role'] == 'supervisor')): ?>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/devices/profiles.php" class="nav-link <?php echo ($currentPage == 'profiles.php' && $currentDir == 'devices') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon fa-xs"></i>
                <p>Perfiles de Configuración</p>
              </a>
            </li>
            <?php endif; ?>
          </ul>
        </li>

        <li class="nav-item <?php echo (strpos($currentDir, 'machines') !== false) ? 'menu-open' : ''; ?>">
          <a href="#" class="nav-link <?php echo (strpos($currentDir, 'machines') !== false) ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-truck"></i>
            <p>
              Máquinas
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>
          <ul class="nav nav-treeview">
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/machines/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'machines') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Lista de Máquinas</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/machines/create.php" class="nav-link <?php echo ($currentPage == 'create.php' && $currentDir == 'machines') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Nueva Máquina</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/machines/maintenance.php" class="nav-link <?php echo ($currentPage == 'maintenance.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Mantenimiento</p>
              </a>
            </li>
          </ul>
        </li>

        <!-- Alertas -->
        <li class="nav-item <?php echo isMenuOpen(['alerts']); ?>">
          <a href="#" class="nav-link <?php echo isActive('alerts'); ?>">
            <i class="nav-icon fas fa-bell"></i>
            <p>
              Alertas
              <?php if ($pendingAlerts > 0): ?>
                <span class="badge badge-warning right"><?php echo $pendingAlerts; ?></span>
              <?php endif; ?>
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>
          <ul class="nav nav-treeview">
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/alerts/index.php" class="nav-link <?php echo isActive('alerts/index.php'); ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Listar Alertas</p>
              </a>
            </li>
          </ul>
        </li>

        <!-- Estadísticas -->
        <li class="nav-item <?php echo isMenuOpen(['dashboard']); ?>">
          <a href="#" class="nav-link <?php echo isActive('dashboard'); ?>">
            <i class="nav-icon fas fa-chart-bar"></i>
            <p>
              Estadísticas
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>      

          <ul class="nav nav-treeview">
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/dashboard/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'dashboard') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Generales</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/dashboard/stats.php" class="nav-link <?php echo ($currentPage == 'stats.php' && $currentDir == 'dashboard') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Detalladas</p>
              </a>
            </li>            
          </ul>
        </li>

        <li class="nav-item <?php echo (strpos($currentDir, 'reports') !== false) ? 'menu-open' : ''; ?>">
          <a href="#" class="nav-link <?php echo (strpos($currentDir, 'reports') !== false) ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-chart-bar"></i>
            <p>
              Reportes
              <i class="right fas fa-angle-left"></i>
            </p>
          </a>
          <ul class="nav nav-treeview">
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/reports/daily.php" class="nav-link <?php echo ($currentPage == 'daily.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Reporte Diario</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/reports/weekly.php" class="nav-link <?php echo ($currentPage == 'weekly.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Reporte Semanal</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/reports/monthly.php" class="nav-link <?php echo ($currentPage == 'monthly.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Reporte Mensual</p>
              </a>
            </li>
            <li class="nav-item">
              <a href="<?php echo BASE_URL; ?>/pages/reports/custom.php" class="nav-link <?php echo ($currentPage == 'custom.php') ? 'active' : ''; ?>">
                <i class="far fa-circle nav-icon"></i>
                <p>Reporte Personalizado</p>
              </a>
            </li>
          </ul>
        </li>

        <?php if (isset($_SESSION['user_id']) && ($_SESSION['role'] == 'admin' || $_SESSION['role'] == 'supervisor')): ?>
  <li class="nav-item <?php echo (strpos($currentDir, 'users') !== false) ? 'menu-open' : ''; ?>">
    <a href="#" class="nav-link <?php echo (strpos($currentDir, 'users') !== false) ? 'active' : ''; ?>">
      <i class="nav-icon fas fa-user-shield"></i>
      <p>
        Usuarios
        <i class="right fas fa-angle-left"></i>
      </p>
    </a>
    <ul class="nav nav-treeview">
      <li class="nav-item">
        <a href="<?php echo BASE_URL; ?>/pages/users/index.php" class="nav-link <?php echo ($currentPage == 'index.php' && $currentDir == 'users') ? 'active' : ''; ?>">
          <i class="far fa-circle nav-icon"></i>
          <p>Lista de Usuarios</p>
        </a>
      </li>
      <?php if (isset($_SESSION['user_id']) && $_SESSION['role'] == 'admin'): ?>
        <li class="nav-item">
          <a href="<?php echo BASE_URL; ?>/pages/users/create.php" class="nav-link <?php echo ($currentPage == 'create.php' && $currentDir == 'users') ? 'active' : ''; ?>">
            <i class="far fa-circle nav-icon"></i>
            <p>Nuevo Usuario</p>
          </a>
        </li>
      <?php endif; ?>
      <!-- Eliminamos el enlace directo a permissions.php -->
    </ul>
  </li>
<?php endif; ?>

        <li class="nav-item">
          <a href="<?php echo BASE_URL; ?>/pages/system_logs.php" class="nav-link <?php echo ($currentPage == 'system_logs.php') ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-history"></i>
            <p>Logs del Sistema</p>
          </a>
        </li>

        <li class="nav-item">
          <a href="<?php echo BASE_URL; ?>/pages/settings.php" class="nav-link <?php echo ($currentPage == 'settings.php' && $currentDir == 'server') ? 'active' : ''; ?>">
            <i class="nav-icon fas fa-cog"></i>
            <p>Configuración</p>
          </a>
        </li>

        <?php if (isset($_SESSION['user_id'])): ?>
          <li class="nav-item">
            <a href="<?php echo BASE_URL; ?>/logout.php" class="nav-link">
              <i class="nav-icon fas fa-sign-out-alt"></i>
              <p>Cerrar Sesión</p>
            </a>
          </li>
        <?php else: ?>
          <li class="nav-item">
            <a href="<?php echo BASE_URL; ?>/login.php" class="nav-link <?php echo ($currentPage == 'login.php') ? 'active' : ''; ?>">
              <i class="nav-icon fas fa-sign-in-alt"></i>
              <p>Iniciar Sesión</p>
            </a>
          </li>
        <?php endif; ?>
      </ul>
    </nav>
    <!-- /.sidebar-menu -->
  </div>
  <!-- /.sidebar -->
</aside>