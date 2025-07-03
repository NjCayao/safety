<?php
/**
 * Verifica si un menú debe mostrarse abierto basado en la URL actual
 * @param array $paths Rutas a verificar
 * @return string 'menu-open' si alguna ruta coincide, '' en caso contrario
 */
function isMenuOpen($paths) {
    $currentDir = basename(dirname($_SERVER['PHP_SELF']));
    foreach ($paths as $path) {
        if (strpos($currentDir, $path) !== false) {
            return 'menu-open';
        }
    }
    return '';
}

/**
 * Verifica si un enlace debe mostrarse activo
 * @param string $path Ruta a verificar
 * @return string 'active' si la ruta coincide, '' en caso contrario
 */
function isActive($path) {
    $currentPage = basename($_SERVER['PHP_SELF']);
    $currentDir = basename(dirname($_SERVER['PHP_SELF']));
    
    if (strpos($path, '/') !== false) {
        // Si la ruta incluye un directorio (ej: 'alerts/index.php')
        list($dir, $page) = explode('/', $path);
        return ($currentDir == $dir && $currentPage == $page) ? 'active' : '';
    } else {
        // Si es solo una verificación de directorio (ej: 'alerts/')
        return (strpos($currentDir, $path) !== false) ? 'active' : '';
    }
}

/**
 * Verifica si un usuario tiene permiso para ver un menú específico
 * @param string $moduleUrl URL del módulo a verificar
 * @return bool true si tiene permiso, false en caso contrario
 */
function canShowMenu($moduleUrl) {
    // Si no hay usuario autenticado, no mostrar
    if (!isset($_SESSION['user_id'])) {
        return false;
    }
    
    // Admin ve todo
    if ($_SESSION['user_role'] == 'admin') {
        return true;
    }
    
    // Verificar permisos específicos
    require_once __DIR__ . '/../models/Permission.php';
    
    static $permissionModel = null;
    if ($permissionModel === null) {
        $permissionModel = new Permission();
    }
    
    return $permissionModel->hasAccess($_SESSION['user_id'], $moduleUrl, 'view');
}