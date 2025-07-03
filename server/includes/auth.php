<?php
// includes/auth.php

class Auth {
    /**
     * Verifica si hay un usuario autenticado
     */
    public static function isLoggedIn() {
        return isset($_SESSION['user_id']);
    }
    
    /**
     * Verifica si el usuario actual tiene un rol específico
     */
    public static function hasRole($role) {
        if (!self::isLoggedIn()) {
            return false;
        }
        
        if (is_array($role)) {
            return in_array($_SESSION['user_role'], $role);
        }
        
        return $_SESSION['user_role'] === $role;
    }
    
    /**
     * Verifica si el usuario tiene acceso a una página
     */
    public static function checkAccess($moduleUrl, $action = 'view') {
        if (!self::isLoggedIn()) {
            return false;
        }
        
        // Si es admin, tiene acceso a todo
        if ($_SESSION['user_role'] === 'admin') {
            return true;
        }
        
        // Crear instancia de Permission
        require_once __DIR__ . '/../models/Permission.php';
        $permissionModel = new Permission();
        return $permissionModel->hasAccess($_SESSION['user_id'], $moduleUrl, $action);
    }
    
    /**
     * Requiere que el usuario esté autenticado, redirige si no lo está
     */
    public static function requireLogin() {
        if (!self::isLoggedIn()) {
            $_SESSION['redirect_after_login'] = $_SERVER['REQUEST_URI'];
            header('Location: ' . BASE_URL . '/login.php');
            exit;
        }
    }
    
    /**
     * Requiere que el usuario tenga un rol específico
     */
    public static function requireRole($role) {
        self::requireLogin();
        
        if (!self::hasRole($role)) {
            self::redirectUnauthorized();
        }
    }
    
    /**
     * Requiere que el usuario tenga un permiso específico
     */
    public static function requirePermission($moduleUrl, $action = 'view') {
        self::requireLogin();
        
        if (!self::checkAccess($moduleUrl, $action)) {
            self::redirectUnauthorized();
        }
    }
    
    /**
     * Redirige a una página de acceso no autorizado
     */
    private static function redirectUnauthorized() {
        header('Location: ' . BASE_URL . '/401.php');
        exit;
    }
    
    /**
     * Inicia sesión para un usuario
     */
    public static function login($user) {
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['username'] = $user['username'];
        $_SESSION['name'] = $user['name'];
        $_SESSION['user_role'] = $user['role'];
        $_SESSION['role'] = $user['role']; // Para mantener compatibilidad
        
        // Actualizar último inicio de sesión
        db_update('users', ['last_login' => date('Y-m-d H:i:s')], 'id = ?', [$user['id']]);
        
        // Registrar en el log
        log_system_message(
            'info',
            'Usuario ' . $user['username'] . ' ha iniciado sesión',
            null,
            'IP: ' . $_SERVER['REMOTE_ADDR']
        );
        
        // Redirección después del login
        if (isset($_SESSION['redirect_after_login'])) {
            $redirect = $_SESSION['redirect_after_login'];
            unset($_SESSION['redirect_after_login']);
            header('Location: ' . $redirect);
        } else {
            header('Location: ' . BASE_URL . '/index.php');
        }
        exit;
    }
    
    /**
     * Cierra la sesión
     */
    public static function logout() {
        if (isset($_SESSION['user_id'])) {
            log_system_message(
                'info',
                'Usuario ' . ($_SESSION['username'] ?? 'desconocido') . ' ha cerrado sesión',
                null,
                'IP: ' . $_SERVER['REMOTE_ADDR']
            );
        }
        
        session_unset();
        session_destroy();
        
        header('Location: ' . BASE_URL . '/login.php');
        exit;
    }
}