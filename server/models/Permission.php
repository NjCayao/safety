<?php
// models/Permission.php

class Permission {
    /**
     * Obtiene todos los módulos
     */
    public function getAllModules() {
        return db_fetch_all("SELECT * FROM modules WHERE status = 'active' ORDER BY `order`");
    }
    
    /**
     * Obtiene módulos para un menú jerárquico
     */
    public function getModulesForMenu() {
        $allModules = $this->getAllModules();
        $moduleTree = [];
        
        // Primero, encontrar todos los módulos padre
        foreach ($allModules as $module) {
            if ($module['parent_id'] === null) {
                $module['children'] = [];
                $moduleTree[$module['id']] = $module;
            }
        }
        
        // Luego, agregar los hijos a sus padres
        foreach ($allModules as $module) {
            if ($module['parent_id'] !== null && isset($moduleTree[$module['parent_id']])) {
                $moduleTree[$module['parent_id']]['children'][] = $module;
            }
        }
        
        return $moduleTree;
    }
    
    /**
     * Obtiene permisos de un usuario
     */
    public function getUserPermissions($userId) {
        $permissions = db_fetch_all(
            "SELECT p.*, m.name as module_name 
             FROM permissions p 
             JOIN modules m ON p.module_id = m.id 
             WHERE p.user_id = ?", 
            [$userId]
        );
        
        $result = [];
        foreach ($permissions as $permission) {
            $result[$permission['module_id']] = $permission;
        }
        
        return $result;
    }
    
    /**
     * Verifica si un usuario tiene acceso a un módulo específico
     */
    public function hasAccess($userId, $moduleUrl, $action = 'view') {
        // Los administradores tienen acceso total
        $user = db_fetch_one("SELECT role FROM users WHERE id = ?", [$userId]);
        if ($user && $user['role'] === 'admin') {
            return true;
        }
        
        // Obtener el ID del módulo por la URL
        $moduleInfo = db_fetch_one("SELECT id FROM modules WHERE url = ? OR url LIKE ?", [$moduleUrl, $moduleUrl . '%']);
        
        if (!$moduleInfo) {
            return false;
        }
        
        $moduleId = $moduleInfo['id'];
        
        // Verificar permiso específico
        $permissionColumn = 'can_' . $action;
        $permission = db_fetch_one(
            "SELECT $permissionColumn FROM permissions WHERE user_id = ? AND module_id = ?", 
            [$userId, $moduleId]
        );
        
        return ($permission && $permission[$permissionColumn] == 1);
    }
    
    /**
     * Asigna permisos a un usuario
     */
    public function assignPermissions($userId, $permissions) {
        // Primero eliminar permisos existentes
        db_delete('permissions', 'user_id = ?', [$userId]);
        
        // Luego insertar los nuevos
        foreach ($permissions as $moduleId => $permission) {
            $data = [
                'user_id' => $userId,
                'module_id' => $moduleId,
                'can_view' => isset($permission['view']) ? 1 : 0,
                'can_create' => isset($permission['create']) ? 1 : 0,
                'can_edit' => isset($permission['edit']) ? 1 : 0,
                'can_delete' => isset($permission['delete']) ? 1 : 0
            ];
            
            db_insert('permissions', $data);
        }
        
        return true;
    }
}