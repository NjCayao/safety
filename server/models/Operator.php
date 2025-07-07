<?php
// require_once 'config/database.php';

class Operator {
    /**
     * Obtiene todos los operadores
     */
    public function getAllOperators() {
        return db_fetch_all("SELECT * FROM operators ORDER BY name");
    }
    
    /**
     * Obtiene un operador específico por ID
     */
    public function getById($id) {
        return db_fetch_one("SELECT * FROM operators WHERE id = ?", [$id]);
    }
    
    /**
     * Cuenta el número de operadores activos
     */
    public function countOperators($onlyActive = true) {
        $sql = "SELECT COUNT(*) as count FROM operators";
        if ($onlyActive) {
            $sql .= " WHERE status = 'active'";
        }
        $result = db_fetch_one($sql);
        return $result ? $result['count'] : 0;
    }
    
    /**
     * Crea un nuevo operador
     */
    public function create($data) {
        return db_insert('operators', $data);
    }
    
    /**
     * Actualiza un operador existente
     */
    public function update($id, $data) {
        return db_update('operators', $data, 'id = ?', [$id]);
    }
    
    /**
     * Elimina un operador
     */
    public function delete($id) {
        return db_delete('operators', 'id = ?', [$id]);
    }
}