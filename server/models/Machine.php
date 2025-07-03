<?php
// require_once 'config/database.php';

class Machine {
    /**
     * Obtiene todas las máquinas
     */
    public function getAllMachines() {
        return db_fetch_all("SELECT * FROM machines ORDER BY name");
    }
    
    /**
     * Obtiene una máquina específica por ID
     */
    public function getById($id) {
        return db_fetch_one("SELECT * FROM machines WHERE id = ?", [$id]);
    }
    
    /**
     * Cuenta el número de máquinas activas
     */
    public function countMachines($onlyActive = true) {
        $sql = "SELECT COUNT(*) as count FROM machines";
        if ($onlyActive) {
            $sql .= " WHERE status = 'active'";
        }
        $result = db_fetch_one($sql);
        return $result ? $result['count'] : 0;
    }
    
    /**
     * Crea una nueva máquina
     */
    public function create($data) {
        return db_insert('machines', $data);
    }
    
    /**
     * Actualiza una máquina existente
     */
    public function update($id, $data) {
        return db_update('machines', $data, 'id = ?', [$id]);
    }
    
    /**
     * Elimina una máquina
     */
    public function delete($id) {
        return db_delete('machines', 'id = ?', [$id]);
    }
}