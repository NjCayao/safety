<?php
// server/api/config/database.php

class Database {
    // Credenciales de la base de datos
    private $host = "localhost";
    private $db_name = "safetysystem";
    private $username = "safetysystesdm";
    private $password = "Thenilfer141ds4";
    public $conn;
    
    // Obtener la conexión a la base de datos
    public function getConnection() {
        $this->conn = null;
        
        try {
            $this->conn = new PDO(
                "mysql:host=" . $this->host . ";dbname=" . $this->db_name,
                $this->username,
                $this->password
            );
            $this->conn->exec("set names utf8");
            $this->conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        } catch(PDOException $exception) {
            echo "Error de conexión: " . $exception->getMessage();
        }
        
        return $this->conn;
    }
}
?>