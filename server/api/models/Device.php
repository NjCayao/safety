<?php
// server/api/models/Device.php

class Device {
    private $conn;
    private $table_name = "devices";
    
    // Propiedades
    public $id;
    public $device_id;
    public $api_key;
    public $device_type;
    public $machine_id;
    public $status;
    public $last_sync;
    public $last_access;
    
    public function __construct($db) {
        $this->conn = $db;
    }
    
    public function verifyCredentials($device_id, $api_key) {
        $query = "SELECT * FROM " . $this->table_name . " 
                  WHERE device_id = ? LIMIT 0,1";
        
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $device_id);
        $stmt->execute();
        
        if ($stmt->rowCount() > 0) {
            $row = $stmt->fetch(PDO::FETCH_ASSOC);
            
            $this->id = $row['id'];
            $this->device_id = $row['device_id'];
            $this->api_key = $row['api_key'];
            $this->device_type = $row['device_type'];
            $this->machine_id = $row['machine_id'];
            $this->status = $row['status'];
            $this->last_sync = $row['last_sync'];
            $this->last_access = $row['last_access'];
            
            // Verificar que la API key coincida (idealmente usar hash)
            return password_verify($api_key, $this->api_key);
        }
        
        return false;
    }
    
    public function updateLastAccess() {
        $query = "UPDATE " . $this->table_name . "
                  SET last_access = NOW()
                  WHERE id = ?";
        
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $this->id);
        
        return $stmt->execute();
    }
    
    public function updateSyncStatus($status = 'synced') {
        $query = "UPDATE " . $this->table_name . "
                  SET status = ?, last_sync = NOW()
                  WHERE id = ?";
        
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $status);
        $stmt->bindParam(2, $this->id);
        
        return $stmt->execute();
    }
}
?>