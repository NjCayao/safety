<?php
// server/api/models/SyncBatch.php

class SyncBatch {
    private $conn;
    private $table_name = "sync_batches";
    
    // Propiedades
    public $id;
    public $device_id;
    public $batch_size;
    public $status;
    public $started_at;
    public $completed_at;
    
    public function __construct($db) {
        $this->conn = $db;
    }
    
    public function create($data) {
        $query = "INSERT INTO " . $this->table_name . "
                  (id, device_id, batch_size, status)
                  VALUES (?, ?, ?, 'processing')";
        
        $stmt = $this->conn->prepare($query);
        
        // Sanitizar datos
        $id = htmlspecialchars(strip_tags($data['id']));
        $device_id = htmlspecialchars(strip_tags($data['device_id']));
        $batch_size = htmlspecialchars(strip_tags($data['batch_size']));
        
        // Vincular parámetros
        $stmt->bindParam(1, $id);
        $stmt->bindParam(2, $device_id);
        $stmt->bindParam(3, $batch_size);
        
        if ($stmt->execute()) {
            return true;
        }
        
        return false;
    }
    
    public function complete($batch_id) {
        $query = "UPDATE " . $this->table_name . "
                  SET status = 'completed', completed_at = NOW()
                  WHERE id = ?";
        
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $batch_id);
        
        return $stmt->execute();
    }
    
    public function fail($batch_id) {
        $query = "UPDATE " . $this->table_name . "
                  SET status = 'failed', completed_at = NOW()
                  WHERE id = ?";
        
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $batch_id);
        
        return $stmt->execute();
    }
    
    public function getBatchStatus($batch_id) {
        $query = "SELECT * FROM " . $this->table_name . " WHERE id = ? LIMIT 0,1";
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $batch_id);
        $stmt->execute();
        
        if ($stmt->rowCount() > 0) {
            $row = $stmt->fetch(PDO::FETCH_ASSOC);
            
            $this->id = $row['id'];
            $this->device_id = $row['device_id'];
            $this->batch_size = $row['batch_size'];
            $this->status = $row['status'];
            $this->started_at = $row['started_at'];
            $this->completed_at = $row['completed_at'];
            
            return true;
        }
        
        return false;
    }
}
?>