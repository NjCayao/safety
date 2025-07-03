<?php
// server/api/models/Event.php

class Event {
    private $conn;
    private $table_name = "events";
    
    // Propiedades
    public $id;
    public $device_id;
    public $event_type;
    public $operator_id;
    public $machine_id;
    public $event_data;
    public $image_path;
    public $event_time;
    public $server_time;
    public $sync_batch_id;
    public $is_synced;
    
    public function __construct($db) {
        $this->conn = $db;
    }
    
    // ... otros métodos ...
    
    public function findById($id) {
        $query = "SELECT * FROM " . $this->table_name . " WHERE id = ? LIMIT 0,1";
        $stmt = $this->conn->prepare($query);
        $stmt->bindParam(1, $id);
        $stmt->execute();
        
        if ($stmt->rowCount() > 0) {
            $row = $stmt->fetch(PDO::FETCH_ASSOC);
            
            $this->id = $row['id'];
            $this->device_id = $row['device_id'];
            $this->event_type = $row['event_type'];
            $this->operator_id = $row['operator_id'];
            $this->machine_id = $row['machine_id'];
            $this->event_data = $row['event_data'];
            $this->image_path = $row['image_path'];
            $this->event_time = $row['event_time'];
            $this->server_time = $row['server_time'];
            $this->sync_batch_id = $row['sync_batch_id'];
            $this->is_synced = $row['is_synced'];
            
            return true;
        }
        
        return false;
    }
    
    public function updateImagePath($id, $image_path) {
        $query = "UPDATE " . $this->table_name . " SET image_path = ? WHERE id = ?";
        $stmt = $this->conn->prepare($query);
        
        $stmt->bindParam(1, $image_path);
        $stmt->bindParam(2, $id);
        
        return $stmt->execute();
    }
}
?>