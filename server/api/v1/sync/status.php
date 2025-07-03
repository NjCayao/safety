<?php
// server/api/v1/sync/status.php

// Configuración de cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: GET");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/authenticate.php';
include_once '../../models/Device.php';
include_once '../../models/SyncBatch.php';

// Verificar método
if ($_SERVER['REQUEST_METHOD'] !== 'GET') {
    echo Response::error('Método no permitido', 405);
    exit();
}

// Autenticar solicitud
$device_data = authenticate();

// Obtener conexión a la base de datos
$database = new Database();
$db = $database->getConnection();

// Preparar objetos
$device = new Device($db);
$syncBatch = new SyncBatch($db);

// Buscar dispositivo por device_id
$device_id = $device_data['device_id'];
$query = "SELECT * FROM devices WHERE device_id = ? LIMIT 0,1";
$stmt = $db->prepare($query);
$stmt->bindParam(1, $device_id);
$stmt->execute();

if ($stmt->rowCount() > 0) {
    $row = $stmt->fetch(PDO::FETCH_ASSOC);
    
    // Obtener información de último lote procesado
    $last_batch_query = "SELECT * FROM sync_batches 
                         WHERE device_id = ? 
                         ORDER BY started_at DESC 
                         LIMIT 0,1";
    $batch_stmt = $db->prepare($last_batch_query);
    $batch_stmt->bindParam(1, $device_id);
    $batch_stmt->execute();
    
    $last_batch = null;
    if ($batch_stmt->rowCount() > 0) {
        $last_batch = $batch_stmt->fetch(PDO::FETCH_ASSOC);
    }
    
    // Contar eventos pendientes por sincronizar
    $pending_events_query = "SELECT COUNT(*) as pending_count 
                            FROM events 
                            WHERE device_id = ? AND is_synced = 0";
    $pending_stmt = $db->prepare($pending_events_query);
    $pending_stmt->bindParam(1, $device_id);
    $pending_stmt->execute();
    $pending_row = $pending_stmt->fetch(PDO::FETCH_ASSOC);
    
    echo Response::success([
        'device_id' => $row['device_id'],
        'status' => $row['status'],
        'last_sync' => $row['last_sync'],
        'last_access' => $row['last_access'],
        'last_batch' => $last_batch,
        'pending_events' => intval($pending_row['pending_count'])
    ], 'Estado de sincronización obtenido');
} else {
    echo Response::error('Dispositivo no encontrado', 404);
}
?>