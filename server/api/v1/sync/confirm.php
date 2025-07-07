<?php
// server/api/v1/sync/confirm.php

// Configuración de cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/authenticate.php';
include_once '../../models/Device.php';
include_once '../../models/SyncBatch.php';

// Verificar método
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
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

// Recibir los datos
$data = json_decode(file_get_contents("php://input"));

// Verificar datos requeridos
if (!isset($data->batch_id)) {
    echo Response::error('ID de lote no proporcionado', 400);
    exit();
}

// Verificar que el lote exista
if (!$syncBatch->getBatchStatus($data->batch_id)) {
    echo Response::error('Lote no encontrado', 404);
    exit();
}

// Verificar que el lote pertenezca al dispositivo
if ($syncBatch->device_id !== $device_data['device_id']) {
    echo Response::error('No autorizado para confirmar este lote', 403);
    exit();
}

// Verificar estado del lote
if ($syncBatch->status !== 'completed') {
    echo Response::error('No se puede confirmar un lote no completado', 400);
    exit();
}

// Obtener el dispositivo para actualizar su estado
$device_query = "SELECT id FROM devices WHERE device_id = ? LIMIT 0,1";
$device_stmt = $db->prepare($device_query);
$device_stmt->bindParam(1, $device_data['device_id']);
$device_stmt->execute();

if ($device_stmt->rowCount() > 0) {
    $device_row = $device_stmt->fetch(PDO::FETCH_ASSOC);
    $device->id = $device_row['id'];
    
    // Actualizar estado del dispositivo
    if ($device->updateSyncStatus('online')) {
        echo Response::success([
            'batch_id' => $data->batch_id,
            'status' => 'confirmed',
            'timestamp' => date('Y-m-d H:i:s')
        ], 'Sincronización confirmada');
    } else {
        echo Response::error('Error al actualizar estado de sincronización', 500);
    }
} else {
    echo Response::error('Dispositivo no encontrado', 404);
}
?>