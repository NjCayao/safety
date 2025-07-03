<?php
// server/api/v1/sync/batch.php

// Configuración de cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/authenticate.php';
include_once '../../models/Event.php';
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

// Crear objetos necesarios
$event = new Event($db);
$syncBatch = new SyncBatch($db);

// Recibir los datos del lote
$data = json_decode(file_get_contents("php://input"));

// Verificar datos requeridos
if (!isset($data->batch_id) || !isset($data->events) || !is_array($data->events)) {
    echo Response::error('Datos de sincronización incompletos', 400);
    exit();
}

// Crear registro de lote
$batch = $syncBatch->create([
    'id' => $data->batch_id,
    'device_id' => $device_data['device_id'],
    'batch_size' => count($data->events)
]);

if (!$batch) {
    echo Response::error('Error al registrar lote de sincronización', 500);
    exit();
}

// Procesar cada evento del lote
$success_count = 0;
$errors = [];
$created_events = [];

foreach ($data->events as $eventData) {
    // Verificar datos mínimos requeridos
    if (!isset($eventData->event_type) || !isset($eventData->event_time)) {
        $errors[] = "Evento con datos incompletos";
        continue;
    }

    // Procesar imagen si existe (será recibida en otra solicitud)
    $image_path = null;
    if (isset($eventData->has_image) && $eventData->has_image) {
        // La imagen será subida en otra solicitud usando el event_id como referencia
        $image_path = "pending"; // Marcador para indicar que se espera una imagen
    }

    // Guardar evento
    $event_id = $event->create([
        'device_id' => $device_data['device_id'],
        'event_type' => $eventData->event_type,
        'operator_id' => isset($eventData->operator_id) ? $eventData->operator_id : null,
        'machine_id' => $device_data['machine_id'],
        'event_data' => json_encode($eventData->data ?? []),
        'image_path' => $image_path,
        'event_time' => $eventData->event_time,
        'sync_batch_id' => $data->batch_id,
        'is_synced' => 1
    ]);

    if ($event_id) {
        $success_count++;
        // Guardar mapeo de ID local a ID del servidor
        $created_events[] = [
            'local_id' => isset($eventData->local_id) ? $eventData->local_id : null,
            'id' => $event_id
        ];
    } else {
        $errors[] = "Error al guardar evento tipo " . $eventData->event_type;
    }
}

// Actualizar estado del lote
if ($success_count == count($data->events)) {
    $syncBatch->complete($data->batch_id);
    echo Response::success([
        'batch_id' => $data->batch_id,
        'success_count' => $success_count,
        'total' => count($data->events),
        'events' => $created_events
    ], 'Lote sincronizado completamente');
} else {
    if ($success_count > 0) {
        // Sincronización parcial
        echo Response::success([
            'batch_id' => $data->batch_id,
            'success_count' => $success_count,
            'total' => count($data->events),
            'events' => $created_events,
            'errors' => $errors
        ], 'Sincronización parcial', 206);
    } else {
        // Error total
        $syncBatch->fail($data->batch_id);
        echo Response::error('Error al sincronizar eventos', 500, [
            'batch_id' => $data->batch_id,
            'errors' => $errors
        ]);
    }
}
?>