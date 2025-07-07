<?php
// server/api/v1/events/upload_image.php

// Configuración de cabeceras CORS
header("Access-Control-Allow-Origin: *");
header("Access-Control-Allow-Methods: POST");
header("Access-Control-Max-Age: 3600");
header("Access-Control-Allow-Headers: Content-Type, Access-Control-Allow-Headers, Authorization, X-Requested-With");

// Incluir archivos necesarios
include_once '../../../config/database.php';
include_once '../../utils/authenticate.php';
include_once '../../models/Event.php';
include_once '../../utils/FileManager.php';

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
$fileManager = new FileManager();

// Verificar que se haya enviado un archivo y un event_id
if (!isset($_FILES['image']) || !isset($_POST['event_id'])) {
    echo Response::error('No se proporcionó imagen o ID del evento', 400);
    exit();
}

$event_id = $_POST['event_id'];

// Verificar que el evento exista
if (!$event->findById($event_id)) {
    echo Response::error('Evento no encontrado', 404);
    exit();
}

// Verificar que el evento pertenezca al dispositivo autenticado
if ($event->device_id !== $device_data['device_id']) {
    echo Response::error('No autorizado para modificar este evento', 403);
    exit();
}

// Procesar la imagen
$result = $fileManager->saveEventImage($event_id, $_FILES['image'], $event->event_type);

if ($result) {
    // Actualizar el evento con la ruta de la imagen
    if ($event->updateImagePath($event_id, $result['path'])) {
        echo Response::success([
            'event_id' => $event_id,
            'image_path' => $result['url']
        ], 'Imagen cargada correctamente');
    } else {
        echo Response::error('Error al actualizar la ruta de la imagen', 500);
    }
} else {
    echo Response::error('Error al subir la imagen', 500);
}
?>