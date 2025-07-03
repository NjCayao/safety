<?php
// Archivo para verificar si un DNI ya existe en la base de datos
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';

// Verificar si se recibió un DNI
if (isset($_POST['dni'])) {
    $dni = trim($_POST['dni']);
    
    // Buscar el DNI en la base de datos
    $operator = db_fetch_one(
        "SELECT id, name FROM operators WHERE dni_number = ?",
        [$dni]
    );
    
    // Preparar respuesta
    $response = [
        'exists' => false,
        'id' => '',
        'name' => ''
    ];
    
    if ($operator) {
        $response['exists'] = true;
        $response['id'] = $operator['id'];
        $response['name'] = $operator['name'];
    }
    
    // Enviar respuesta como JSON
    header('Content-Type: application/json');
    echo json_encode($response);
    exit;
}

// Si no se recibió un DNI, devolver error
header('HTTP/1.1 400 Bad Request');
echo 'No se proporcionó un DNI para verificar';