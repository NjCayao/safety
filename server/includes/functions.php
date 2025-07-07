<?php

/**
 * Funciones comunes para todo el sistema
 */

// Función para mostrar mensajes de alerta
function showAlert($message, $type = 'success')
{
    $validTypes = ['success', 'danger', 'warning', 'info'];
    $type = in_array($type, $validTypes) ? $type : 'info';

    echo '<div class="alert alert-' . $type . ' alert-dismissible fade show" role="alert">';
    echo $message;
    echo '<button type="button" class="close" data-dismiss="alert" aria-label="Close">';
    echo '<span aria-hidden="true">&times;</span>';
    echo '</button>';
    echo '</div>';
}

// Función para escapar HTML y prevenir XSS
function escape($string)
{
    return htmlspecialchars($string, ENT_QUOTES, 'UTF-8');
}

// Función para obtener la fecha formateada
function formatDate($date, $format = 'd/m/Y H:i')
{
    return date($format, strtotime($date));
}

// Función para obtener tiempo transcurrido en formato legible
function timeAgo($datetime)
{
    // Asegurarnos de que se use la zona horaria correcta
    $default_timezone = date_default_timezone_get();
    date_default_timezone_set('America/Lima'); // Ajusta esto a tu zona horaria

    $time = strtotime($datetime);
    $now = time();
    $diff = $now - $time;

    // Restaurar zona horaria
    date_default_timezone_set($default_timezone);

    if ($diff < 60) {
        return 'hace menos de un minuto';
    } elseif ($diff < 3600) {
        $minutes = floor($diff / 60);
        return 'hace ' . $minutes . ' minuto' . ($minutes > 1 ? 's' : '');
    } elseif ($diff < 86400) {
        $hours = floor($diff / 3600);
        return 'hace ' . $hours . ' hora' . ($hours > 1 ? 's' : '');
    } elseif ($diff < 2592000) {
        $days = floor($diff / 86400);
        return 'hace ' . $days . ' día' . ($days > 1 ? 's' : '');
    } elseif ($diff < 31536000) {
        $months = floor($diff / 2592000);
        return 'hace ' . $months . ' mes' . ($months > 1 ? 'es' : '');
    } else {
        $years = floor($diff / 31536000);
        return 'hace ' . $years . ' año' . ($years > 1 ? 's' : '');
    }
}

// Función para generar un token CSRF
function generateCSRFToken()
{
    if (!isset($_SESSION['csrf_token'])) {
        $_SESSION['csrf_token'] = bin2hex(random_bytes(32));
    }
    return $_SESSION['csrf_token'];
}

// Función para verificar un token CSRF
function verifyCSRFToken($token)
{
    if (!isset($_SESSION['csrf_token']) || $token !== $_SESSION['csrf_token']) {
        return false;
    }
    return true;
}

function log_system_message($type, $message, $machine_id = null, $details = null)
{
    try {
        // Registro en la base de datos (más confiable que archivos)
        $data = [
            'log_type' => $type,
            'machine_id' => $machine_id,
            'message' => $message,
            'details' => $details,
            'timestamp' => date('Y-m-d H:i:s')
        ];

        return db_insert('system_logs', $data);
    } catch (Exception $e) {
        // En caso de error, intentamos guardar en un archivo
        $logDir = __DIR__ . '/../logs';

        // Crear directorio logs si no existe
        if (!is_dir($logDir)) {
            mkdir($logDir, 0777, true);
        }

        $logFile = $logDir . '/system.log';
        $timestamp = date('Y-m-d H:i:s');

        // Formato del mensaje
        $logMessage = "[$timestamp] [$type] ";
        $logMessage .= $machine_id ? "[Machine: $machine_id] " : "";
        $logMessage .= $details ? "[Details: $details] " : "";
        $logMessage .= $message . PHP_EOL;

        // Escribir en el archivo de log
        error_log($logMessage, 3, $logFile);
    }
}




// Función para subir archivos
function uploadFile($file, $directory = 'uploads', $allowedTypes = ['jpg', 'jpeg', 'png'], $maxSize = 5242880)
{
    // Verificar si hay errores
    if ($file['error'] !== UPLOAD_ERR_OK) {
        return ['success' => false, 'message' => 'Error al subir el archivo.'];
    }

    // Verificar tamaño
    if ($file['size'] > $maxSize) {
        return ['success' => false, 'message' => 'El archivo es demasiado grande.'];
    }

    // Verificar tipo
    $fileExtension = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    if (!in_array($fileExtension, $allowedTypes)) {
        return ['success' => false, 'message' => 'Tipo de archivo no permitido.'];
    }

    // Crear directorio si no existe
    if (!file_exists($directory)) {
        mkdir($directory, 0777, true);
    }

    // Generar nombre de archivo único
    $newFilename = uniqid() . '.' . $fileExtension;
    $destination = $directory . '/' . $newFilename;

    // Mover archivo
    if (move_uploaded_file($file['tmp_name'], $destination)) {
        return ['success' => true, 'path' => $destination, 'filename' => $newFilename];
    } else {
        return ['success' => false, 'message' => 'Error al guardar el archivo.'];
    }
}

// Función para obtener el estado de conexión de un dispositivo
function getDeviceConnectionStatus($lastAccess) {
    if (!$lastAccess) return 'offline';
    
    $now = new DateTime();
    $last = new DateTime($lastAccess);
    $diff = $now->diff($last);
    
    $minutes = ($diff->days * 24 * 60) + ($diff->h * 60) + $diff->i;
    
    if ($minutes <= 5) {
        return 'online';
    } elseif ($minutes <= 15) {
        return 'warning';
    } else {
        return 'offline';
    }
}

// Función para registrar eventos de dispositivos
function logDeviceEvent($deviceId, $eventType, $message, $details = null) {
    return db_insert('system_logs', [
        'log_type' => 'info',
        'message' => $message,
        'details' => json_encode([
            'device_id' => $deviceId,
            'event_type' => $eventType,
            'details' => $details
        ]),
        'timestamp' => date('Y-m-d H:i:s')
    ]);
} 
