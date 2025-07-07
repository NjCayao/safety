<?php
/**
 * API para ejecutar el monitor de reportes bajo demanda
 * 
 * Este endpoint permite ejecutar el script monitor_reports.php desde una solicitud AJAX
 */

// Asegurarse de que siempre se devuelva JSON, incluso en caso de error
function returnJsonResponse($status, $message = '', $data = []) {
    header('Content-Type: application/json');
    echo json_encode([
        'status' => $status,
        'message' => $message,
        'data' => $data
    ]);
    exit;
}

// Capturar errores fatales para convertirlos en JSON
register_shutdown_function(function() {
    $error = error_get_last();
    if ($error && in_array($error['type'], [E_ERROR, E_PARSE, E_CORE_ERROR, E_COMPILE_ERROR])) {
        header('Content-Type: application/json');
        echo json_encode([
            'status' => 'error',
            'message' => 'Error fatal: ' . $error['message'] . ' en ' . $error['file'] . ' línea ' . $error['line'],
            'data' => []
        ]);
        exit;
    }
});

// Definir constantes de rutas
define('BASE_PATH', dirname(dirname(dirname(__FILE__)))); // Ruta base del proyecto
define('MONITOR_SCRIPT', BASE_PATH . '/scripts/monitor_reports.php'); // Ruta al script monitor

// Verificar si el archivo existe
if (!file_exists(MONITOR_SCRIPT)) {
    returnJsonResponse('error', 'Script de monitoreo no encontrado en: ' . MONITOR_SCRIPT);
}

try {
    // Iniciar buffer de salida para capturar la salida del script
    ob_start();
    
    // Incluir archivos necesarios
    require_once BASE_PATH . '/config/config.php';
    require_once BASE_PATH . '/config/database.php';
    
    // Variables para almacenar resultados
    $processedCount = 0;
    $failedCount = 0;
    
    // Ejecutar el script como un proceso separado para evitar errores
    $command = 'php ' . MONITOR_SCRIPT;
    $output = [];
    $returnCode = 0;
    
    // Ejecutar el comando y capturar la salida
    exec($command, $output, $returnCode);
    
    // Verificar si el comando se ejecutó correctamente
    if ($returnCode !== 0) {
        returnJsonResponse('error', 'Error al ejecutar el monitor de reportes', [
            'output' => implode("\n", $output),
            'return_code' => $returnCode
        ]);
    }
    
    // Procesar la salida para extraer información relevante
    $processedCount = 0;
    $failedCount = 0;
    $message = '';
    
    foreach ($output as $line) {
        // Buscar información sobre archivos procesados
        if (strpos($line, 'Procesados:') !== false) {
            $message = $line;
            if (preg_match('/Procesados: (\d+). Fallidos: (\d+)/', $line, $matches)) {
                $processedCount = (int)$matches[1];
                $failedCount = (int)$matches[2];
            }
        }
    }
    
    // Devolver respuesta exitosa
    returnJsonResponse('success', $message ?: 'Monitor ejecutado correctamente', [
        'processed' => $processedCount,
        'failed' => $failedCount,
        'output' => implode("\n", $output)
    ]);
    
} catch (Exception $e) {
    // En caso de error, devolver respuesta de error
    returnJsonResponse('error', 'Error al ejecutar el monitor: ' . $e->getMessage());
}
?>