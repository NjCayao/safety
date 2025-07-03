<?php
/**
 * Funciones para el manejo de logs del sistema
 */

/**
 * Registra un mensaje en el archivo de log del sistema
 * @param string $message Mensaje a registrar
 * @param string $level Nivel del mensaje (INFO, WARNING, ERROR)
 * @param string $module Módulo que genera el mensaje
 * @return bool True si se registró correctamente, False si no
 */
function log_message($message, $level = 'INFO', $module = 'SYSTEM') {
    // Ruta al archivo de log
    $log_dir = dirname(dirname(__FILE__)) . '/logs';
    
    // Crea el directorio si no existe
    if (!file_exists($log_dir)) {
        mkdir($log_dir, 0755, true);
    }
    
    // Nombre del archivo con la fecha actual
    $log_file = $log_dir . '/system_' . date('Y-m-d') . '.log';
    
    // Formato del mensaje
    $log_entry = '[' . date('Y-m-d H:i:s') . '] [' . $level . '] [' . $module . '] ' . $message . PHP_EOL;
    
    // Escribe en el archivo
    return file_put_contents($log_file, $log_entry, FILE_APPEND) !== false;
}

/**
 * Registra un error
 * @param string $message Mensaje de error
 * @param string $module Módulo que genera el error
 * @return bool True si se registró correctamente, False si no
 */
function log_error($message, $module = 'SYSTEM') {
    return log_message($message, 'ERROR', $module);
}

/**
 * Registra una advertencia
 * @param string $message Mensaje de advertencia
 * @param string $module Módulo que genera la advertencia
 * @return bool True si se registró correctamente, False si no
 */
function log_warning($message, $module = 'SYSTEM') {
    return log_message($message, 'WARNING', $module);
}

/**
 * Registra información
 * @param string $message Mensaje informativo
 * @param string $module Módulo que genera la información
 * @return bool True si se registró correctamente, False si no
 */
function log_info($message, $module = 'SYSTEM') {
    return log_message($message, 'INFO', $module);
}
?>