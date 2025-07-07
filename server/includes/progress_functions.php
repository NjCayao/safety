<?php
/**
 * Funciones para el manejo del progreso de operaciones
 */

/**
 * Devuelve la fecha y hora actual en la zona horaria de Perú
 * @return string Fecha y hora en formato Y-m-d H:i:s
 */
function get_peru_datetime() {
    date_default_timezone_set('America/Lima');
    return date('Y-m-d H:i:s');
}

/**
 * Lee el progreso actual desde el archivo
 * @param string $operation_type Tipo de operación (por defecto: update_encodings)
 * @return int Porcentaje de progreso (0-100)
 */
function get_progress($operation_type = 'update_encodings') {
    // Ruta explícita al archivo de progreso
    $safety_root = "C:/xampp/htdocs/safety_system";
    $operators_dir = "$safety_root/operators";
    $progress_file = "$operators_dir/update_progress.txt";
    $progress_file = str_replace('/', DIRECTORY_SEPARATOR, $progress_file);
    
    // Si el archivo no existe, el progreso es 0
    if (!file_exists($progress_file)) {
        return 0;
    }
    
    // Lee el contenido del archivo
    $progress = intval(file_get_contents($progress_file));
    
    // Asegura que el progreso esté entre 0 y 100
    return max(0, min(100, $progress));
}

/**
 * Verifica si una operación está en curso
 * @param string $operation_type Tipo de operación
 * @return bool True si está en curso, False si no
 */
function is_operation_in_progress($operation_type = 'update_encodings') {
    $progress = get_progress($operation_type);
    return $progress > 0 && $progress < 100;
}

/**
 * Lee el log de la operación
 * @param string $operation_type Tipo de operación
 * @return string Contenido del log
 */
function get_operation_log($operation_type = 'update_encodings') {
    // Ruta explícita al archivo de log
    $safety_root = "C:/xampp/htdocs/safety_system";
    $operators_dir = "$safety_root/operators";
    $log_file = "$operators_dir/update_log.txt";
    $log_file = str_replace('/', DIRECTORY_SEPARATOR, $log_file);
    
    // Si el archivo no existe, devuelve mensaje vacío
    if (!file_exists($log_file)) {
        return "No hay registro de operaciones.";
    }
    
    // Lee el contenido del archivo
    return file_get_contents($log_file);
}

/**
 * Obtiene información del archivo encodings.pkl
 * @return array Información del archivo
 */
function get_encodings_info() {
    // Asegurar que se use la zona horaria de Perú
    date_default_timezone_set('America/Lima');
    
    // Ruta explícita al archivo encodings.pkl
    $safety_root = "C:/xampp/htdocs/safety_system";
    $operators_dir = "$safety_root/operators";
    $pickle_file = "$operators_dir/encodings.pkl";
    $pickle_file = str_replace('/', DIRECTORY_SEPARATOR, $pickle_file);
    
    $info = array(
        'exists' => false,
        'size' => 0,
        'modified' => '',
        'modified_timestamp' => 0
    );
    
    if (file_exists($pickle_file)) {
        $info['exists'] = true;
        $info['size'] = filesize($pickle_file);
        $info['modified_timestamp'] = filemtime($pickle_file);
        $info['modified'] = date('Y-m-d H:i:s', $info['modified_timestamp']);
    }
    
    return $info;
}
?>