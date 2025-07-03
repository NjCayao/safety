<?php
// Configuración para mostrar errores
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

echo "<h1>Actualización de progress_functions.php</h1>";

// Ruta correcta al archivo
$prog_file = "C:/xampp/htdocs/safety_system/server/includes/progress_functions.php";
$prog_file = str_replace('/', DIRECTORY_SEPARATOR, $prog_file);

echo "<p>Ruta al archivo: $prog_file</p>";

// Verificar si el directorio existe
$includes_dir = dirname($prog_file);
if (!file_exists($includes_dir)) {
    echo "<p>El directorio no existe. Intentando crear: $includes_dir</p>";
    if (mkdir($includes_dir, 0777, true)) {
        echo "<p style='color:green'>Directorio creado con éxito.</p>";
    } else {
        echo "<p style='color:red'>Error al crear el directorio.</p>";
    }
}

// Contenido del archivo
$prog_content = <<<EOT
<?php
/**
 * Funciones para el manejo del progreso de operaciones
 */

/**
 * Lee el progreso actual desde el archivo
 * @param string \$operation_type Tipo de operación (por defecto: update_encodings)
 * @return int Porcentaje de progreso (0-100)
 */
function get_progress(\$operation_type = 'update_encodings') {
    // Ruta explícita al archivo de progreso
    \$safety_root = "C:/xampp/htdocs/safety_system";
    \$operators_dir = "\$safety_root/operators";
    \$progress_file = "\$operators_dir/update_progress.txt";
    \$progress_file = str_replace('/', DIRECTORY_SEPARATOR, \$progress_file);
    
    // Si el archivo no existe, el progreso es 0
    if (!file_exists(\$progress_file)) {
        return 0;
    }
    
    // Lee el contenido del archivo
    \$progress = intval(file_get_contents(\$progress_file));
    
    // Asegura que el progreso esté entre 0 y 100
    return max(0, min(100, \$progress));
}

/**
 * Verifica si una operación está en curso
 * @param string \$operation_type Tipo de operación
 * @return bool True si está en curso, False si no
 */
function is_operation_in_progress(\$operation_type = 'update_encodings') {
    \$progress = get_progress(\$operation_type);
    return \$progress > 0 && \$progress < 100;
}

/**
 * Lee el log de la operación
 * @param string \$operation_type Tipo de operación
 * @return string Contenido del log
 */
function get_operation_log(\$operation_type = 'update_encodings') {
    // Ruta explícita al archivo de log
    \$safety_root = "C:/xampp/htdocs/safety_system";
    \$operators_dir = "\$safety_root/operators";
    \$log_file = "\$operators_dir/update_log.txt";
    \$log_file = str_replace('/', DIRECTORY_SEPARATOR, \$log_file);
    
    // Si el archivo no existe, devuelve mensaje vacío
    if (!file_exists(\$log_file)) {
        return "No hay registro de operaciones.";
    }
    
    // Lee el contenido del archivo
    return file_get_contents(\$log_file);
}

/**
 * Obtiene información del archivo encodings.pkl
 * @return array Información del archivo
 */
function get_encodings_info() {
    // Ruta explícita al archivo encodings.pkl
    \$safety_root = "C:/xampp/htdocs/safety_system";
    \$operators_dir = "\$safety_root/operators";
    \$pickle_file = "\$operators_dir/encodings.pkl";
    \$pickle_file = str_replace('/', DIRECTORY_SEPARATOR, \$pickle_file);
    
    \$info = array(
        'exists' => false,
        'size' => 0,
        'modified' => '',
        'modified_timestamp' => 0
    );
    
    if (file_exists(\$pickle_file)) {
        \$info['exists'] = true;
        \$info['size'] = filesize(\$pickle_file);
        \$info['modified_timestamp'] = filemtime(\$pickle_file);
        \$info['modified'] = date('Y-m-d H:i:s', \$info['modified_timestamp']);
    }
    
    return \$info;
}
?>
EOT;

// Escribir el archivo
if (file_put_contents($prog_file, $prog_content)) {
    echo "<p style='color:green'>Archivo actualizado con éxito.</p>";
} else {
    echo "<p style='color:red'>Error al actualizar el archivo.</p>";
}

echo "<p><a href='../index.php' class='btn btn-primary'>Volver a la página principal</a></p>";
?>