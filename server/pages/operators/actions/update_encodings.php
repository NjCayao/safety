<?php
// Incluir funciones necesarias
require_once('../../../includes/progress_functions.php');
require_once('../../../includes/log_functions.php');

// Configurar zona horaria a Perú
date_default_timezone_set('America/Lima');

// Función para ejecutar el script de Python
function execute_python_script() {
     // Rutas relativas correctas
    $current_file = __FILE__; // /server/pages/operators/actions/update_encodings.php
    $actions_dir = dirname($current_file); // .../actions/
    $operators_page_dir = dirname($actions_dir); // .../operators/
    $pages_dir = dirname($operators_page_dir); // .../pages/
    $server_dir = dirname($pages_dir); // .../server/
    $base_dir = dirname($server_dir); // .../safety_system/
    
    $operators_dir = $base_dir . DIRECTORY_SEPARATOR . 'operators';
    $python_script = $operators_dir . DIRECTORY_SEPARATOR . 'update_encodings.py';
    
    // Normalizar rutas
    $operators_dir = str_replace('/', DIRECTORY_SEPARATOR, $operators_dir);
    $python_script = str_replace('/', DIRECTORY_SEPARATOR, $python_script);

    // Crea un archivo para indicar que está iniciando
    if (!file_exists($operators_dir)) {
        mkdir($operators_dir, 0755, true);
    }

    // Reinicia el progreso a 0 (importante para la barra de progreso)
    file_put_contents($operators_dir . '/update_progress.txt', '0');
    
    // Comando para ejecutar Python
    $python_cmd = 'python';
    
    // Crea un archivo para indicar que está iniciando
    if (!file_exists($operators_dir)) {
        mkdir($operators_dir, 0755, true);
    }
    
    file_put_contents($operators_dir . '/update_progress.txt', '1');
    file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Iniciando ejecución del script Python' . PHP_EOL, FILE_APPEND);
    
    // Registra inicio de la operación
    log_info('Iniciando actualización de encodings faciales', 'FACIAL_RECOGNITION');
    
    // Registra las rutas para depuración
    file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Script Python: ' . $python_script . PHP_EOL, FILE_APPEND);
    file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Directorio de trabajo: ' . $operators_dir . PHP_EOL, FILE_APPEND);
    
    // Obtiene información del archivo encodings.pkl antes de la actualización
    $encodings_info_before = get_encodings_info();
    if ($encodings_info_before['exists']) {
        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Archivo encodings.pkl existente - Tamaño: ' . 
            $encodings_info_before['size'] . ' bytes, Modificado: ' . $encodings_info_before['modified'] . PHP_EOL, FILE_APPEND);
    } else {
        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Archivo encodings.pkl no existente' . PHP_EOL, FILE_APPEND);
    }
    
    // Inicia el proceso en segundo plano (non-blocking)
    if (substr(php_uname(), 0, 7) == "Windows") {
        // Para Windows, usamos un enfoque diferente que nos da más información
        $cmd = "cd /d \"$operators_dir\" && $python_cmd \"$python_script\" > \"$operators_dir/python_output.txt\" 2>&1";
        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Comando ejecutado: ' . $cmd . PHP_EOL, FILE_APPEND);
        
        // Ejecuta el comando y captura cualquier salida
        $output = [];
        $return_var = 0;
        exec($cmd, $output, $return_var);
        
        // Registra el resultado
        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Código de retorno: ' . $return_var . PHP_EOL, FILE_APPEND);
        
        if ($return_var !== 0) {
            // Hubo un error
            file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: ERROR al ejecutar el script Python' . PHP_EOL, FILE_APPEND);
        } else {
            file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Script Python ejecutado exitosamente' . PHP_EOL, FILE_APPEND);
            
            // Obtiene información del archivo encodings.pkl después de la actualización
            $encodings_info_after = get_encodings_info();
            if ($encodings_info_after['exists']) {
                file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Archivo encodings.pkl actualizado - Tamaño: ' . 
                    $encodings_info_after['size'] . ' bytes, Modificado: ' . $encodings_info_after['modified'] . PHP_EOL, FILE_APPEND);
                
                // Verifica si el archivo fue modificado
                if ($encodings_info_before['exists']) {
                    if ($encodings_info_after['modified_timestamp'] > $encodings_info_before['modified_timestamp']) {
                        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: El archivo fue modificado correctamente' . PHP_EOL, FILE_APPEND);
                    } else {
                        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: ADVERTENCIA: El archivo no fue modificado' . PHP_EOL, FILE_APPEND);
                    }
                }
            } else {
                file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: ERROR: No se encontró el archivo encodings.pkl después de la ejecución' . PHP_EOL, FILE_APPEND);
            }
        }
    } else {
        // Para sistemas Unix/Linux
        $cmd = "cd $operators_dir && $python_cmd $python_script > $operators_dir/python_output.txt 2>&1 &";
        file_put_contents($operators_dir . '/update_log.txt', '[' . date('Y-m-d H:i:s') . '] PHP: Comando ejecutado: ' . $cmd . PHP_EOL, FILE_APPEND);
        exec($cmd);
    }
    
    // Espera un momento para que el script comience y genere el archivo de progreso
    sleep(1);
    
    return true;
}

// Verifica si ya hay una operación en curso
$in_progress = is_operation_in_progress('update_encodings');

// Si hay una solicitud AJAX para verificar el progreso
if (isset($_GET['check_progress'])) {
    header('Content-Type: application/json');
    $operators_dir = "C:/xampp/htdocs/safety_system/operators";
    $operators_dir = str_replace('/', DIRECTORY_SEPARATOR, $operators_dir);
    
    $log_content = file_exists($operators_dir . '/update_log.txt') ? 
                file_get_contents($operators_dir . '/update_log.txt') : 
                'No hay archivo de log';
                
    // También incluye el contenido del archivo python_output.txt si existe
    $python_output = '';
    if (file_exists($operators_dir . '/python_output.txt')) {
        $python_output = file_get_contents($operators_dir . '/python_output.txt');
    }
    
    // Información del archivo encodings.pkl
    $encodings_info = get_encodings_info();
    
    echo json_encode([
        'progress' => get_progress('update_encodings'),
        'in_progress' => $in_progress,
        'log' => $log_content,
        'python_output' => $python_output,
        'encodings_info' => $encodings_info
    ]);
    exit;
}

// Si se recibe una solicitud POST (botón presionado) y no hay operación en curso
if ($_SERVER['REQUEST_METHOD'] === 'POST' && !$in_progress) {
    // Ejecuta el script Python
    execute_python_script();
    
    // Redirecciona a la página principal con un mensaje
    header('Location: ../index.php?msg=update_started');
    exit;
} elseif ($_SERVER['REQUEST_METHOD'] === 'POST' && $in_progress) {
    // Redirecciona con mensaje de que ya hay una operación en curso
    header('Location: ../index.php?msg=already_in_progress');
    exit;
} else {
    // Si no es una solicitud válida, redirige a la página principal
    header('Location: ../index.php');
    exit;
}
?>