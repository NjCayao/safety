<?php
// Incluir funciones necesarias
require_once('../../../includes/progress_functions.php');
require_once('../../../includes/log_functions.php');

// Configurar zona horaria a Perú
date_default_timezone_set('America/Lima');

// Función para ejecutar el script de calibración
function execute_calibration_script($operator_id) {
    // Obtener la ruta base correctamente
    // Desde /server/pages/operators/actions/ subimos a la raíz
    $current_file = __FILE__; // /server/pages/operators/actions/calibrate_operator.php
    $actions_dir = dirname($current_file); // /server/pages/operators/actions/
    $operators_page_dir = dirname($actions_dir); // /server/pages/operators/
    $pages_dir = dirname($operators_page_dir); // /server/pages/
    $server_dir = dirname($pages_dir); // /server/
    $base_dir = dirname($server_dir); // /safety_system/ (raíz)
    
    // Ahora apuntamos a /operators/ (NO /server/operators/)
    $operators_dir = $base_dir . DIRECTORY_SEPARATOR . 'operators';
    $python_script = $operators_dir . DIRECTORY_SEPARATOR . 'calibrate_single_operator.py';
   
    // Crear directorio si no existe
    if (!file_exists($operators_dir)) {
        mkdir($operators_dir, 0755, true);
    }

    // Reiniciar el progreso a 0
    file_put_contents($operators_dir . DIRECTORY_SEPARATOR . 'calibration_progress.txt', '0');
    
    // Comando para ejecutar Python
    $python_cmd = 'python';
    
    // Iniciar log
    $log_file = $operators_dir . DIRECTORY_SEPARATOR . 'calibration_log.txt';
    file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Iniciando calibración para operador ' . $operator_id . PHP_EOL, FILE_APPEND);
    
    // Registra inicio de la operación
    log_info('Iniciando calibración biométrica para operador: ' . $operator_id, 'CALIBRATION');
    
    // Registra las rutas para depuración
    file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Script Python: ' . $python_script . PHP_EOL, FILE_APPEND);
    file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Directorio de trabajo: ' . $operators_dir . PHP_EOL, FILE_APPEND);
    
    // Construir comando
    if (substr(php_uname(), 0, 7) == "Windows") {
        // Para Windows
        $output_file = $operators_dir . DIRECTORY_SEPARATOR . 'calibration_output.txt';
        $cmd = "cd /d \"$operators_dir\" && \"$python_cmd\" \"calibrate_single_operator.py\" \"$operator_id\" > \"$output_file\" 2>&1";
        file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Comando ejecutado: ' . $cmd . PHP_EOL, FILE_APPEND);
        
        // Ejecutar el comando
        $output = [];
        $return_var = 0;
        exec($cmd, $output, $return_var);
        
        // Registrar el resultado
        file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Código de retorno: ' . $return_var . PHP_EOL, FILE_APPEND);
        
        if ($return_var !== 0) {
            file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: ERROR al ejecutar el script Python' . PHP_EOL, FILE_APPEND);
            
            // Leer el archivo de salida para ver el error
            if (file_exists($output_file)) {
                $error_content = file_get_contents($output_file);
                file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Error Python: ' . $error_content . PHP_EOL, FILE_APPEND);
            }
        } else {
            file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Script Python ejecutado exitosamente' . PHP_EOL, FILE_APPEND);
        }
    } else {
        // Para sistemas Unix/Linux
        $cmd = "cd $operators_dir && $python_cmd calibrate_single_operator.py $operator_id > calibration_output.txt 2>&1 &";
        file_put_contents($log_file, '[' . date('Y-m-d H:i:s') . '] PHP: Comando ejecutado: ' . $cmd . PHP_EOL, FILE_APPEND);
        exec($cmd);
    }
    
    // Esperar un momento para que el script comience
    sleep(1);
    
    return true;
}

// Función para obtener el progreso de calibración
function get_calibration_progress() {
    // Misma lógica para obtener la ruta
    $current_file = __FILE__;
    $actions_dir = dirname($current_file);
    $operators_page_dir = dirname($actions_dir);
    $pages_dir = dirname($operators_page_dir);
    $server_dir = dirname($pages_dir);
    $base_dir = dirname($server_dir);
    $operators_dir = $base_dir . DIRECTORY_SEPARATOR . 'operators';
    
    $progress_file = $operators_dir . DIRECTORY_SEPARATOR . 'calibration_progress.txt';
    
    if (!file_exists($progress_file)) {
        return 0;
    }
    
    $progress = intval(file_get_contents($progress_file));
    return max(0, min(100, $progress));
}

// Función para verificar si hay una calibración en curso
function is_calibration_in_progress() {
    $progress = get_calibration_progress();
    return $progress > 0 && $progress < 100;
}

// Si hay una solicitud AJAX para verificar el progreso
if (isset($_GET['check_progress'])) {
    header('Content-Type: application/json');
    
    // Obtener rutas
    $current_file = __FILE__;
    $actions_dir = dirname($current_file);
    $operators_page_dir = dirname($actions_dir);
    $pages_dir = dirname($operators_page_dir);
    $server_dir = dirname($pages_dir);
    $base_dir = dirname($server_dir);
    $operators_dir = $base_dir . DIRECTORY_SEPARATOR . 'operators';
    
    $log_file = $operators_dir . DIRECTORY_SEPARATOR . 'calibration_log.txt';
    $output_file = $operators_dir . DIRECTORY_SEPARATOR . 'calibration_output.txt';
    
    $log_content = file_exists($log_file) ? file_get_contents($log_file) : 'No hay archivo de log';
    $python_output = file_exists($output_file) ? file_get_contents($output_file) : '';
    
    echo json_encode([
        'progress' => get_calibration_progress(),
        'in_progress' => is_calibration_in_progress(),
        'log' => $log_content,
        'python_output' => $python_output
    ]);
    exit;
}

// Si se recibe una solicitud POST (botón presionado)
if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['operator_id'])) {
    $operator_id = $_POST['operator_id'];
    
    // Verificar si ya hay una calibración en curso
    if (is_calibration_in_progress()) {
        header('Content-Type: application/json');
        echo json_encode(['error' => 'Ya hay una calibración en curso']);
        exit;
    }
    
    // Ejecutar el script de calibración
    execute_calibration_script($operator_id);
    
    header('Content-Type: application/json');
    echo json_encode(['success' => true, 'message' => 'Calibración iniciada']);
    exit;
}

// Si no es una solicitud válida, devolver error
header('HTTP/1.1 400 Bad Request');
echo json_encode(['error' => 'Solicitud inválida']);
?>