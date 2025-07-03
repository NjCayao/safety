<?php
/**
 * Script para monitorear la carpeta de reportes y procesarlos automáticamente
 * 
 * Este script escanea la carpeta 'reports' en busca de nuevos archivos de imagen y texto,
 * extrae la información relevante y crea alertas en la base de datos.
 */

// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Configurar tiempo de ejecución máximo
set_time_limit(300);

// Definir constantes de rutas
if (!defined('BASE_PATH')) {
    define('BASE_PATH', dirname(dirname(__FILE__))); // Ruta base del proyecto
}

// Definir rutas específicas del script
define('REPORTS_PATH', BASE_PATH . '/../reports'); // Carpeta donde llegan los reportes
define('PROCESSED_PATH', BASE_PATH . '/../reports/processed'); // Carpeta para archivos procesados
define('FAILED_PATH', BASE_PATH . '/../reports/failed'); // Carpeta para archivos con errores
define('UPLOADS_PATH', BASE_PATH . '/uploads/alerts'); // Carpeta destino para las imágenes

// Incluir archivos necesarios
require_once BASE_PATH . '/config/config.php';
require_once BASE_PATH . '/config/database.php';

// Crear instancia de Database para obtener la conexión
$database = new Database();
$pdo = $database->getConnection();

// Verificar si la conexión fue establecida
if ($pdo === null) {
    die("Error fatal: No se pudo establecer la conexión a la base de datos: " . $database->getError());
}

// Función para registrar mensajes en la consola y la base de datos
function log_message($level, $message, $machine_id = null, $details = null) {
    global $pdo; // Usar la conexión global
    
    echo date('Y-m-d H:i:s') . " [$level] $message\n";
    
    try {
        $sql = "INSERT INTO system_logs (log_type, machine_id, message, details) VALUES (?, ?, ?, ?)";
        $stmt = $pdo->prepare($sql);
        $stmt->execute([$level, $machine_id, $message, $details]);
    } catch (Exception $e) {
        echo "Error al registrar en base de datos: " . $e->getMessage() . "\n";
    }
}

// Crear carpetas si no existen
if (!file_exists(PROCESSED_PATH)) {
    if (!mkdir(PROCESSED_PATH, 0755, true)) {
        log_message('error', "No se pudo crear la carpeta: " . PROCESSED_PATH);
    }
}
if (!file_exists(FAILED_PATH)) {
    if (!mkdir(FAILED_PATH, 0755, true)) {
        log_message('error', "No se pudo crear la carpeta: " . FAILED_PATH);
    }
}
if (!file_exists(UPLOADS_PATH)) {
    if (!mkdir(UPLOADS_PATH, 0755, true)) {
        log_message('error', "No se pudo crear la carpeta: " . UPLOADS_PATH);
    }
}

// Registrar inicio del proceso
log_message('info', 'Iniciando monitoreo de reportes', null, 'Script: monitor_reports.php');

// Variables para contabilizar resultados
$processedCount = 0;
$failedCount = 0;

// Verificar si la carpeta reports existe
if (!file_exists(REPORTS_PATH) || !is_dir(REPORTS_PATH)) {
    $message = "La carpeta de reportes no existe en: " . REPORTS_PATH;
    log_message('error', $message, null, 'Script: monitor_reports.php');
    exit("ERROR: $message\n");
}

echo "Buscando archivos en: " . REPORTS_PATH . "\n";

// Obtener todos los archivos de la carpeta reports (excepto subcarpetas processed y failed)
$files = new DirectoryIterator(REPORTS_PATH);

// Procesar los archivos
foreach ($files as $file) {
    // Saltear ".", "..", directorios y carpetas processed/failed
    if ($file->isDot() || $file->isDir() || 
        $file->getFilename() === 'processed' || 
        $file->getFilename() === 'failed') {
        continue;
    }
    
    $filename = $file->getFilename();
    $extension = strtolower($file->getExtension());
    
    echo "Procesando archivo: $filename ($extension)\n";
    
    // Solo procesamos archivos JPG y TXT
    if ($extension !== 'jpg' && $extension !== 'txt') {
        continue;
    }
    
    // Procesar archivos por pares (imagen + texto)
    $baseFilename = substr($filename, 0, strrpos($filename, '.'));
    
    // Verificar si es un archivo de imagen
    if ($extension === 'jpg') {
        $txtFile = REPORTS_PATH . '/' . $baseFilename . '.txt';
        
        // Verificar si existe el archivo TXT correspondiente
        if (!file_exists($txtFile)) {
            // Esperar un poco, puede que el TXT aún no haya terminado de escribirse
            sleep(2);
            if (!file_exists($txtFile)) {
                log_message('warning', "Archivo de texto no encontrado para $filename", null, 'El sistema esperará al siguiente ciclo');
                continue;
            }
        }
        
        // Intentar procesar el par de archivos (imagen + texto)
        try {
            // Extraer el DNI del nombre del archivo (primer segmento antes del guion bajo)
            if (preg_match('/^(\d+)_/', $baseFilename, $matches)) {
                $dni = $matches[1];
                
                // Extraer el tipo de alerta (segundo segmento después del primer guion bajo)
                $parts = explode('_', $baseFilename);
                $alertType = isset($parts[1]) ? strtolower($parts[1]) : 'other';
                
                // Validar que el tipo de alerta sea válido
                $validAlertTypes = ['fatigue', 'phone', 'smoking', 'unauthorized', 'yawn', 'distraction', 'behavior', 'other'];
                if (!in_array($alertType, $validAlertTypes)) {
                    $alertType = 'other';
                }
                
                // Extraer fecha y hora del nombre (formato: YYYYMMDD_HHMMSS)
                $dateTimeStr = isset($parts[2]) && isset($parts[3]) ? $parts[2] . '_' . $parts[3] : '';
                $timestamp = null;
                
                if (!empty($dateTimeStr) && strlen($dateTimeStr) >= 15) {
                    // Formatear fecha y hora
                    $year = substr($dateTimeStr, 0, 4);
                    $month = substr($dateTimeStr, 4, 2);
                    $day = substr($dateTimeStr, 6, 2);
                    $hour = substr($dateTimeStr, 9, 2);
                    $minute = substr($dateTimeStr, 11, 2);
                    $second = substr($dateTimeStr, 13, 2);
                    
                    $timestamp = "$year-$month-$day $hour:$minute:$second";
                } else {
                    // Usar la fecha de modificación del archivo
                    $timestamp = date('Y-m-d H:i:s', $file->getMTime());
                }
                
                // Leer el contenido del archivo TXT
                $txtContent = file_get_contents($txtFile);
                
                // Buscar el operador por DNI usando PDO
                $operatorQuery = "SELECT id, name FROM operators WHERE dni_number = ?";
                $stmt = $pdo->prepare($operatorQuery);
                $stmt->execute([$dni]);
                $operator = $stmt->fetch(PDO::FETCH_ASSOC);
                
                if (!$operator) {
                    throw new Exception("No se encontró un operador con DNI: $dni");
                }
                
                $operatorId = $operator['id'];
                $operatorName = $operator['name'];
                
                // Buscar la máquina asignada al operador usando PDO
                $machineQuery = "SELECT m.id, m.name 
                                FROM operator_machine om 
                                JOIN machines m ON om.machine_id = m.id 
                                WHERE om.operator_id = ? AND om.is_current = 1 
                                LIMIT 1";
                $stmt = $pdo->prepare($machineQuery);
                $stmt->execute([$operatorId]);
                $machine = $stmt->fetch(PDO::FETCH_ASSOC);
                
                $machineId = $machine ? $machine['id'] : null;
                
                // Crear directorio para el mes actual si no existe
                $yearMonth = date('Y-m', strtotime($timestamp));
                $uploadDir = UPLOADS_PATH . '/' . $yearMonth;
                
                if (!file_exists($uploadDir)) {
                    if (!mkdir($uploadDir, 0755, true)) {
                        throw new Exception("No se pudo crear el directorio: $uploadDir");
                    }
                }
                
                // Copiar imagen a la carpeta de uploads
                $newImageFilename = $baseFilename . '.jpg';
                $newImagePath = $uploadDir . '/' . $newImageFilename;
                $relativeImagePath = 'uploads/alerts/' . $yearMonth . '/' . $newImageFilename;
                
                echo "Copiando imagen a: " . $newImagePath . "\n";
                
                // Mover archivos a las carpetas correspondientes
                if (copy($file->getPathname(), BASE_PATH . '/' . $relativeImagePath)) {
                    // También guardamos el archivo de texto en la misma ubicación
                    $txtRelativePath = 'uploads/alerts/' . $yearMonth . '/' . $baseFilename . '.txt';
                    copy($txtFile, BASE_PATH . '/' . $txtRelativePath);
                    
                    // Ahora movemos los originales a la carpeta de procesados
                    if (!rename($file->getPathname(), PROCESSED_PATH . '/' . $filename)) {
                        log_message('warning', "No se pudo mover el archivo original a processed: $filename", null, "Pero la copia fue exitosa");
                    }
                    
                    if (!rename($txtFile, PROCESSED_PATH . '/' . basename($txtFile))) {
                        log_message('warning', "No se pudo mover el archivo de texto a processed: " . basename($txtFile), null, "Pero la copia fue exitosa");
                    }
                    
                    // Insertar alerta en la base de datos
                    $alertData = [
                        'operator_id' => $operatorId,
                        'machine_id' => $machineId,
                        'alert_type' => $alertType,
                        'timestamp' => $timestamp,
                        'image_path' => $relativeImagePath,
                        'details' => $txtContent,
                        'acknowledged' => 0
                    ];
                    
                    // Mostrar datos que se insertarán en la BD
                    echo "Datos a insertar en la BD:\n";
                    print_r($alertData);
                    
                    // Insertar en la base de datos usando PDO directamente
                    $columns = implode(', ', array_keys($alertData));
                    $placeholders = implode(', ', array_fill(0, count($alertData), '?'));
                    
                    $sql = "INSERT INTO alerts ($columns) VALUES ($placeholders)";
                    $stmt = $pdo->prepare($sql);
                    $result = $stmt->execute(array_values($alertData));
                    
                    $alertId = $pdo->lastInsertId();
                    
                    if ($alertId) {
                        // Incrementar contador de procesados
                        $processedCount++;
                        
                        // Registrar éxito
                        log_message('info', "Alerta creada para $operatorName (ID: $alertId)", 
                                  $machineId, "Tipo: $alertType, Archivo: $filename");
                    } else {
                        throw new Exception("Error al insertar la alerta en la base de datos: " . print_r($pdo->errorInfo(), true));
                    }
                } else {
                    throw new Exception("No se pudo copiar la imagen a la carpeta de destino: " . $newImagePath);
                }
            } else {
                throw new Exception("No se pudo extraer el DNI del nombre de archivo: $filename");
            }
        } catch (Exception $e) {
            // En caso de error, mover archivos a la carpeta de fallidos
            $failedCount++;
            
            // Intentar mover la imagen
            if (file_exists($file->getPathname())) {
                if (!rename($file->getPathname(), FAILED_PATH . '/' . $filename)) {
                    log_message('error', "No se pudo mover el archivo fallido: $filename", null, "Error adicional al mover a FAILED_PATH");
                }
            }
            
            // Intentar mover el archivo TXT si existe
            if (file_exists($txtFile)) {
                if (!rename($txtFile, FAILED_PATH . '/' . basename($txtFile))) {
                    log_message('error', "No se pudo mover el archivo de texto fallido: " . basename($txtFile), null, "Error adicional al mover a FAILED_PATH");
                }
            }
            
            // Registrar error
            log_message('error', "Error al procesar archivo: $filename", null, $e->getMessage());
        }
    }
}

// Registrar finalización del proceso
$message = "Monitoreo de reportes finalizado. Procesados: $processedCount. Fallidos: $failedCount";
log_message('info', $message, null, 'Script: monitor_reports.php');

echo "$message\n";
?>