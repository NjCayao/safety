<?php
/**
 * API para obtener imágenes de reportes en tiempo real
 * 
 * Este endpoint escanea la carpeta de reportes y devuelve las imágenes
 * que aún no han sido procesadas para mostrarlas en tiempo real
 */

// Encabezados para permitir AJAX
header('Content-Type: application/json');
header('Cache-Control: no-cache, must-revalidate');

// Definir constantes de rutas
define('BASE_PATH', dirname(dirname(dirname(__FILE__)))); // Ruta base del proyecto
define('REPORTS_PATH', BASE_PATH . '/../reports'); // Carpeta donde llegan los reportes

// Incluir archivos necesarios
require_once BASE_PATH . '/config/config.php';
require_once BASE_PATH . '/config/database.php';

// Parámetros de la solicitud
$limit = isset($_GET['limit']) ? (int)$_GET['limit'] : 5;

// Validar parámetros
if ($limit > 20) {
    $limit = 20; // Limitar a 20 registros como máximo
}

// Array para almacenar las imágenes encontradas
$images = [];

// Verificar si la carpeta existe
if (!file_exists(REPORTS_PATH) || !is_dir(REPORTS_PATH)) {
    // La carpeta no existe, devolver respuesta vacía
    echo json_encode([
        'status' => 'error',
        'message' => 'La carpeta de reportes no existe: ' . REPORTS_PATH,
        'images' => []
    ]);
    exit;
}

try {
    // Obtener todos los archivos de la carpeta reports
    $files = new DirectoryIterator(REPORTS_PATH);
    
    // Array para almacenar los archivos encontrados
    $foundFiles = [];
    
    // Primero, recopilar todos los archivos para procesarlos después
    foreach ($files as $file) {
        // Saltear ".", "..", directorios y carpetas processed/failed
        if ($file->isDot() || $file->isDir() || 
            $file->getFilename() === 'processed' || 
            $file->getFilename() === 'failed') {
            continue;
        }
        
        $filename = $file->getFilename();
        $extension = strtolower($file->getExtension());
        
        // Solo procesamos archivos JPG y TXT
        if ($extension !== 'jpg' && $extension !== 'txt') {
            continue;
        }
        
        // Guardar la información del archivo
        $foundFiles[] = [
            'path' => $file->getPathname(),
            'filename' => $filename,
            'extension' => $extension,
            'mtime' => $file->getMTime()
        ];
    }
    
    // Ordenar los archivos por fecha de modificación (más recientes primero)
    usort($foundFiles, function($a, $b) {
        return $b['mtime'] - $a['mtime'];
    });
    
    // Procesar los archivos JPG
    $processedCount = 0;
    foreach ($foundFiles as $fileInfo) {
        // Solo procesamos imágenes JPG
        if ($fileInfo['extension'] !== 'jpg') {
            continue;
        }
        
        $filename = $fileInfo['filename'];
        $baseFilename = substr($filename, 0, strrpos($filename, '.'));
        $txtFilename = $baseFilename . '.txt';
        $txtPath = REPORTS_PATH . '/' . $txtFilename;
        
        // Verificar si existe el archivo TXT correspondiente
        $hasTxtFile = file_exists($txtPath);
        $txtContent = '';
        
        if ($hasTxtFile) {
            $txtContent = file_get_contents($txtPath);
        }
        
        // Extraer información del nombre del archivo
        $alertInfo = [
            'dni' => '',
            'type' => 'other',
            'timestamp' => date('Y-m-d H:i:s', $fileInfo['mtime']),
            'operator_id' => null,
            'operator_name' => 'Desconocido'
        ];
        
        // Extraer el DNI del nombre del archivo (primer segmento antes del guion bajo)
        if (preg_match('/^(\d+)_/', $baseFilename, $matches)) {
            $alertInfo['dni'] = $matches[1];
            
            // Extraer el tipo de alerta (segundo segmento después del primer guion bajo)
            $parts = explode('_', $baseFilename);
            if (isset($parts[1])) {
                $alertInfo['type'] = strtolower($parts[1]);
            }
            
            // Extraer fecha y hora del nombre (formato: YYYYMMDD_HHMMSS)
            if (isset($parts[2]) && isset($parts[3]) && strlen($parts[2]) >= 8 && strlen($parts[3]) >= 6) {
                $year = substr($parts[2], 0, 4);
                $month = substr($parts[2], 4, 2);
                $day = substr($parts[2], 6, 2);
                $hour = substr($parts[3], 0, 2);
                $minute = substr($parts[3], 2, 2);
                $second = substr($parts[3], 4, 2);
                
                $alertInfo['timestamp'] = "$year-$month-$day $hour:$minute:$second";
            }
        }
        
        // Buscar el operador por DNI
        if (!empty($alertInfo['dni'])) {
            try {
                $operatorQuery = "SELECT id, name FROM operators WHERE dni_number = ?";
                $stmt = $db->prepare($operatorQuery);
                $stmt->execute([$alertInfo['dni']]);
                $operator = $stmt->fetch(PDO::FETCH_ASSOC);
                
                if ($operator) {
                    $alertInfo['operator_id'] = $operator['id'];
                    $alertInfo['operator_name'] = $operator['name'];
                }
            } catch (Exception $e) {
                // Si hay error en la consulta, continuar con los valores por defecto
            }
        }
        
        // Obtener máquina asignada al operador (si está disponible)
        $machineInfo = [
            'id' => null,
            'name' => 'No asignada'
        ];
        
        if ($alertInfo['operator_id']) {
            try {
                $machineQuery = "SELECT m.id, m.name 
                                FROM operator_machine om 
                                JOIN machines m ON om.machine_id = m.id 
                                WHERE om.operator_id = ? AND om.is_current = 1 
                                LIMIT 1";
                $stmt = $db->prepare($machineQuery);
                $stmt->execute([$alertInfo['operator_id']]);
                $machine = $stmt->fetch(PDO::FETCH_ASSOC);
                
                if ($machine) {
                    $machineInfo['id'] = $machine['id'];
                    $machineInfo['name'] = $machine['name'];
                }
            } catch (Exception $e) {
                // Si hay error en la consulta, continuar con los valores por defecto
            }
        }
        
        // Definir etiquetas y clases para los tipos de alerta
        $alertTypeInfo = getAlertTypeInfo($alertInfo['type']);
        
        // Construir URL relativa para la imagen
        $imageUrl = '../../../reports/' . $filename;
        
        // Añadir la imagen al resultado
        $images[] = [
            'filename' => $filename,
            'base_filename' => $baseFilename,
            'image_url' => $imageUrl,
            'timestamp' => $alertInfo['timestamp'],
            'formatted_time' => date('d/m/Y H:i', strtotime($alertInfo['timestamp'])),
            'operator' => [
                'id' => $alertInfo['operator_id'],
                'name' => $alertInfo['operator_name'],
                'dni' => $alertInfo['dni']
            ],
            'machine' => $machineInfo,
            'alert_type' => $alertInfo['type'],
            'alert_type_label' => $alertTypeInfo['label'],
            'alert_type_class' => $alertTypeInfo['class'],
            'alert_type_icon' => $alertTypeInfo['icon'],
            'has_txt_file' => $hasTxtFile,
            'txt_content' => $txtContent
        ];
        
        $processedCount++;
        if ($processedCount >= $limit) {
            break;
        }
    }
    
    // Devolver respuesta JSON
    echo json_encode([
        'status' => 'success',
        'timestamp' => date('Y-m-d H:i:s'),
        'count' => count($images),
        'images' => $images
    ]);
    
} catch (Exception $e) {
    // En caso de error, devolver respuesta con error
    echo json_encode([
        'status' => 'error',
        'message' => $e->getMessage(),
        'images' => []
    ]);
}

/**
 * Obtiene información de estilo y etiqueta para un tipo de alerta
 */
function getAlertTypeInfo($type) {
    // Mapeo de tipos de alerta a etiquetas, clases y íconos
    $alertTypeInfo = [
        'fatigue' => [
            'label' => 'Fatiga',
            'class' => 'danger',
            'icon' => 'fas fa-bed'
        ],
        'yawn' => [
            'label' => 'Bostezo',
            'class' => 'warning',
            'icon' => 'fas fa-tired'
        ],
        'phone' => [
            'label' => 'Uso de Teléfono',
            'class' => 'info',
            'icon' => 'fas fa-mobile-alt'
        ],
        'smoking' => [
            'label' => 'Fumando',
            'class' => 'secondary',
            'icon' => 'fas fa-smoking'
        ],
        'distraction' => [
            'label' => 'Distracción',
            'class' => 'primary',
            'icon' => 'fas fa-eye-slash'
        ],
        'unauthorized' => [
            'label' => 'Operador No Autorizado',
            'class' => 'dark',
            'icon' => 'fas fa-user-slash'
        ],
        'behavior' => [
            'label' => 'Comportamiento Inadecuado',
            'class' => 'info',
            'icon' => 'fas fa-exclamation-circle'
        ],
        'other' => [
            'label' => 'Otra Alerta',
            'class' => 'light',
            'icon' => 'fas fa-question-circle'
        ]
    ];
    
    // Devolver información del tipo de alerta, o la información por defecto si no existe
    return isset($alertTypeInfo[$type]) ? $alertTypeInfo[$type] : $alertTypeInfo['other'];
}
?>