<?php
/**
 * API para operadores (usado por las Raspberry Pi)
 */

// Encabezados para API
header('Content-Type: application/json');
header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST');
header('Access-Control-Allow-Headers: Access-Control-Allow-Headers, Content-Type, Access-Control-Allow-Methods, Authorization, X-Requested-With');

// Incluir archivos necesarios
require_once '../config/database.php';
require_once '../includes/functions.php';

// En un sistema real, verificarías el token de acceso aquí
// ...

// Obtener método de solicitud
$method = $_SERVER['REQUEST_METHOD'];

switch ($method) {
    case 'GET':
        // Verificar si se busca un operador específico
        if (isset($_GET['identification'])) {
            $identification = $_GET['identification'];
            
            // Buscar operador por número de identificación
            $operator = db_fetch_one(
                "SELECT id, name, identification_number, position, photo_path, status 
                 FROM operators 
                 WHERE identification_number = ? AND status = 'active'",
                [$identification]
            );
            
            if ($operator) {
                // Agregar URL completa a la foto
                if ($operator['photo_path']) {
                    $operator['photo_url'] = 'http://' . $_SERVER['HTTP_HOST'] . '/safety_system/server/' . $operator['photo_path'];
                } else {
                    $operator['photo_url'] = null;
                }
                
                echo json_encode([
                    'success' => true,
                    'operator' => $operator
                ]);
            } else {
                echo json_encode([
                    'success' => false,
                    'message' => 'Operador no encontrado o inactivo'
                ]);
            }
        } else {
            // Obtener todos los operadores activos
            $operators = db_fetch_all(
                "SELECT id, name, identification_number, position, status 
                 FROM operators 
                 WHERE status = 'active' 
                 ORDER BY name"
            );
            
            echo json_encode([
                'success' => true,
                'operators' => $operators
            ]);
        }
        break;
        
    case 'POST':
        // En un sistema real, esta parte sería para registrar datos faciales del operador
        // u otra información desde la Raspberry Pi
        
        // Obtener datos enviados
        $data = json_decode(file_get_contents('php://input'), true);
        
        // Si no hay datos JSON, intentar con POST
        if (!$data) {
            $data = $_POST;
        }
        
        // Verificar datos mínimos necesarios
        if (!isset($data['operator_id']) || !isset($data['face_data'])) {
            echo json_encode([
                'success' => false,
                'message' => 'Faltan parámetros requeridos'
            ]);
            exit;
        }
        
        $operatorId = $data['operator_id'];
        $faceData = $data['face_data'];
        
        // Guardar datos faciales (en un escenario real, esto sería más complejo)
        // Por ejemplo, guardarías un archivo JSON con los datos faciales
        $filename = 'face_data_' . $operatorId . '.json';
        $directory = '../uploads/face_data';
        
        // Crear directorio si no existe
        if (!file_exists($directory)) {
            mkdir($directory, 0777, true);
        }
        
        $filePath = $directory . '/' . $filename;
        
        if (file_put_contents($filePath, $faceData)) {
            // Actualizar referencia en la base de datos
            $updated = db_update(
                'operators',
                ['face_data_path' => 'uploads/face_data/' . $filename],
                'id = ?',
                [$operatorId]
            );
            
            if ($updated) {
                echo json_encode([
                    'success' => true,
                    'message' => 'Datos faciales guardados correctamente'
                ]);
            } else {
                echo json_encode([
                    'success' => false,
                    'message' => 'Error al actualizar la base de datos'
                ]);
            }
        } else {
            echo json_encode([
                'success' => false,
                'message' => 'Error al guardar los datos faciales'
            ]);
        }
        break;
        
    default:
        echo json_encode([
            'success' => false,
            'message' => 'Método no permitido'
        ]);
        break;
}