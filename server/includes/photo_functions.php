<?php
// Funciones para el manejo de fotos de operadores

/**
 * Crea el directorio para las fotos de un operador si no existe
 * @param string $operatorId ID del operador
 * @param string $dniNumber DNI del operador para nombrar la carpeta
 * @param string $name Nombre del operador
 * @param string $position Posición/cargo del operador
 * @param string $machine Máquina asignada (opcional)
 * @return array Resultado de la operación
 */
function create_operator_directory($operatorId, $dniNumber, $name, $position = '', $machine = '') {
    // Verificar que el DNI no esté vacío
    if (empty($dniNumber)) {
        return [
            'success' => false,
            'message' => 'El número de DNI es necesario para crear el directorio de fotos'
        ];
    }
    
    // Definir ruta raíz para imágenes - dentro de server
    $rootPath = __DIR__ . '/../operator-photo';
    
    // Verificar/crear directorio principal si no existe
    if (!file_exists($rootPath)) {
        if (!mkdir($rootPath, 0777, true)) {
            return [
                'success' => false,
                'message' => 'No se pudo crear el directorio principal para fotos: ' . $rootPath
            ];
        }
    }
    
    // Ruta para el directorio específico del operador
    $operatorPath = $rootPath . '/' . $dniNumber;
    
    // Crear directorio del operador si no existe
    if (!file_exists($operatorPath)) {
        if (!mkdir($operatorPath, 0777, true)) {
            return [
                'success' => false,
                'message' => 'No se pudo crear el directorio para las fotos del operador: ' . $operatorPath
            ];
        }
        
        // Crear archivo info.txt
        $infoContent = "Operador: $name\n";
        $infoContent .= "DNI: $dniNumber\n";
        $infoContent .= "ID: $operatorId\n";
        
        if (!empty($position)) {
            $infoContent .= "Rol: $position\n";
        }
        
        if (!empty($machine)) {
            $infoContent .= "Máquina asignada: $machine\n";
        }
        
        $infoContent .= "Fecha de creación: " . date('Y-m-d H:i:s') . "\n";
        
        // Guardar el archivo info.txt
        file_put_contents($operatorPath . '/info.txt', $infoContent);
    }
    
    return [
        'success' => true,
        'path' => $operatorPath
    ];
}

/**
 * Actualiza el archivo info.txt con la información más reciente
 * @param string $dniNumber DNI del operador
 * @param array $data Datos actualizados (name, position, machine, etc.)
 * @return bool Éxito de la operación
 */
function update_operator_info_file($dniNumber, $data) {
    $operatorPath = __DIR__ . '/../operator-photo/' . $dniNumber;
    
    if (!file_exists($operatorPath)) {
        return false;
    }
    
    $infoContent = "";
    
    if (isset($data['name'])) {
        $infoContent .= "Operador: " . $data['name'] . "\n";
    }
    
    $infoContent .= "DNI: $dniNumber\n";
    
    if (isset($data['id'])) {
        $infoContent .= "ID: " . $data['id'] . "\n";
    }
    
    if (isset($data['position']) && !empty($data['position'])) {
        $infoContent .= "Rol: " . $data['position'] . "\n";
    }
    
    if (isset($data['machine']) && !empty($data['machine'])) {
        $infoContent .= "Máquina asignada: " . $data['machine'] . "\n";
    }
    
    $infoContent .= "Última actualización: " . date('Y-m-d H:i:s') . "\n";
    
    return file_put_contents($operatorPath . '/info.txt', $infoContent) !== false;
}

/**
 * Valida y sube una foto de operador
 * @param array $file Archivo subido ($_FILES[campo])
 * @param string $operatorId ID del operador
 * @param string $dniNumber DNI del operador para la carpeta
 * @param string $type Tipo de foto (profile, face1, face2, face3)
 * @param string $name Nombre del operador (para info.txt)
 * @param string $position Posición/rol del operador (para info.txt)
 * @param string $machine Máquina asignada (para info.txt)
 * @return array Resultado de la operación
 */
function upload_operator_photo($file, $operatorId, $dniNumber, $type, $name = '', $position = '', $machine = '') {
    // Verificar que el archivo existe
    if (empty($file) || !is_array($file) || empty($file['tmp_name'])) {
        return [
            'success' => false,
            'message' => 'No se proporcionó un archivo válido'
        ];
    }
    
    // Verificar errores de carga
    if ($file['error'] !== UPLOAD_ERR_OK) {
        $errorMessages = [
            UPLOAD_ERR_INI_SIZE => 'El archivo excede el tamaño máximo permitido por el servidor',
            UPLOAD_ERR_FORM_SIZE => 'El archivo excede el tamaño máximo permitido por el formulario',
            UPLOAD_ERR_PARTIAL => 'El archivo fue subido parcialmente',
            UPLOAD_ERR_NO_FILE => 'No se subió ningún archivo',
            UPLOAD_ERR_NO_TMP_DIR => 'Falta la carpeta temporal del servidor',
            UPLOAD_ERR_CANT_WRITE => 'Error al escribir el archivo en el servidor',
            UPLOAD_ERR_EXTENSION => 'Una extensión de PHP detuvo la carga'
        ];
        
        $errorMessage = isset($errorMessages[$file['error']]) 
            ? $errorMessages[$file['error']] 
            : 'Error desconocido al subir el archivo (código: ' . $file['error'] . ')';
        
        return [
            'success' => false,
            'message' => $errorMessage
        ];
    }
    
    // Verificar que es una imagen válida
    $extension = strtolower(pathinfo($file['name'], PATHINFO_EXTENSION));
    
    if (!in_array($extension, ['jpg', 'jpeg', 'png'])) {
        return [
            'success' => false,
            'message' => 'Tipo de archivo no permitido. Solo se aceptan imágenes JPG o PNG'
        ];
    }
    
    // Crear el directorio para el operador si no existe
    $dirResult = create_operator_directory($operatorId, $dniNumber, $name, $position, $machine);
    
    if (!$dirResult['success']) {
        return $dirResult; // Devolver el error de creación de directorio
    }
    
    // Crear nombre de archivo único
    $fileName = $type . '_' . date('Ymd_His') . '.' . $extension;
    
    // Ruta física completa donde se guardará el archivo
    $filePath = $dirResult['path'] . '/' . $fileName;
    
    // Ruta para guardar en la base de datos (URL relativa)
    $dbPath = '/safety_system/server/operator-photo/' . $dniNumber . '/' . $fileName;
    
    // Intentar mover el archivo subido a su ubicación final
    if (!move_uploaded_file($file['tmp_name'], $filePath)) {
        return [
            'success' => false,
            'message' => 'Error al mover el archivo subido. Verifique permisos en: ' . $filePath
        ];
    }
    
    // Cambiar permisos del archivo para asegurar que sea legible
    chmod($filePath, 0644);
    
    // Devolver éxito y la ruta para guardar en la base de datos
    return [
        'success' => true,
        'path' => $dbPath,
        'filename' => $fileName
    ];
}

/**
 * Obtener la máquina asignada a un operador de forma segura
 * @param string $operatorId ID del operador
 * @return string Nombre de la máquina o cadena vacía si no tiene asignación
 */
function get_operator_machine($operatorId) {
    $machine = '';
    
    try {
        $result = db_fetch_one(
            "SELECT m.name FROM operator_machine om 
             JOIN machines m ON om.machine_id = m.id 
             WHERE om.operator_id = ? AND om.is_current = 1 
             LIMIT 1",
            [$operatorId]
        );
        
        if ($result && isset($result['name'])) {
            $machine = $result['name'];
        }
    } catch (Exception $e) {
        // Si hay error, simplemente devolver cadena vacía
    }
    
    return $machine;
}
?>