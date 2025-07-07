<?php
/**
 * Script auxiliar: Exportar configuración de dispositivo
 * server/pages/devices/export_config.php
 */

// Verificar sesión
session_start();
if (!isset($_SESSION['user_id'])) {
    header('Location: ../../login.php');
    exit;
}

// Incluir dependencias
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/device_config.php';

// Obtener datos
$device_id = $_GET['device_id'] ?? null;
$format = $_GET['format'] ?? 'yaml';

if (!$device_id) {
    header('Location: config.php?error=ID de dispositivo requerido');
    exit;
}

try {
    // Verificar que el dispositivo existe
    $device = db_fetch_one("SELECT * FROM devices WHERE device_id = ?", [$device_id]);
    if (!$device) {
        header('Location: config.php?error=Dispositivo no encontrado');
        exit;
    }

    // Obtener configuración actual
    $config_data = DeviceConfigManager::getDeviceConfig($device_id);
    $config = $config_data['config'];

    // Agregar metadatos
    $export_data = [
        'metadata' => [
            'device_id' => $device_id,
            'device_name' => $device['name'],
            'device_type' => $device['device_type'],
            'config_version' => $config_data['version'],
            'exported_at' => date('Y-m-d H:i:s'),
            'exported_by' => $_SESSION['username'] ?? 'Usuario',
            'machine_id' => $device['machine_id']
        ],
        'config' => $config
    ];

    // Exportar según el formato solicitado
    switch ($format) {
        case 'json':
            exportAsJson($export_data, $device_id);
            break;
        case 'yaml':
            exportAsYaml($export_data, $device_id);
            break;
        case 'txt':
            exportAsText($export_data, $device_id);
            break;
        default:
            exportAsYaml($export_data, $device_id);
    }

} catch (Exception $e) {
    error_log("Error en export_config.php: " . $e->getMessage());
    header('Location: config.php?error=Error al exportar configuración');
    exit;
}

/**
 * Exportar como JSON
 */
function exportAsJson($data, $device_id) {
    $filename = "config_{$device_id}_" . date('Y-m-d_H-i-s') . ".json";
    
    header('Content-Type: application/json');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

/**
 * Exportar como YAML
 */
function exportAsYaml($data, $device_id) {
    $filename = "config_{$device_id}_" . date('Y-m-d_H-i-s') . ".yaml";
    
    header('Content-Type: application/x-yaml');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo generateYamlContent($data);
}

/**
 * Exportar como texto plano
 */
function exportAsText($data, $device_id) {
    $filename = "config_{$device_id}_" . date('Y-m-d_H-i-s') . ".txt";
    
    header('Content-Type: text/plain; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo generateTextContent($data);
}

/**
 * Generar contenido YAML
 */
function generateYamlContent($data) {
    $yaml = "# Configuración exportada del dispositivo\n";
    $yaml .= "# Generado el: " . date('Y-m-d H:i:s') . "\n\n";
    
    // Metadatos
    $yaml .= "# Información del dispositivo\n";
    $yaml .= "metadata:\n";
    foreach ($data['metadata'] as $key => $value) {
        $yaml .= "  {$key}: " . yamlValue($value) . "\n";
    }
    
    $yaml .= "\n# Configuración\n";
    $yaml .= "config:\n";
    $yaml .= arrayToYaml($data['config'], 1);
    
    return $yaml;
}

/**
 * Generar contenido de texto plano
 */
function generateTextContent($data) {
    $text = "CONFIGURACIÓN DEL DISPOSITIVO\n";
    $text .= str_repeat("=", 50) . "\n\n";
    
    // Metadatos
    $text .= "INFORMACIÓN DEL DISPOSITIVO:\n";
    $text .= str_repeat("-", 30) . "\n";
    foreach ($data['metadata'] as $key => $value) {
        $text .= sprintf("%-20s: %s\n", ucfirst(str_replace('_', ' ', $key)), $value);
    }
    
    $text .= "\n\nCONFIGURACIÓN:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= arrayToText($data['config'], 0);
    
    $text .= "\n\nNOTAS:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= "- Esta configuración puede ser importada en otro dispositivo del mismo tipo\n";
    $text .= "- Verificar que los valores sean apropiados para el hardware de destino\n";
    $text .= "- Realizar pruebas antes de aplicar en producción\n";
    
    return $text;
}

/**
 * Convertir array a YAML
 */
function arrayToYaml($array, $indent = 0) {
    $yaml = '';
    $spaces = str_repeat('  ', $indent);
    
    foreach ($array as $key => $value) {
        if (is_array($value)) {
            $yaml .= "{$spaces}{$key}:\n";
            $yaml .= arrayToYaml($value, $indent + 1);
        } else {
            $yaml .= "{$spaces}{$key}: " . yamlValue($value) . "\n";
        }
    }
    
    return $yaml;
}

/**
 * Convertir array a texto plano
 */
function arrayToText($array, $indent = 0) {
    $text = '';
    $spaces = str_repeat('  ', $indent);
    
    foreach ($array as $key => $value) {
        if (is_array($value)) {
            $text .= "{$spaces}" . strtoupper($key) . ":\n";
            $text .= arrayToText($value, $indent + 1);
        } else {
            $text .= sprintf("%s%-25s: %s\n", 
                $spaces, 
                ucfirst(str_replace('_', ' ', $key)), 
                $value
            );
        }
    }
    
    return $text;
}

/**
 * Formatear valor para YAML
 */
function yamlValue($value) {
    if (is_bool($value)) {
        return $value ? 'true' : 'false';
    } elseif (is_string($value)) {
        // Escapar strings si contienen caracteres especiales
        if (preg_match('/[:\[\]{}|>]/', $value) || is_numeric($value)) {
            return '"' . addslashes($value) . '"';
        }
        return $value;
    } else {
        return $value;
    }
}
?>