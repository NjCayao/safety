<?php
/**
 * Script auxiliar: Exportar configuración específica del historial
 * server/pages/devices/export_config_history.php
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

// Obtener datos
$history_id = $_GET['history_id'] ?? null;
$format = $_GET['format'] ?? 'yaml';

if (!$history_id) {
    header('Location: history.php?error=ID de historial requerido');
    exit;
}

try {
    // Obtener registro de historial específico
    $history_record = db_fetch_one("
        SELECT dch.*, 
               d.device_id, 
               d.name as device_name,
               d.device_type,
               d.location as device_location,
               m.name as machine_name,
               m.location as machine_location,
               u.username as changed_by_name,
               u.name as changed_by_full_name
        FROM device_config_history dch
        LEFT JOIN devices d ON dch.device_id = d.device_id
        LEFT JOIN machines m ON d.machine_id = m.id
        LEFT JOIN users u ON dch.changed_by = u.id
        WHERE dch.id = ?
    ", [$history_id]);

    if (!$history_record) {
        header('Location: history.php?error=Registro de historial no encontrado');
        exit;
    }

    // Obtener configuración
    $config = json_decode($history_record['config_after'], true);
    $config_before = $history_record['config_before'] ? json_decode($history_record['config_before'], true) : null;

    if (!$config) {
        header('Location: history.php?error=Configuración inválida');
        exit;
    }

    // Agregar metadatos completos
    $export_data = [
        'metadata' => [
            'history_id' => $history_record['id'],
            'device_id' => $history_record['device_id'],
            'device_name' => $history_record['device_name'],
            'device_type' => $history_record['device_type'],
            'device_location' => $history_record['device_location'],
            'machine_name' => $history_record['machine_name'],
            'machine_location' => $history_record['machine_location'],
            'change_type' => $history_record['change_type'],
            'change_summary' => $history_record['changes_summary'],
            'changed_by' => $history_record['changed_by_full_name'] ?: $history_record['changed_by_name'],
            'created_at' => $history_record['created_at'],
            'applied_successfully' => $history_record['applied_successfully'],
            'applied_at' => $history_record['applied_at'],
            'error_message' => $history_record['error_message'],
            'exported_at' => date('Y-m-d H:i:s'),
            'exported_by' => $_SESSION['username'] ?? 'Usuario'
        ],
        'config_before' => $config_before,
        'config_after' => $config
    ];

    // Exportar según el formato solicitado
    switch ($format) {
        case 'json':
            exportAsJson($export_data, $history_record);
            break;
        case 'yaml':
            exportAsYaml($export_data, $history_record);
            break;
        case 'txt':
            exportAsText($export_data, $history_record);
            break;
        default:
            exportAsYaml($export_data, $history_record);
    }

} catch (Exception $e) {
    error_log("Error en export_config_history.php: " . $e->getMessage());
    header('Location: history.php?error=Error al exportar configuración');
    exit;
}

/**
 * Exportar como JSON
 */
function exportAsJson($data, $history_record) {
    $filename = "config_history_{$history_record['device_id']}_" . $history_record['id'] . "_" . date('Y-m-d_H-i-s') . ".json";
    
    header('Content-Type: application/json');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

/**
 * Exportar como YAML
 */
function exportAsYaml($data, $history_record) {
    $filename = "config_history_{$history_record['device_id']}_" . $history_record['id'] . "_" . date('Y-m-d_H-i-s') . ".yaml";
    
    header('Content-Type: application/x-yaml');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo generateYamlContent($data, $history_record);
}

/**
 * Exportar como texto plano
 */
function exportAsText($data, $history_record) {
    $filename = "config_history_{$history_record['device_id']}_" . $history_record['id'] . "_" . date('Y-m-d_H-i-s') . ".txt";
    
    header('Content-Type: text/plain; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo generateTextContent($data, $history_record);
}

/**
 * Generar contenido YAML
 */
function generateYamlContent($data, $history_record) {
    $yaml = "# Configuración del historial ID: {$history_record['id']}\n";
    $yaml .= "# Exportado el: " . date('Y-m-d H:i:s') . "\n\n";
    
    // Metadatos
    $yaml .= "# Información del registro de historial\n";
    $yaml .= "metadata:\n";
    foreach ($data['metadata'] as $key => $value) {
        if ($value !== null) {
            $yaml .= "  {$key}: " . yamlValue($value) . "\n";
        }
    }
    
    // Configuración anterior (si existe)
    if ($data['config_before']) {
        $yaml .= "\n# Configuración anterior\n";
        $yaml .= "config_before:\n";
        $yaml .= arrayToYaml($data['config_before'], 1);
    }
    
    // Nueva configuración
    $yaml .= "\n# Nueva configuración\n";
    $yaml .= "config_after:\n";
    $yaml .= arrayToYaml($data['config_after'], 1);
    
    return $yaml;
}

/**
 * Generar contenido de texto plano
 */
function generateTextContent($data, $history_record) {
    $text = "HISTORIAL DE CONFIGURACIÓN - REGISTRO #{$history_record['id']}\n";
    $text .= str_repeat("=", 60) . "\n\n";
    
    // Información del registro
    $text .= "INFORMACIÓN DEL REGISTRO:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= sprintf("%-20s: %s\n", "ID Historial", $data['metadata']['history_id']);
    $text .= sprintf("%-20s: %s\n", "Dispositivo", $data['metadata']['device_id']);
    $text .= sprintf("%-20s: %s\n", "Nombre", $data['metadata']['device_name'] ?: 'N/A');
    $text .= sprintf("%-20s: %s\n", "Tipo", $data['metadata']['device_type'] ?: 'N/A');
    $text .= sprintf("%-20s: %s\n", "Máquina", $data['metadata']['machine_name'] ?: 'N/A');
    $text .= sprintf("%-20s: %s\n", "Tipo de cambio", ucfirst($data['metadata']['change_type']));
    $text .= sprintf("%-20s: %s\n", "Modificado por", $data['metadata']['changed_by'] ?: 'Sistema');
    $text .= sprintf("%-20s: %s\n", "Fecha", $data['metadata']['created_at']);
    
    if ($data['metadata']['change_summary']) {
        $text .= sprintf("%-20s: %s\n", "Resumen", $data['metadata']['change_summary']);
    }
    
    // Estado de aplicación
    $status = $data['metadata']['applied_successfully'];
    if ($status === null) {
        $text .= sprintf("%-20s: %s\n", "Estado", "Pendiente de aplicar");
    } elseif ($status) {
        $text .= sprintf("%-20s: %s\n", "Estado", "Aplicado exitosamente");
        $text .= sprintf("%-20s: %s\n", "Aplicado el", $data['metadata']['applied_at']);
    } else {
        $text .= sprintf("%-20s: %s\n", "Estado", "Error al aplicar");
        if ($data['metadata']['error_message']) {
            $text .= sprintf("%-20s: %s\n", "Error", $data['metadata']['error_message']);
        }
    }
    
    // Configuración anterior
    if ($data['config_before']) {
        $text .= "\n\nCONFIGURACIÓN ANTERIOR:\n";
        $text .= str_repeat("-", 30) . "\n";
        $text .= arrayToText($data['config_before'], 0);
    }
    
    // Nueva configuración
    $text .= "\n\nNUEVA CONFIGURACIÓN:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= arrayToText($data['config_after'], 0);
    
    // Información de exportación
    $text .= "\n\nINFORMACIÓN DE EXPORTACIÓN:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= sprintf("%-20s: %s\n", "Exportado el", $data['metadata']['exported_at']);
    $text .= sprintf("%-20s: %s\n", "Exportado por", $data['metadata']['exported_by']);
    
    $text .= "\n\nNOTAS:\n";
    $text .= str_repeat("-", 30) . "\n";
    $text .= "- Esta configuración puede ser aplicada manualmente a otros dispositivos\n";
    $text .= "- Verificar compatibilidad de hardware antes de aplicar\n";
    $text .= "- Realizar pruebas en entorno controlado antes de producción\n";
    
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
                formatValueForText($value)
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
        if (preg_match('/[:\[\]{}|>"\']/', $value) || is_numeric($value)) {
            return '"' . addslashes($value) . '"';
        }
        return $value;
    } elseif ($value === null) {
        return 'null';
    } else {
        return $value;
    }
}

/**
 * Formatear valor para texto
 */
function formatValueForText($value) {
    if (is_bool($value)) {
        return $value ? 'Sí' : 'No';
    } elseif ($value === null) {
        return 'N/A';
    } else {
        return $value;
    }
}
?>