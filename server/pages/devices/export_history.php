<?php
/**
 * Script auxiliar: Exportar historial de configuraciones
 * server/pages/devices/export_history.php
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

// Obtener parámetros de filtro (mismos que history.php)
$device_filter = $_GET['device'] ?? '';
$user_filter = $_GET['user'] ?? '';
$change_type_filter = $_GET['change_type'] ?? '';
$date_from = $_GET['date_from'] ?? '';
$date_to = $_GET['date_to'] ?? '';
$format = $_GET['format'] ?? 'excel'; // excel, csv, json

try {
    // Construir consulta de historial con filtros
    $where_conditions = [];
    $params = [];

    if ($device_filter) {
        $where_conditions[] = "(d.device_id LIKE ? OR d.name LIKE ?)";
        $params[] = "%$device_filter%";
        $params[] = "%$device_filter%";
    }

    if ($user_filter) {
        $where_conditions[] = "u.username LIKE ?";
        $params[] = "%$user_filter%";
    }

    if ($change_type_filter) {
        $where_conditions[] = "dch.change_type = ?";
        $params[] = $change_type_filter;
    }

    if ($date_from) {
        $where_conditions[] = "DATE(dch.created_at) >= ?";
        $params[] = $date_from;
    }

    if ($date_to) {
        $where_conditions[] = "DATE(dch.created_at) <= ?";
        $params[] = $date_to;
    }

    $where_clause = !empty($where_conditions) ? 'WHERE ' . implode(' AND ', $where_conditions) : '';

    // Obtener historial completo (sin límite para exportación)
    $history_query = "
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
        $where_clause
        ORDER BY dch.created_at DESC
    ";

    $history = db_fetch_all($history_query, $params);

    if (empty($history)) {
        header('Content-Type: text/plain');
        echo "No hay datos para exportar con los filtros aplicados.";
        exit;
    }

    // Exportar según formato solicitado
    switch ($format) {
        case 'csv':
            exportAsCSV($history);
            break;
        case 'json':
            exportAsJSON($history);
            break;
        case 'excel':
        default:
            exportAsExcel($history);
            break;
    }

} catch (Exception $e) {
    error_log("Error en export_history.php: " . $e->getMessage());
    
    header('Content-Type: text/plain');
    echo "Error al exportar historial: " . $e->getMessage();
    exit;
}

/**
 * Exportar como CSV
 */
function exportAsCSV($history) {
    $filename = "historial_configuraciones_" . date('Y-m-d_H-i-s') . ".csv";
    
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    // Crear salida CSV
    $output = fopen('php://output', 'w');
    
    // BOM para UTF-8 (para Excel)
    fputs($output, "\xEF\xBB\xBF");
    
    // Headers
    fputcsv($output, [
        'ID',
        'Fecha/Hora',
        'Dispositivo ID',
        'Nombre Dispositivo',
        'Tipo Dispositivo',
        'Máquina',
        'Tipo de Cambio',
        'Usuario',
        'Resumen',
        'Estado Aplicación',
        'Fecha Aplicación',
        'Error'
    ]);
    
    // Datos
    foreach ($history as $record) {
        $status = getApplicationStatusText($record['applied_successfully']);
        
        fputcsv($output, [
            $record['id'],
            $record['created_at'],
            $record['device_id'],
            $record['device_name'] ?: '',
            $record['device_type'] ?: '',
            $record['machine_name'] ?: '',
            ucfirst($record['change_type']),
            $record['changed_by_full_name'] ?: $record['changed_by_name'] ?: 'Sistema',
            $record['changes_summary'] ?: '',
            $status,
            $record['applied_at'] ?: '',
            $record['error_message'] ?: ''
        ]);
    }
    
    fclose($output);
}

/**
 * Exportar como JSON
 */
function exportAsJSON($history) {
    $filename = "historial_configuraciones_" . date('Y-m-d_H-i-s') . ".json";
    
    header('Content-Type: application/json; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    // Preparar datos para JSON
    $export_data = [
        'metadata' => [
            'exported_at' => date('Y-m-d H:i:s'),
            'exported_by' => $_SESSION['username'] ?? 'Usuario',
            'total_records' => count($history),
            'filters_applied' => array_filter([
                'device' => $_GET['device'] ?? null,
                'user' => $_GET['user'] ?? null,
                'change_type' => $_GET['change_type'] ?? null,
                'date_from' => $_GET['date_from'] ?? null,
                'date_to' => $_GET['date_to'] ?? null
            ])
        ],
        'history' => array_map(function($record) {
            return [
                'id' => $record['id'],
                'created_at' => $record['created_at'],
                'device' => [
                    'device_id' => $record['device_id'],
                    'name' => $record['device_name'],
                    'type' => $record['device_type'],
                    'location' => $record['device_location']
                ],
                'machine' => [
                    'name' => $record['machine_name'],
                    'location' => $record['machine_location']
                ],
                'change' => [
                    'type' => $record['change_type'],
                    'summary' => $record['changes_summary'],
                    'changed_by' => $record['changed_by_full_name'] ?: $record['changed_by_name']
                ],
                'application' => [
                    'status' => getApplicationStatusText($record['applied_successfully']),
                    'applied_at' => $record['applied_at'],
                    'error_message' => $record['error_message']
                ],
                'config_before' => $record['config_before'] ? json_decode($record['config_before'], true) : null,
                'config_after' => $record['config_after'] ? json_decode($record['config_after'], true) : null
            ];
        }, $history)
    ];
    
    echo json_encode($export_data, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE);
}

/**
 * Exportar como Excel (HTML table que Excel puede interpretar)
 */
function exportAsExcel($history) {
    $filename = "historial_configuraciones_" . date('Y-m-d_H-i-s') . ".xls";
    
    header('Content-Type: application/vnd.ms-excel; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $filename . '"');
    header('Cache-Control: no-cache, must-revalidate');
    header('Pragma: no-cache');
    
    echo '<?xml version="1.0" encoding="UTF-8"?>' . "\n";
    echo '<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:x="urn:schemas-microsoft-com:office:excel" xmlns="http://www.w3.org/TR/REC-html40">' . "\n";
    echo '<head><meta charset="UTF-8"></head>' . "\n";
    echo '<body>' . "\n";
    
    echo '<table border="1">' . "\n";
    
    // Headers
    echo '<tr style="background-color: #4472C4; color: white; font-weight: bold;">' . "\n";
    echo '<th>ID</th>';
    echo '<th>Fecha/Hora</th>';
    echo '<th>Dispositivo ID</th>';
    echo '<th>Nombre Dispositivo</th>';
    echo '<th>Tipo Dispositivo</th>';
    echo '<th>Ubicación Dispositivo</th>';
    echo '<th>Máquina</th>';
    echo '<th>Ubicación Máquina</th>';
    echo '<th>Tipo de Cambio</th>';
    echo '<th>Usuario</th>';
    echo '<th>Resumen</th>';
    echo '<th>Estado Aplicación</th>';
    echo '<th>Fecha Aplicación</th>';
    echo '<th>Error</th>';
    echo '</tr>' . "\n";
    
    // Datos
    foreach ($history as $record) {
        $status = getApplicationStatusText($record['applied_successfully']);
        $status_color = getStatusColor($record['applied_successfully']);
        
        echo '<tr>' . "\n";
        echo '<td>' . htmlspecialchars($record['id']) . '</td>';
        echo '<td>' . htmlspecialchars($record['created_at']) . '</td>';
        echo '<td>' . htmlspecialchars($record['device_id']) . '</td>';
        echo '<td>' . htmlspecialchars($record['device_name'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars($record['device_type'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars($record['device_location'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars($record['machine_name'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars($record['machine_location'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars(ucfirst($record['change_type'])) . '</td>';
        echo '<td>' . htmlspecialchars($record['changed_by_full_name'] ?: $record['changed_by_name'] ?: 'Sistema') . '</td>';
        echo '<td>' . htmlspecialchars($record['changes_summary'] ?: '') . '</td>';
        echo '<td style="background-color: ' . $status_color . ';">' . htmlspecialchars($status) . '</td>';
        echo '<td>' . htmlspecialchars($record['applied_at'] ?: '') . '</td>';
        echo '<td>' . htmlspecialchars($record['error_message'] ?: '') . '</td>';
        echo '</tr>' . "\n";
    }
    
    echo '</table>' . "\n";
    
    // Agregar información adicional
    echo '<br><br>' . "\n";
    echo '<table>' . "\n";
    echo '<tr><td><strong>Exportado el:</strong></td><td>' . date('d/m/Y H:i:s') . '</td></tr>' . "\n";
    echo '<tr><td><strong>Exportado por:</strong></td><td>' . htmlspecialchars($_SESSION['username'] ?? 'Usuario') . '</td></tr>' . "\n";
    echo '<tr><td><strong>Total de registros:</strong></td><td>' . count($history) . '</td></tr>' . "\n";
    
    // Mostrar filtros aplicados
    $filters = array_filter([
        'Dispositivo' => $_GET['device'] ?? null,
        'Usuario' => $_GET['user'] ?? null,
        'Tipo de cambio' => $_GET['change_type'] ?? null,
        'Fecha desde' => $_GET['date_from'] ?? null,
        'Fecha hasta' => $_GET['date_to'] ?? null
    ]);
    
    if (!empty($filters)) {
        echo '<tr><td colspan="2"><strong>Filtros aplicados:</strong></td></tr>' . "\n";
        foreach ($filters as $label => $value) {
            echo '<tr><td>' . $label . ':</td><td>' . htmlspecialchars($value) . '</td></tr>' . "\n";
        }
    }
    
    echo '</table>' . "\n";
    echo '</body></html>' . "\n";
}

/**
 * Obtener texto de estado de aplicación
 */
function getApplicationStatusText($applied_successfully) {
    if ($applied_successfully === null) {
        return 'Pendiente';
    } elseif ($applied_successfully) {
        return 'Aplicado';
    } else {
        return 'Error';
    }
}

/**
 * Obtener color de estado para Excel
 */
function getStatusColor($applied_successfully) {
    if ($applied_successfully === null) {
        return '#fff3cd'; // Amarillo claro
    } elseif ($applied_successfully) {
        return '#d4edda'; // Verde claro
    } else {
        return '#f8d7da'; // Rojo claro
    }
}
?>