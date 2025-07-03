<?php
/**
 * Clase de conexión a la base de datos
 */
class Database {
    private static $pdo = null;
    private static $db_error = null;
    
    /**
     * Constructor
     */
    public function __construct() {
        if (self::$pdo === null) {
            $this->connect();
        }
    }
    
    /**
     * Establece la conexión a la base de datos
     */
    private function connect() {
        // Parámetros de conexión
        $db_host = 'localhost';
        $db_name = 'safetysystem';
        $db_user = 'safetysystem';  // Cambia esto por tu usuario de MySQL
        $db_pass = 'Thenilfer1414';      // Cambia esto por tu contraseña de MySQL
        $db_charset = 'utf8mb4';
        
        // Opciones para PDO
        $options = [
            PDO::ATTR_ERRMODE            => PDO::ERRMODE_EXCEPTION,
            PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
            PDO::ATTR_EMULATE_PREPARES   => false,
        ];
        
        try {
            // Crear conexión PDO
            self::$pdo = new PDO(
                "mysql:host=$db_host;dbname=$db_name;charset=$db_charset",
                $db_user,
                $db_pass,
                $options
            );
        } catch (PDOException $e) {
            // Guardar mensaje de error
            self::$db_error = 'Error de conexión a la base de datos: ' . $e->getMessage();
            
            // En producción, es mejor no mostrar el error específico al usuario
            // pero registrarlo en un archivo de log
            error_log(self::$db_error);
        }
    }
    
    /**
     * Obtiene la conexión a la base de datos
     */
    public function getConnection() {
        return self::$pdo;
    }
    
    /**
     * Obtiene el error de conexión, si existe
     */
    public function getError() {
        return self::$db_error;
    }
}

// Funciones auxiliares
/**
 * Función para ejecutar consultas SQL
 * 
 * @param string $sql Consulta SQL con marcadores de posición
 * @param array $params Parámetros para la consulta
 * @return PDOStatement|false Resultado de la consulta o false en caso de error
 */
function db_query($sql, $params = []) {
    $db = new Database();
    $pdo = $db->getConnection();
    
    if ($pdo === null) {
        return false;
    }
    
    try {
        $stmt = $pdo->prepare($sql);
        $stmt->execute($params);
        return $stmt;
    } catch (PDOException $e) {
        $error = 'Error en la consulta: ' . $e->getMessage();
        error_log($error);
        return false;
    }
}

/**
 * Obtiene un solo registro
 * 
 * @param string $sql Consulta SQL
 * @param array $params Parámetros
 * @return array|false Registro o false si no se encontró
 */
function db_fetch_one($sql, $params = []) {
    $stmt = db_query($sql, $params);
    return $stmt ? $stmt->fetch() : false;
}

/**
 * Obtiene todos los registros
 * 
 * @param string $sql Consulta SQL
 * @param array $params Parámetros
 * @return array Registros obtenidos
 */
function db_fetch_all($sql, $params = []) {
    $stmt = db_query($sql, $params);
    return $stmt ? $stmt->fetchAll() : [];
}

/**
 * Inserta un registro y devuelve el ID insertado
 * 
 * @param string $table Nombre de la tabla
 * @param array $data Datos a insertar (columna => valor)
 * @return int|string|false ID del registro insertado o false en caso de error
 */
function db_insert($table, $data) {
    $db = new Database();
    $pdo = $db->getConnection();
    
    if ($pdo === null) {
        return false;
    }
    
    try {
        $columns = implode(', ', array_keys($data));
        $placeholders = implode(', ', array_fill(0, count($data), '?'));
        
        $sql = "INSERT INTO $table ($columns) VALUES ($placeholders)";
        
        $stmt = $pdo->prepare($sql);
        $stmt->execute(array_values($data));
        
        // Para tablas con AUTO_INCREMENT, usamos lastInsertId
        // Para tablas con PK definida manualmente, devolvemos ese valor
        if ($table == 'operators' || $table == 'machines') {
            return $data['id']; // Asumiendo que 'id' está en $data
        } else {
            return $pdo->lastInsertId();
        }
    } catch (PDOException $e) {
        $error = 'Error al insertar datos: ' . $e->getMessage();
        error_log($error);
        return false;
    }
}

/**
 * Actualiza registros
 * 
 * @param string $table Nombre de la tabla
 * @param array $data Datos a actualizar (columna => valor)
 * @param string $where Condición WHERE
 * @param array $params Parámetros para la condición WHERE
 * @return int|false Número de registros actualizados o false en caso de error
 */
function db_update($table, $data, $where, $params = []) {
    $db = new Database();
    $pdo = $db->getConnection();
    
    if ($pdo === null) {
        return false;
    }
    
    try {
        $set = [];
        foreach ($data as $column => $value) {
            $set[] = "$column = ?";
        }
        
        $sql = "UPDATE $table SET " . implode(', ', $set) . " WHERE $where";
        
        $stmt = $pdo->prepare($sql);
        $stmt->execute(array_merge(array_values($data), $params));
        
        return $stmt->rowCount();
    } catch (PDOException $e) {
        $error = 'Error al actualizar datos: ' . $e->getMessage();
        error_log($error);
        return false;
    }
}

/**
 * Elimina registros
 * 
 * @param string $table Nombre de la tabla
 * @param string $where Condición WHERE
 * @param array $params Parámetros para la condición WHERE
 * @return int|false Número de registros eliminados o false en caso de error
 */
function db_delete($table, $where, $params = []) {
    $db = new Database();
    $pdo = $db->getConnection();
    
    if ($pdo === null) {
        return false;
    }
    
    try {
        $sql = "DELETE FROM $table WHERE $where";
        
        $stmt = $pdo->prepare($sql);
        $stmt->execute($params);
        
        return $stmt->rowCount();
    } catch (PDOException $e) {
        $error = 'Error al eliminar datos: ' . $e->getMessage();
        error_log($error);
        return false;
    }
}

/**
 * Genera un ID alfanumérico único para las tablas operators y machines
 * 
 * @param string $prefix Prefijo para el ID (ej: OP, MAQ)
 * @param string $table Nombre de la tabla para verificar duplicados
 * @return string ID único generado
 */
function generate_unique_id($prefix, $table) {
    $db = new Database();
    $pdo = $db->getConnection();
    
    if ($pdo === null) {
        return $prefix . rand(100, 999);
    }
    
    do {
        // Generar número aleatorio de 3 dígitos
        $number = str_pad(rand(1, 999), 3, '0', STR_PAD_LEFT);
        $id = $prefix . $number;
        
        // Verificar si ya existe
        $stmt = $pdo->prepare("SELECT id FROM $table WHERE id = ?");
        $stmt->execute([$id]);
        $exists = $stmt->fetch();
    } while ($exists);
    
    return $id;
}

// Las demás funciones que ya tenías...
function get_active_alerts($limit = 10) {
    return db_fetch_all(
        "SELECT a.id, a.alert_type, a.timestamp, a.details, a.image_path, 
                o.id as operator_id, o.name as operator_name, 
                m.id as machine_id, m.name as machine_name 
         FROM alerts a
         LEFT JOIN operators o ON a.operator_id = o.id
         LEFT JOIN machines m ON a.machine_id = m.id
         WHERE a.acknowledged = 0
         ORDER BY a.timestamp DESC
         LIMIT ?",
        [$limit]
    );
}

function get_dashboard_stats() {
    $stats = [];
    
    // Total de operadores activos
    $stats['active_operators'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM operators WHERE status = 'active'",
        []
    )['count'];
    
    // Total de máquinas activas
    $stats['active_machines'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM machines WHERE status = 'active'",
        []
    )['count'];
    
    // Alertas de hoy
    $stats['today_alerts'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM alerts WHERE DATE(timestamp) = CURDATE()",
        []
    )['count'];
    
    // Alertas pendientes
    $stats['pending_alerts'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM alerts WHERE acknowledged = 0",
        []
    )['count'];
    
    // NUEVAS estadísticas de dispositivos
    $stats['total_devices'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM devices"
    )['count'];
    
    $stats['online_devices'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM devices WHERE status = 'online'"
    )['count'];
    
    $stats['offline_devices'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM devices WHERE status = 'offline'"
    )['count'];
    
    $stats['device_alerts'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM alerts WHERE alert_type = 'device_error' AND acknowledged = 0"
    )['count'];
    
    $stats['devices_unassigned'] = db_fetch_one(
        "SELECT COUNT(*) as count FROM devices WHERE machine_id IS NULL"
    )['count'];
    
    return $stats;
}


function get_chart_data($chart_type, $days = 7) {
    $data = [];
    
    switch ($chart_type) {
        case 'alerts_by_type':
            $results = db_fetch_all(
                "SELECT alert_type, COUNT(*) as count 
                 FROM alerts 
                 WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL ? DAY)
                 GROUP BY alert_type 
                 ORDER BY count DESC",
                [$days]
            );
            
            foreach ($results as $row) {
                $data[] = [
                    'label' => ucfirst($row['alert_type']),
                    'value' => $row['count']
                ];
            }
            break;
            
        case 'alerts_by_day':
            $results = db_fetch_all(
                "SELECT DATE(timestamp) as date, COUNT(*) as count 
                 FROM alerts 
                 WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL ? DAY)
                 GROUP BY DATE(timestamp) 
                 ORDER BY date",
                [$days]
            );
            
            foreach ($results as $row) {
                $data[] = [
                    'label' => date('d/m', strtotime($row['date'])),
                    'value' => $row['count']
                ];
            }
            break;
    }
    
    return $data;
}
?>