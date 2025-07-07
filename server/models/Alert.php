<?php
class Alert
{
    private $pdo;

    public function __construct()
    {
        // Usar __FILE__ para determinar la ruta base del proyecto
        $base_dir = dirname(dirname(__FILE__));
        require_once $base_dir . '/config/database.php';
        $database = new Database();
        $this->pdo = $database->getConnection();
    }

    /**
     * Obtiene todas las alertas con filtros opcionales
     * @param array $filters Arreglo con filtros (operator_id, machine_id, alert_type, date_from, date_to, acknowledged)
     * @param int $limit Límite de resultados
     * @param int $offset Desplazamiento para paginación
     * @return array Arreglo con las alertas
     */
    public function getAlerts($filters = [], $limit = 10, $offset = 0)
    {
        $sql = "SELECT a.*, o.name as operator_name, o.dni_number as operator_dni, 
                m.name as machine_name 
                FROM alerts a
                LEFT JOIN operators o ON a.operator_id = o.id
                LEFT JOIN machines m ON a.machine_id = m.id
                WHERE 1=1";

        $params = [];

        // Aplicar filtros
        if (isset($filters['operator_id']) && !empty($filters['operator_id'])) {
            $sql .= " AND a.operator_id = ?";
            $params[] = $filters['operator_id'];
        }

        if (isset($filters['machine_id']) && !empty($filters['machine_id'])) {
            $sql .= " AND a.machine_id = ?";
            $params[] = $filters['machine_id'];
        }

        //filtro por dispositivo
        if (!empty($filters['device_id'])) {
            $where[] = "a.device_id = :device_id";
            $params['device_id'] = $filters['device_id'];
        }

        if (isset($filters['alert_type']) && !empty($filters['alert_type'])) {
            $sql .= " AND a.alert_type = ?";
            $params[] = $filters['alert_type'];
        }

        if (isset($filters['date_from']) && !empty($filters['date_from'])) {
            $sql .= " AND a.timestamp >= ?";
            $params[] = $filters['date_from'] . ' 00:00:00';
        }

        if (isset($filters['date_to']) && !empty($filters['date_to'])) {
            $sql .= " AND a.timestamp <= ?";
            $params[] = $filters['date_to'] . ' 23:59:59';
        }

        if (isset($filters['acknowledged'])) {
            $sql .= " AND a.acknowledged = ?";
            $params[] = $filters['acknowledged'];
        }

        // Ordenar por fecha descendente (más recientes primero)
        $sql .= " ORDER BY a.timestamp DESC";

        // Aplicar límite y offset para paginación
        if ($limit > 0) {
            $sql .= " LIMIT ?, ?";
            $params[] = (int)$offset;
            $params[] = (int)$limit;
        }

        $stmt = $this->pdo->prepare($sql);

        // Bind de parámetros
        if (!empty($params)) {
            $i = 1;
            foreach ($params as $param) {
                $stmt->bindValue($i++, $param, is_int($param) ? PDO::PARAM_INT : PDO::PARAM_STR);
            }
        }

        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    /**
     * Cuenta el total de alertas con los filtros aplicados (para paginación)
     */
    public function countAlerts($filters = [])
    {
        $sql = "SELECT COUNT(*) as total FROM alerts a WHERE 1=1";

        $params = [];

        // Aplicar los mismos filtros que en getAlerts
        if (isset($filters['operator_id']) && !empty($filters['operator_id'])) {
            $sql .= " AND a.operator_id = ?";
            $params[] = $filters['operator_id'];
        }

        if (isset($filters['machine_id']) && !empty($filters['machine_id'])) {
            $sql .= " AND a.machine_id = ?";
            $params[] = $filters['machine_id'];
        }

        if (isset($filters['alert_type']) && !empty($filters['alert_type'])) {
            $sql .= " AND a.alert_type = ?";
            $params[] = $filters['alert_type'];
        }

        if (isset($filters['date_from']) && !empty($filters['date_from'])) {
            $sql .= " AND a.timestamp >= ?";
            $params[] = $filters['date_from'] . ' 00:00:00';
        }

        if (isset($filters['date_to']) && !empty($filters['date_to'])) {
            $sql .= " AND a.timestamp <= ?";
            $params[] = $filters['date_to'] . ' 23:59:59';
        }

        if (isset($filters['acknowledged'])) {
            $sql .= " AND a.acknowledged = ?";
            $params[] = $filters['acknowledged'];
        }

        $stmt = $this->pdo->prepare($sql);

        // Bind de parámetros
        if (!empty($params)) {
            $i = 1;
            foreach ($params as $param) {
                $stmt->bindValue($i++, $param, is_int($param) ? PDO::PARAM_INT : PDO::PARAM_STR);
            }
        }

        $stmt->execute();
        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        return $result['total'];
    }

    /**
     * Obtiene una alerta específica por ID
     */
    public function getAlertById($alertId)
    {
        $sql = "SELECT a.*, o.name as operator_name, o.dni_number as operator_dni, 
                m.name as machine_name 
                FROM alerts a
                LEFT JOIN operators o ON a.operator_id = o.id
                LEFT JOIN machines m ON a.machine_id = m.id
                WHERE a.id = ?";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $alertId, PDO::PARAM_INT);
        $stmt->execute();

        return $stmt->fetch(PDO::FETCH_ASSOC);
    }

    /**
     * Obtiene las imágenes asociadas a una alerta
     * En tu caso, parece que la ruta de la imagen está en la columna image_path directamente
     */
    public function getAlertImages($alertId)
    {
        $sql = "SELECT id, image_path, timestamp FROM alerts WHERE id = ?";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $alertId, PDO::PARAM_INT);
        $stmt->execute();

        $result = $stmt->fetch(PDO::FETCH_ASSOC);

        $images = [];
        if ($result && !empty($result['image_path'])) {
            $images[] = [
                'id' => $result['id'],
                'image_path' => $result['image_path'],
                'created_at' => $result['timestamp']
            ];
        }

        return $images;
    }

    /**
     * Marca una alerta como revisada
     */
    public function markAsAcknowledged($alertId, $acknowledgedBy)
    {
        $sql = "UPDATE alerts SET acknowledged = 1, 
                acknowledged_by = ?, acknowledgement_time = NOW() 
                WHERE id = ?";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $acknowledgedBy, PDO::PARAM_STR);
        $stmt->bindValue(2, $alertId, PDO::PARAM_INT);

        return $stmt->execute();
    }

    /**
     * Obtiene estadísticas de alertas por tipo
     */
    public function getAlertsByType($dateFrom = null, $dateTo = null)
    {
        $sql = "SELECT alert_type, COUNT(*) as count 
                FROM alerts 
                WHERE 1=1";

        $params = [];

        if ($dateFrom) {
            $sql .= " AND timestamp >= ?";
            $params[] = $dateFrom . ' 00:00:00';
        }

        if ($dateTo) {
            $sql .= " AND timestamp <= ?";
            $params[] = $dateTo . ' 23:59:59';
        }

        $sql .= " GROUP BY alert_type ORDER BY count DESC";

        $stmt = $this->pdo->prepare($sql);

        // Bind de parámetros si existen
        if (!empty($params)) {
            $i = 1;
            foreach ($params as $param) {
                $stmt->bindValue($i++, $param, PDO::PARAM_STR);
            }
        }

        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    /**
     * Obtiene estadísticas de alertas por operador
     */
    public function getAlertsByOperator($limit = 10, $dateFrom = null, $dateTo = null)
    {
        $sql = "SELECT o.id, o.name, o.dni_number, COUNT(a.id) as alert_count 
                FROM operators o
                LEFT JOIN alerts a ON o.id = a.operator_id";

        $params = [];
        $whereAdded = false;

        if ($dateFrom) {
            $sql .= " WHERE a.timestamp >= ?";
            $params[] = $dateFrom . ' 00:00:00';
            $whereAdded = true;
        }

        if ($dateTo) {
            $sql .= $whereAdded ? " AND a.timestamp <= ?" : " WHERE a.timestamp <= ?";
            $params[] = $dateTo . ' 23:59:59';
        }

        $sql .= " GROUP BY o.id ORDER BY alert_count DESC LIMIT ?";
        $params[] = (int)$limit;

        $stmt = $this->pdo->prepare($sql);

        // Bind de parámetros
        if (!empty($params)) {
            $i = 1;
            foreach ($params as $param) {
                $stmt->bindValue($i++, $param, (is_int($param) ? PDO::PARAM_INT : PDO::PARAM_STR));
            }
        }

        $stmt->execute();
        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    /**
     * Obtiene datos para el gráfico de tendencias
     */
    public function getAlertsTrend($days = 30, $groupBy = 'day')
    {
        $format = ($groupBy == 'hour') ? '%Y-%m-%d %H:00' : '%Y-%m-%d';

        $sql = "SELECT 
                DATE_FORMAT(timestamp, ?) as date_group,
                COUNT(*) as count
                FROM alerts
                WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL ? DAY)
                GROUP BY date_group
                ORDER BY timestamp";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $format, PDO::PARAM_STR);
        $stmt->bindValue(2, $days, PDO::PARAM_INT);
        $stmt->execute();

        return $stmt->fetchAll(PDO::FETCH_ASSOC);
    }

    /**
     * Obtiene el ID de la alerta anterior
     */
    public function getPreviousAlertId($alertId)
    {
        $sql = "SELECT id FROM alerts WHERE id < ? ORDER BY id DESC LIMIT 1";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $alertId, PDO::PARAM_INT);
        $stmt->execute();

        $result = $stmt->fetch(PDO::FETCH_ASSOC);
        return $result ? $result['id'] : null;
    }

    /**
     * Obtiene el ID de la alerta siguiente
     */
    public function getNextAlertId($alertId)
    {
        $sql = "SELECT id FROM alerts WHERE id > ? ORDER BY id ASC LIMIT 1";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $alertId, PDO::PARAM_INT);
        $stmt->execute();

        $result = $stmt->fetch(PDO::FETCH_ASSOC);
        return $result ? $result['id'] : null;
    }


    /**
     * Marca una alerta como revisada con detalles adicionales
     */
    public function markAsAcknowledgedWithDetails($alertId, $acknowledgedBy, $actionsTaken = '', $comments = '')
    {
        $sql = "UPDATE alerts SET 
            acknowledged = 1, 
            acknowledged_by = ?, 
            acknowledgement_time = NOW(),
            actions_taken = ?,
            ack_comments = ?
            WHERE id = ?";

        $stmt = $this->pdo->prepare($sql);
        $stmt->bindValue(1, $acknowledgedBy, PDO::PARAM_STR);
        $stmt->bindValue(2, $actionsTaken, PDO::PARAM_STR);
        $stmt->bindValue(3, $comments, PDO::PARAM_STR);
        $stmt->bindValue(4, $alertId, PDO::PARAM_INT);

        return $stmt->execute();
    }
}
