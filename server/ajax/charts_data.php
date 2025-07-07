<?php
// Este archivo proporciona datos para los gráficos vía AJAX
require_once "../config/database.php";

// Habilitar CORS para solicitudes AJAX
header("Access-Control-Allow-Origin: *");
header("Content-Type: application/json; charset=UTF-8");

// Inicializar conexión a la base de datos
$database = new Database();
$pdo = $database->getConnection();

// Verificar qué gráfico se solicita
$chart = isset($_GET["chart"]) ? $_GET["chart"] : "";

// Devolver datos según el gráfico solicitado
switch ($chart) {
    case "alerts_by_type":
        // Consultar alertas por tipo
        $sql = "SELECT alert_type, COUNT(*) as count FROM alerts GROUP BY alert_type ORDER BY count DESC";
        $stmt = $pdo->prepare($sql);
        $stmt->execute();
        $results = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        // Definir etiquetas para los tipos de alerta
        $alertTypeLabels = [
            "fatigue" => "Fatiga",
            "phone" => "Uso de celular",
            "smoking" => "Fumando",
            "unauthorized" => "No autorizado",
            "yawn" => "Bostezo",
            "distraction" => "Distracción",
            "behavior" => "Comportamiento anómalo",
            "other" => "Otro"
        ];
        
        // Formatear resultados
        $data = [];
        foreach ($results as $row) {
            $label = isset($alertTypeLabels[$row["alert_type"]]) ? $alertTypeLabels[$row["alert_type"]] : ucfirst($row["alert_type"]);
            $data[] = [
                "type" => $row["alert_type"],
                "count" => (int)$row["count"],
                "label" => $label
            ];
        }
        
        echo json_encode($data);
        break;
        
    case "alerts_by_operator":
        // Consultar alertas por operador
        $sql = "SELECT o.id, o.name, COUNT(a.id) as count 
                FROM operators o 
                JOIN alerts a ON o.id = a.operator_id 
                GROUP BY o.id 
                ORDER BY count DESC 
                LIMIT 10";
        $stmt = $pdo->prepare($sql);
        $stmt->execute();
        $data = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        echo json_encode($data);
        break;
        
    case "alerts_trend":
        // Consultar tendencia de alertas (últimos 7 días)
        $sql = "SELECT DATE(timestamp) as date, COUNT(*) as count 
                FROM alerts 
                WHERE timestamp >= DATE_SUB(CURDATE(), INTERVAL 7 DAY) 
                GROUP BY DATE(timestamp) 
                ORDER BY date";
        $stmt = $pdo->prepare($sql);
        $stmt->execute();
        $data = $stmt->fetchAll(PDO::FETCH_ASSOC);
        
        echo json_encode($data);
        break;
        
    default:
        // Si no se especifica un gráfico válido
        echo json_encode(["error" => "Gráfico no especificado o inválido"]);
        break;
}
