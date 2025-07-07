<?php
// Este archivo genera los datos y el código JavaScript para el gráfico de alertas por operador

require_once 'models/Alert.php';

// Parámetros opcionales para filtrar por fecha
$dateFrom = isset($_GET['date_from']) ? $_GET['date_from'] : null;
$dateTo = isset($_GET['date_to']) ? $_GET['date_to'] : null;
$limit = isset($_GET['limit']) ? intval($_GET['limit']) : 10;

$alertModel = new Alert();
$operatorStats = $alertModel->getAlertsByOperator($limit, $dateFrom, $dateTo);

// Preparar datos para el gráfico
$labels = [];
$data = [];

foreach ($operatorStats as $stat) {
    $labels[] = $stat['name']; // Nombre del operador
    $data[] = $stat['alert_count']; // Cantidad de alertas
}

// Si no hay datos, mostrar un mensaje
if (empty($data)) {
    echo '<div class="alert alert-info">No hay datos disponibles para el período seleccionado.</div>';
} else {
    // ID único para el canvas
    $chartId = 'alertsByOperatorChart';
    
    // Generar el HTML y JavaScript
    ?>
    <canvas id="<?php echo $chartId; ?>" style="min-height: 250px; height: 250px; max-height: 250px; max-width: 100%;"></canvas>
    
    <script>
    // Usar una función anónima para evitar conflictos de variables
    (function() {
        // Obtener el contexto del canvas
        var ctx = document.getElementById('<?php echo $chartId; ?>').getContext('2d');
        
        // Datos para el gráfico
        var data = {
            labels: <?php echo json_encode($labels); ?>,
            datasets: [{
                label: 'Cantidad de Alertas',
                data: <?php echo json_encode($data); ?>,
                backgroundColor: 'rgba(60, 141, 188, 0.7)',
                borderColor: 'rgba(60, 141, 188, 1)',
                borderWidth: 1
            }]
        };
        
        // Opciones del gráfico
        var options = {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    precision: 0
                }
            }
        };
        
        // Crear el gráfico de barras horizontales
        var alertsByOperatorChart = new Chart(ctx, {
            type: 'horizontalBar',
            data: data,
            options: options
        });
    })();
    </script>
    <?php
}
?>