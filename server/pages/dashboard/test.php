<?php
// Activar la visualización de errores
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// Función para probar la inclusión de un archivo
function test_include($file_path) {
    echo "Probando inclusión de $file_path: ";
    if (!file_exists($file_path)) {
        echo "<span style='color:red'>FALLIDO - El archivo no existe</span><br>";
        return false;
    }
    
    try {
        include_once $file_path;
        echo "<span style='color:green'>OK</span><br>";
        return true;
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
        return false;
    }
}

// Función para probar la creación de una clase
function test_class($class_name, $file_path = null) {
    echo "Probando clase $class_name: ";
    
    if ($file_path && !file_exists($file_path)) {
        echo "<span style='color:red'>FALLIDO - El archivo de clase no existe</span><br>";
        return false;
    }
    
    if ($file_path) {
        include_once $file_path;
    }
    
    if (!class_exists($class_name)) {
        echo "<span style='color:red'>FALLIDO - La clase no existe</span><br>";
        return false;
    }
    
    try {
        $instance = new $class_name();
        echo "<span style='color:green'>OK</span><br>";
        return $instance;
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
        return false;
    }
}

// HTML básico para el test
echo "<!DOCTYPE html>
<html>
<head>
    <title>Test de Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; background: #f9f9f9; }
        h2 { color: #333; }
    </style>
</head>
<body>
    <h1>Test de Dashboard</h1>";

// SECCIÓN 1: Probar archivos básicos
echo "<div class='section'>
    <h2>1. Archivos de Configuración</h2>";

$config_ok = test_include('../../config/config.php');
test_include('../../config/database.php');

echo "</div>";

// SECCIÓN 2: Probar inclusión de header/sidebar/footer
echo "<div class='section'>
    <h2>2. Archivos de Diseño</h2>";

$header_ok = test_include('../../includes/header.php');
$sidebar_ok = test_include('../../includes/sidebar.php');
$footer_ok = test_include('../../includes/footer.php');

echo "</div>";

// SECCIÓN 3: Probar modelos
echo "<div class='section'>
    <h2>3. Modelos</h2>";

$alert_ok = test_include('../../models/Alert.php');
$operator_ok = test_include('../../models/Operator.php');
$machine_ok = test_include('../../models/Machine.php');

echo "</div>";

// SECCIÓN 4: Probar instancias de modelos
echo "<div class='section'>
    <h2>4. Instancias de Modelos</h2>";

if ($alert_ok) {
    $alert_instance = test_class('Alert');
} else {
    echo "No se puede probar instancia de Alert porque no se cargó el archivo<br>";
}

if ($operator_ok) {
    $operator_instance = test_class('Operator');
} else {
    echo "No se puede probar instancia de Operator porque no se cargó el archivo<br>";
}

if ($machine_ok) {
    $machine_instance = test_class('Machine');
} else {
    echo "No se puede probar instancia de Machine porque no se cargó el archivo<br>";
}

echo "</div>";

// SECCIÓN 5: Probar archivos de gráficos
echo "<div class='section'>
    <h2>5. Archivos de Gráficos</h2>";

test_include('../../includes/charts/alerts_by_type.php');
test_include('../../includes/charts/alerts_by_operator.php');
test_include('../../includes/charts/alerts_trend.php');

echo "</div>";

// SECCIÓN 6: Probar funciones básicas de los modelos
echo "<div class='section'>
    <h2>6. Funciones de Modelos</h2>";

if (isset($alert_instance) && $alert_instance) {
    echo "Probando método countAlerts: ";
    try {
        $count = $alert_instance->countAlerts(['date_from' => date('Y-m-d')]);
        echo "<span style='color:green'>OK - Resultado: $count</span><br>";
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
    }
    
    echo "Probando método getAlerts: ";
    try {
        $alerts = $alert_instance->getAlerts([], 1, 0);
        echo "<span style='color:green'>OK - Se encontraron " . count($alerts) . " alertas</span><br>";
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
    }
}

if (isset($operator_instance) && $operator_instance) {
    echo "Probando método countOperators: ";
    try {
        $count = $operator_instance->countOperators();
        echo "<span style='color:green'>OK - Resultado: $count</span><br>";
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
    }
}

if (isset($machine_instance) && $machine_instance) {
    echo "Probando método countMachines: ";
    try {
        $count = $machine_instance->countMachines();
        echo "<span style='color:green'>OK - Resultado: $count</span><br>";
    } catch (Throwable $e) {
        echo "<span style='color:red'>FALLIDO - " . $e->getMessage() . "</span><br>";
    }
}

echo "</div>";

echo "</body></html>";