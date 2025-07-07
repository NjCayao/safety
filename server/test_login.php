<?php
echo "Test 1: PHP básico ✓<br>";

session_start();
echo "Test 2: Session ✓<br>";

require_once 'config/config.php';
echo "Test 3: Config ✓<br>";

require_once 'config/database.php';
echo "Test 4: Database ✓<br>";

$database = new Database();
$db = $database->getConnection();
echo "Test 5: Conexión ✓<br>";

if ($_POST) {
    echo "Test 6: POST recibido<br>";
    
    $username = $_POST['username'] ?? '';
    $password = $_POST['password'] ?? '';
    
    echo "Usuario: $username<br>";
    echo "Password: $password<br>";
    
    if ($username === 'admin' && $password === 'admin123') {
        echo "✅ CREDENCIALES CORRECTAS";
    } else {
        echo "❌ Credenciales incorrectas";
    }
}
?>

<form method="post">
    <input type="text" name="username" placeholder="Usuario" value="admin"><br>
    <input type="password" name="password" placeholder="Contraseña" value="admin123"><br>
    <button type="submit">Probar</button>
</form>