<?php
session_start();
require_once 'config/config.php';
require_once 'config/database.php';

if ($_POST) {
    $username = $_POST['username'];
    $password = $_POST['password'];
    
    try {
        // Conexión directa a BD
        $database = new Database();
        $db = $database->getConnection();
        
        $query = "SELECT * FROM users WHERE username = ? AND status = 'active'";
        $stmt = $db->prepare($query);
        $stmt->execute([$username]);
        $user = $stmt->fetch(PDO::FETCH_ASSOC);
        
        // Login simple
        if ($user && ($username === 'admin' && $password === 'admin123')) {
            $_SESSION['user_id'] = $user['id'];
            $_SESSION['username'] = $user['username'];
            $_SESSION['name'] = $user['name'];
            $_SESSION['user_role'] = $user['role'];
            
            echo "Login exitoso! <a href='index.php'>Ir al dashboard</a>";
            exit;
        } else {
            $error = 'Credenciales incorrectas';
        }
    } catch(Exception $e) {
        $error = 'Error: ' . $e->getMessage();
    }
}
?>
<!DOCTYPE html>
<html>
<head><title>Login Test</title></head>
<body>
    <h2>Login de Prueba</h2>
    <?php if (isset($error)) echo "<p style='color:red'>$error</p>"; ?>
    <form method="post">
        Usuario: <input type="text" name="username" value="admin"><br><br>
        Contraseña: <input type="password" name="password" value="admin123"><br><br>
        <button type="submit">Entrar</button>
    </form>
</body>
</html>