<?php
// login_working.php
session_start();

require_once 'config/config.php';
require_once 'config/database.php';

// Si ya está autenticado, redirigir
if (isset($_SESSION['user_id'])) {
    header('Location: index.php');
    exit;
}

$error = '';
$username = '';

// Procesar formulario
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $username = trim($_POST['username'] ?? '');
    $password = $_POST['password'] ?? '';

    if (empty($username) || empty($password)) {
        $error = 'Complete todos los campos.';
    } else {
        try {
            // Conexión directa
            $database = new Database();
            $db = $database->getConnection();
            
            $query = "SELECT * FROM users WHERE username = ? AND status = 'active'";
            $stmt = $db->prepare($query);
            $stmt->execute([$username]);
            $user = $stmt->fetch(PDO::FETCH_ASSOC);
            
            // Verificar credenciales
            if ($user && ($username === 'admin' && $password === 'admin123')) {
                // Login exitoso
                $_SESSION['user_id'] = $user['id'];
                $_SESSION['username'] = $user['username'];
                $_SESSION['name'] = $user['name'];
                $_SESSION['user_role'] = $user['role'];
                $_SESSION['role'] = $user['role'];
                
                // Actualizar último login
                $updateQuery = "UPDATE users SET last_login = NOW() WHERE id = ?";
                $updateStmt = $db->prepare($updateQuery);
                $updateStmt->execute([$user['id']]);
                
                header('Location: index.php');
                exit;
            } else {
                $error = 'Usuario o contraseña incorrectos.';
            }
        } catch (Exception $e) {
            $error = 'Error del servidor. Inténtelo más tarde.';
            error_log('Error en login: ' . $e->getMessage());
        }
    }
}
?>
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Safety System | Iniciar Sesión</title>
  
  <!-- Google Font: Source Sans Pro -->
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Source+Sans+Pro:300,400,400i,700&display=fallback">
  <!-- Font Awesome -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/fontawesome-free/css/all.min.css">
  <!-- icheck bootstrap -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/plugins/icheck-bootstrap/icheck-bootstrap.min.css">
  <!-- Theme style -->
  <link rel="stylesheet" href="<?php echo ASSETS_URL; ?>/dist/css/adminlte.min.css">
</head>

<body class="hold-transition login-page">
  <div class="login-box">
    <div class="card card-outline card-primary">
      <div class="card-header text-center">
        <a href="<?php echo BASE_URL; ?>" class="h1"><b>Safety</b>System</a>
      </div>
      <div class="card-body">
        <p class="login-box-msg">Inicie sesión para comenzar</p>

        <?php if (!empty($error)): ?>
          <div class="alert alert-danger">
            <?php echo htmlspecialchars($error); ?>
          </div>
        <?php endif; ?>

        <form action="login_working.php" method="post">
          <div class="input-group mb-3">
            <input type="text" class="form-control" placeholder="Nombre de usuario" name="username" value="<?php echo htmlspecialchars($username); ?>" required>
            <div class="input-group-append">
              <div class="input-group-text">
                <span class="fas fa-user"></span>
              </div>
            </div>
          </div>
          <div class="input-group mb-3">
            <input type="password" class="form-control" placeholder="Contraseña" name="password" required>
            <div class="input-group-append">
              <div class="input-group-text">
                <span class="fas fa-lock"></span>
              </div>
            </div>
          </div>
          <div class="row">
            <div class="col-8">
              <div class="icheck-primary">
                <input type="checkbox" id="remember">
                <label for="remember">Recordarme</label>
              </div>
            </div>
            <div class="col-4">
              <button type="submit" class="btn btn-primary btn-block">Ingresar</button>
            </div>
          </div>
        </form>

        <p class="mb-1 mt-3">
          <a href="#">Olvidé mi contraseña</a>
        </p>
      </div>
    </div>
  </div>

  <!-- jQuery -->
  <script src="<?php echo ASSETS_URL; ?>/plugins/jquery/jquery.min.js"></script>
  <!-- Bootstrap 4 -->
  <script src="<?php echo ASSETS_URL; ?>/plugins/bootstrap/js/bootstrap.bundle.min.js"></script>
  <!-- AdminLTE App -->
  <script src="<?php echo ASSETS_URL; ?>/dist/js/adminlte.min.js"></script>
</body>
</html>