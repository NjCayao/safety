<?php
// Incluir archivos necesarios
require_once 'config/config.php';
require_once 'config/database.php';
require_once 'includes/functions.php';
require_once 'models/User.php';
require_once 'includes/auth.php';

// Iniciar sesión
session_start();

// Si ya está autenticado, redirigir al dashboard
if (Auth::isLoggedIn()) {
    header('Location: index.php');
    exit;
}

// Variables para manejo de errores y mensajes
$error = '';
$username = '';

// Procesar formulario de login
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener datos del formulario
    $username = trim($_POST['username'] ?? '');
    $password = $_POST['password'] ?? '';

    // Validar datos
    if (empty($username) || empty($password)) {
        $error = 'Por favor, complete todos los campos.';
    } else {
        try {
            // Instanciar modelo de usuario
            $userModel = new User();
            
            // Verificar credenciales
            $user = $userModel->verifyPassword($username, $password);
            
            if ($user) {
                // Verificar si el usuario está activo
                if ($user['status'] !== 'active') {
                    $error = 'Este usuario está inactivo. Contacte al administrador.';
                    
                    // Registrar intento fallido en el log
                    log_system_message(
                        'warning',
                        'Intento de inicio de sesión con usuario inactivo: ' . $username,
                        null,
                        'IP: ' . $_SERVER['REMOTE_ADDR']
                    );
                } else {
                    // Iniciar sesión
                    Auth::login($user);
                }
            } else {
                $error = 'Nombre de usuario o contraseña incorrectos.';

                // Registrar intento fallido en el log
                log_system_message(
                    'warning',
                    'Intento de inicio de sesión fallido para el usuario ' . $username,
                    null,
                    'IP: ' . $_SERVER['REMOTE_ADDR']
                );
            }
        } catch (Exception $e) {
            $error = 'Error en el servidor. Inténtelo de nuevo más tarde.';
            error_log('Error en login.php: ' . $e->getMessage());
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
    <!-- /.login-logo -->
    <div class="card card-outline card-primary">
      <div class="card-header text-center">
        <a href="<?php echo BASE_URL; ?>" class="h1"><b>Safety</b>System</a>
      </div>
      <div class="card-body">
        <p class="login-box-msg">Inicie sesión para comenzar</p>

        <?php if (!empty($error)): ?>
          <div class="alert alert-danger">
            <?php echo $error; ?>
          </div>
        <?php endif; ?>

        <form action="login.php" method="post">
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
                <label for="remember">
                  Recordarme
                </label>
              </div>
            </div>
            <!-- /.col -->
            <div class="col-4">
              <button type="submit" class="btn btn-primary btn-block">Ingresar</button>
            </div>
            <!-- /.col -->
          </div>
        </form>

        <p class="mb-1 mt-3">
          <a href="forgot-password.php">Olvidé mi contraseña</a>
        </p>
      </div>
      <!-- /.card-body -->
    </div>
    <!-- /.card -->
  </div>
  <!-- /.login-box -->

  <!-- jQuery -->
  <script src="<?php echo ASSETS_URL; ?>/plugins/jquery/jquery.min.js"></script>
  <!-- Bootstrap 4 -->
  <script src="<?php echo ASSETS_URL; ?>/plugins/bootstrap/js/bootstrap.bundle.min.js"></script>
  <!-- AdminLTE App -->
  <script src="<?php echo ASSETS_URL; ?>/dist/js/adminlte.min.js"></script>
</body>

</html>