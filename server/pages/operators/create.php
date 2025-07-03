<?php
// Incluir archivos necesarios
require_once '../../config/config.php';
require_once '../../config/database.php';
require_once '../../includes/functions.php';
require_once '../../includes/photo_functions.php';

// Verificar si el usuario está autenticado
session_start();
$isLoggedIn = isset($_SESSION['user_id']);

// Si no está autenticado, redirigir al login
if (!$isLoggedIn) {
    header('Location: ../../login.php');
    exit;
}

// Definir título de la página
$pageTitle = 'Registrar Nuevo Operador';

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Operadores' => 'index.php',
    'Nuevo' => ''
];

// Variables para manejo de errores y mensajes
$errors = [];
$successMessage = '';

// Variables para el formulario
$name = '';
$position = '';
$dni_number = '';
$license_number = '';
$license_expiry = '';
$license_status = 'active';
$status = 'active';
$notes = '';

// Procesar formulario si se envió
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    // Obtener y validar datos del formulario
    $name = trim($_POST['name'] ?? '');
    $position = trim($_POST['position'] ?? '');
    $dni_number = trim($_POST['dni_number'] ?? '');
    $license_number = trim($_POST['license_number'] ?? '');
    $license_expiry = $_POST['license_expiry'] ?? '';
    $license_status = $_POST['license_status'] ?? 'active';
    $status = $_POST['status'] ?? 'active';
    $notes = trim($_POST['notes'] ?? '');

    // Validaciones
    if (empty($name)) {
        $errors[] = 'El nombre es obligatorio';
    }

    if (empty($dni_number)) {
        $errors[] = 'El número de DNI o carnet de extranjería es obligatorio';
    } else {
        // Verificar si ya existe un operador con ese DNI
        $existingOperator = db_fetch_one(
            "SELECT id, name FROM operators WHERE dni_number = ?",
            [$dni_number]
        );

        if ($existingOperator) {
            $errors[] = 'El DNI/carnet ' . $dni_number . ' ya está registrado para el operador ' .
                $existingOperator['name'] . ' (ID: ' . $existingOperator['id'] . ')';
        }
    }

    // Si hay licencia pero no fecha de vencimiento
    if (!empty($license_number) && empty($license_expiry)) {
        $errors[] = 'Si ingresa un número de licencia, debe especificar la fecha de vencimiento';
    }

    // Si no hay errores, procesar los datos
    if (empty($errors)) {
        // Generar ID único para el operador
        $operatorId = generate_unique_id('OP', 'operators');

        // Crear datos básicos del operador
        $operatorData = [
            'id' => $operatorId,
            'name' => $name,
            'position' => $position,
            'dni_number' => $dni_number,
            'license_number' => $license_number,
            'license_expiry' => !empty($license_expiry) ? $license_expiry : null,
            'license_status' => $license_status,
            'status' => $status,
            'notes' => $notes,
            'registration_date' => date('Y-m-d H:i:s')
        ];

        try {
            // No hay máquina asignada para un nuevo operador
            $currentMachine = '';

            // Procesar foto de perfil si se subió una
            if (isset($_FILES['profile_photo']) && $_FILES['profile_photo']['error'] === UPLOAD_ERR_OK) {
                $uploadResult = upload_operator_photo(
                    $_FILES['profile_photo'],
                    $operatorId,
                    $dni_number,
                    'profile',
                    $name,
                    $position,
                    $currentMachine
                );

                if ($uploadResult['success']) {
                    $operatorData['photo_path'] = $uploadResult['path'];
                } else {
                    $errors[] = 'Error al subir la foto de perfil: ' . $uploadResult['message'];
                }
            }

            // Procesar fotos adicionales para reconocimiento facial
            $facialPhotos = ['face_photo1', 'face_photo2', 'face_photo3'];
            $photoDbFields = ['face_photo1', 'face_photo2', 'face_photo3'];

            for ($i = 0; $i < 3; $i++) {
                $fieldName = $facialPhotos[$i];
                $dbField = $photoDbFields[$i];

                if (isset($_FILES[$fieldName]) && $_FILES[$fieldName]['error'] === UPLOAD_ERR_OK) {
                    $uploadResult = upload_operator_photo(
                        $_FILES[$fieldName],
                        $operatorId,
                        $dni_number,
                        'face' . ($i + 1),
                        $name,
                        $position,
                        $currentMachine
                    );

                    if ($uploadResult['success']) {
                        $operatorData[$dbField] = $uploadResult['path'];
                    } else {
                        $errors[] = 'Error al subir la foto facial ' . ($i + 1) . ': ' . $uploadResult['message'];
                    }
                }
            }

            // Crear archivo info.txt
            $infoData = [
                'id' => $operatorId,
                'name' => $name,
                'position' => $position,
                'machine' => $currentMachine
            ];

            // Si no hay errores, insertar en la base de datos
            if (empty($errors)) {
                $result = db_insert('operators', $operatorData);

                if ($result) {
                    // Actualizar el archivo info.txt después de insertar en la base de datos
                    update_operator_info_file($dni_number, $infoData);

                    // Registrar en el log
                    log_system_message(
                        'info',
                        'Operador creado: ' . $name . ' (ID: ' . $operatorId . ')',
                        null,
                        'Usuario: ' . $_SESSION['username']
                    );

                    $successMessage = 'Operador registrado exitosamente con ID: ' . $operatorId;

                    // Limpiar los campos del formulario para un nuevo registro
                    $name = $position = $dni_number = $license_number = $license_expiry = $notes = '';
                    $status = $license_status = 'active';
                } else {
                    $errors[] = 'Error al registrar el operador en la base de datos';
                }
            }
        } catch (Exception $e) {
            $errors[] = 'Error al procesar las fotos: ' . $e->getMessage();
        }
    }
}

// Incluir archivos de cabecera y barra lateral
require_once '../../includes/header.php';
require_once '../../includes/sidebar.php';

// Contenido específico de esta página
ob_start();
?>

<!-- Mensajes de error o éxito -->
<?php if (!empty($errors)): ?>
    <div class="alert alert-danger">
        <h5><i class="icon fas fa-ban"></i> Se encontraron errores:</h5>
        <ul>
            <?php foreach ($errors as $error): ?>
                <li><?php echo $error; ?></li>
            <?php endforeach; ?>
        </ul>
    </div>
<?php endif; ?>

<?php if (!empty($successMessage)): ?>
    <div class="alert alert-success">
        <h5><i class="icon fas fa-check"></i> Éxito!</h5>
        <?php echo $successMessage; ?>
    </div>
<?php endif; ?>

<!-- Formulario de registro -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Información del Operador</h3>
    </div>
    <div class="card-body">
        <form action="create.php" method="POST" enctype="multipart/form-data">
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="name">Nombre Completo <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="name" name="name" value="<?php echo htmlspecialchars($name); ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="dni_number">DNI / Carnet de Extranjería <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="dni_number" name="dni_number" value="<?php echo htmlspecialchars($dni_number); ?>" required>
                        <small class="form-text text-muted">Número de documento de identidad</small>
                    </div>

                    <div class="form-group">
                        <label for="position">Posición/Cargo</label>
                        <input type="text" class="form-control" id="position" name="position" value="<?php echo htmlspecialchars($position); ?>">
                    </div>

                    <div class="form-group">
                        <label for="status">Estado del Operador</label>
                        <select class="form-control" id="status" name="status">
                            <option value="active" <?php echo ($status === 'active') ? 'selected' : ''; ?>>Activo</option>
                            <option value="inactive" <?php echo ($status === 'inactive') ? 'selected' : ''; ?>>Inactivo</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="notes">Notas</label>
                        <textarea class="form-control" id="notes" name="notes" rows="3"><?php echo htmlspecialchars($notes); ?></textarea>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="form-group">
                        <label for="license_number">Número de Licencia Interna</label>
                        <input type="text" class="form-control" id="license_number" name="license_number" value="<?php echo htmlspecialchars($license_number); ?>">
                    </div>

                    <div class="form-group">
                        <label for="license_expiry">Fecha de Vencimiento de Licencia</label>
                        <input type="date" class="form-control" id="license_expiry" name="license_expiry" value="<?php echo htmlspecialchars($license_expiry); ?>">
                    </div>

                    <div class="form-group">
                        <label for="license_status">Estado de Licencia</label>
                        <select class="form-control" id="license_status" name="license_status">
                            <option value="active" <?php echo ($license_status === 'active') ? 'selected' : ''; ?>>Activa</option>
                            <option value="expired" <?php echo ($license_status === 'expired') ? 'selected' : ''; ?>>Vencida</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="profile_photo">Fotografía de Perfil</label>
                        <div class="input-group mb-2">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="profile_photo" name="profile_photo" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="profile_photo" id="profile_photo_label">Seleccionar archivo</label>
                            </div>
                        </div>
                        <div id="preview_profile_photo" class="mt-2 d-none">
                            <img src="" class="img-thumbnail" style="max-height: 150px;">
                        </div>
                        <small class="form-text text-muted">Foto principal del operador (máx. 5MB)</small>
                    </div>
                </div>
            </div>

            <h4 class="mt-4 mb-3">Fotos Adicionales para Reconocimiento Facial</h4>
            <div class="row">
                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo1">Foto Facial 1</label>
                        <div class="input-group mb-2">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo1" name="face_photo1" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo1" id="face_photo1_label">Seleccionar archivo</label>
                            </div>
                        </div>
                        <div id="preview_face_photo1" class="mt-2 d-none">
                            <img src="" class="img-thumbnail" style="max-height: 100px;">
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo2">Foto Facial 2</label>
                        <div class="input-group mb-2">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo2" name="face_photo2" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo2" id="face_photo2_label">Seleccionar archivo</label>
                            </div>
                        </div>
                        <div id="preview_face_photo2" class="mt-2 d-none">
                            <img src="" class="img-thumbnail" style="max-height: 100px;">
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo3">Foto Facial 3</label>
                        <div class="input-group mb-2">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo3" name="face_photo3" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo3" id="face_photo3_label">Seleccionar archivo</label>
                            </div>
                        </div>
                        <div id="preview_face_photo3" class="mt-2 d-none">
                            <img src="" class="img-thumbnail" style="max-height: 100px;">
                        </div>
                    </div>
                </div>
            </div>
            <small class="form-text text-muted mb-4">Suba fotos adicionales para mejorar el reconocimiento facial (máx. 5MB cada una)</small>

            <div class="row">
                <div class="col-12">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Guardar
                    </button>
                    <a href="index.php" class="btn btn-secondary">
                        <i class="fas fa-times"></i> Cancelar
                    </a>
                </div>
            </div>
        </form>
    </div>
</div>

<?php
// Capturar el contenido y guardarlo en $pageContent
$pageContent = ob_get_clean();

// Definir contenido para la sección de acciones (botones en header)
$actions = '
<a href="index.php" class="btn btn-secondary">
    <i class="fas fa-arrow-left"></i> Volver
</a>
';

// JavaScript específico para esta página
$extraJs = '
<script>
  $(function () {
    // Función para manejar la previsualización de imágenes
    function handleFileSelect(event) {
      const fileInput = event.target;
      const fileName = fileInput.value.split("\\\\").pop();
      const fileLabel = $(fileInput).next(".custom-file-label");
      const previewContainer = $("#preview_" + fileInput.id);
      const previewImage = previewContainer.find("img");
      
      // Actualizar el nombre del archivo
      if (fileName) {
        fileLabel.html(fileName);
      } else {
        fileLabel.html("Seleccionar archivo");
      }
      
      // Mostrar previsualización si hay un archivo seleccionado
      if (fileInput.files && fileInput.files[0]) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
          previewImage.attr("src", e.target.result);
          previewContainer.removeClass("d-none");
        };
        
        reader.readAsDataURL(fileInput.files[0]);
      } else {
        previewContainer.addClass("d-none");
      }
    }
    
    // Aplicar el manejador a todos los campos de tipo file
    $("input[type=\'file\']").on("change", handleFileSelect);
    
    // Validar DNI en tiempo real
    $("#dni_number").on("blur", function() {
      var dni = $(this).val().trim();
      if (dni) {
        $.ajax({
          url: "check_dni.php",
          method: "POST",
          data: { dni: dni },
          dataType: "json",
          success: function(response) {
            $(".alert-warning").remove(); // Eliminar alertas previas
            
            if (response.exists) {
              $("<div class=\'alert alert-warning mt-2\'>Este DNI ya está registrado para el operador " + response.name + " (ID: " + response.id + ")</div>")
                .insertAfter("#dni_number").fadeIn();
            }
          }
        });
      }
    });
    
    // Actualizar estado de licencia automáticamente según la fecha
    $("#license_expiry").on("change", function() {
      var expiryDate = new Date($(this).val());
      var today = new Date();
      
      if (expiryDate < today) {
        $("#license_status").val("expired");
      } else {
        $("#license_status").val("active");
      }
    });
  });
</script>
';


// Incluir archivo de contenido
require_once '../../includes/content.php';

// Incluir archivo de pie de página
require_once '../../includes/footer.php';
?>