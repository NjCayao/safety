<?php
// Mostrar todos los errores para depuración
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

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

// Verificar si se proporcionó un ID de operador
if (!isset($_GET['id'])) {
    header('Location: index.php');
    exit;
}

$operatorId = $_GET['id'];

// Obtener información del operador
$operator = db_fetch_one(
    "SELECT * FROM operators WHERE id = ?",
    [$operatorId]
);

// Si no se encuentra el operador, redirigir al listado
if (!$operator) {
    $_SESSION['error_message'] = "Operador no encontrado.";
    header('Location: index.php');
    exit;
}

// Variables para manejo de errores y mensajes
$errors = [];
$successMessage = '';

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
        // Verificar si ya existe un operador con ese DNI (excepto el actual)
        $existingOperator = db_fetch_one(
            "SELECT id FROM operators WHERE dni_number = ? AND id != ?",
            [$dni_number, $operatorId]
        );

        if ($existingOperator) {
            $errors[] = 'Ya existe otro operador con ese número de DNI/carnet';
        }
    }

    // Si hay licencia pero no fecha de vencimiento
    if (!empty($license_number) && empty($license_expiry)) {
        $errors[] = 'Si ingresa un número de licencia, debe especificar la fecha de vencimiento';
    }

    // Si no hay errores, procesar los datos
    if (empty($errors)) {
        // Datos base del operador para actualizar
        $operatorData = [
            'name' => $name,
            'position' => $position,
            'dni_number' => $dni_number,
            'license_number' => $license_number,
            'license_expiry' => !empty($license_expiry) ? $license_expiry : null,
            'license_status' => $license_status,
            'status' => $status,
            'notes' => $notes
        ];

        // Manejo seguro de fotos
        try {
            // Intentar obtener la máquina asignada (de forma segura)
            $currentMachine = get_operator_machine($operatorId);

            // Procesar foto de perfil si se subió una nueva
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

                    // Actualizar inmediatamente la foto en la base de datos para asegurar que se guarde
                    $photoUpdateResult = db_update(
                        'operators',
                        ['photo_path' => $uploadResult['path']],
                        'id = ?',
                        [$operatorId]
                    );

                    if ($photoUpdateResult === false) {
                        $errors[] = 'Error al actualizar la foto en la base de datos';
                    } else {
                        // Verificar que se actualizó correctamente
                        $checkUpdate = db_fetch_one(
                            "SELECT photo_path FROM operators WHERE id = ?",
                            [$operatorId]
                        );

                        if ($checkUpdate['photo_path'] !== $uploadResult['path']) {
                            $errors[] = 'La ruta de la foto no se guardó correctamente en la base de datos';
                        }
                    }
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

            // Actualizar el archivo info.txt
            $infoData = [
                'id' => $operatorId,
                'name' => $name,
                'position' => $position,
                'machine' => $currentMachine
            ];
            update_operator_info_file($dni_number, $infoData);
        } catch (Exception $e) {
            // Capturar cualquier error durante el proceso de archivos
            $errors[] = 'Error al procesar las fotos: ' . $e->getMessage();
        }

        // Si todavía no hay errores, actualizar en la base de datos
        if (empty($errors)) {
            // Verificar que los datos de actualización están correctos
            error_log("Actualizando operador $operatorId con datos: " . print_r($operatorData, true));

            $result = db_update('operators', $operatorData, 'id = ?', [$operatorId]);

            if ($result !== false) {
                // Registrar en el log
                log_system_message(
                    'info',
                    'Operador actualizado: ' . $name . ' (ID: ' . $operatorId . ')',
                    null,
                    'Usuario: ' . $_SESSION['username']
                );

                $successMessage = 'Operador actualizado exitosamente.';

                // Actualizar la variable $operator para mostrar datos actualizados
                $operator = db_fetch_one(
                    "SELECT * FROM operators WHERE id = ?",
                    [$operatorId]
                );
            } else {
                $errors[] = 'Error al actualizar el operador en la base de datos';
            }
        }
    }
}

// Definir título de la página
$pageTitle = 'Editar Operador: ' . htmlspecialchars($operator['name']);

// Configurar breadcrumbs
$breadcrumbs = [
    'Dashboard' => '../../index.php',
    'Operadores' => 'index.php',
    'Detalles' => 'view.php?id=' . $operatorId,
    'Editar' => ''
];

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

<!-- Formulario de edición -->
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Información del Operador</h3>
    </div>
    <div class="card-body">
        <form action="edit.php?id=<?php echo $operatorId; ?>" method="POST" enctype="multipart/form-data">
            <div class="row">
                <div class="col-md-6">
                    <div class="form-group">
                        <label for="name">Nombre Completo <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="name" name="name" value="<?php echo htmlspecialchars($operator['name']); ?>" required>
                    </div>

                    <div class="form-group">
                        <label for="dni_number">DNI / Carnet de Extranjería <span class="text-danger">*</span></label>
                        <input type="text" class="form-control" id="dni_number" name="dni_number" value="<?php echo htmlspecialchars($operator['dni_number'] ?? ''); ?>" required>
                        <small class="form-text text-muted">Número de documento de identidad</small>
                    </div>

                    <div class="form-group">
                        <label for="position">Posición/Cargo</label>
                        <input type="text" class="form-control" id="position" name="position" value="<?php echo htmlspecialchars($operator['position'] ?? ''); ?>">
                    </div>

                    <div class="form-group">
                        <label for="status">Estado del Operador</label>
                        <select class="form-control" id="status" name="status">
                            <option value="active" <?php echo ($operator['status'] === 'active') ? 'selected' : ''; ?>>Activo</option>
                            <option value="inactive" <?php echo ($operator['status'] === 'inactive') ? 'selected' : ''; ?>>Inactivo</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="notes">Notas</label>
                        <textarea class="form-control" id="notes" name="notes" rows="3"><?php echo htmlspecialchars($operator['notes'] ?? ''); ?></textarea>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="form-group">
                        <label for="license_number">Número de Licencia Interna</label>
                        <input type="text" class="form-control" id="license_number" name="license_number" value="<?php echo htmlspecialchars($operator['license_number'] ?? ''); ?>">
                    </div>

                    <div class="form-group">
                        <label for="license_expiry">Fecha de Vencimiento de Licencia</label>
                        <input type="date" class="form-control" id="license_expiry" name="license_expiry" value="<?php echo htmlspecialchars($operator['license_expiry'] ?? ''); ?>">
                    </div>

                    <div class="form-group">
                        <label for="license_status">Estado de Licencia</label>
                        <select class="form-control" id="license_status" name="license_status">
                            <option value="active" <?php echo ($operator['license_status'] === 'active') ? 'selected' : ''; ?>>Activa</option>
                            <option value="expired" <?php echo ($operator['license_status'] === 'expired') ? 'selected' : ''; ?>>Vencida</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="profile_photo">Fotografía de Perfil</label>
                        <?php if (!empty($operator['photo_path'])): ?>
                            <div class="mb-2">
                                <?php
                                // Asegurar que la ruta incluya /server/
                                $photoPath = $operator['photo_path'];
                                if (strpos($photoPath, '/server/') === false && strpos($photoPath, '/operator-photo/') !== false) {
                                    $photoPath = str_replace('/operator-photo/', '/server/operator-photo/', $photoPath);
                                }
                                ?>
                                <img src="<?php echo htmlspecialchars($photoPath); ?>"
                                    alt="Foto actual" class="img-thumbnail" style="max-height: 150px;">
                            </div>
                        <?php endif; ?>
                        <div class="input-group">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="profile_photo" name="profile_photo" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="profile_photo">Seleccionar nueva foto</label>
                            </div>
                        </div>
                        <small class="form-text text-muted">Deje en blanco para mantener la foto actual</small>
                    </div>
                </div>
            </div>

            <h4 class="mt-4 mb-3" id="photos">Fotos Adicionales para Reconocimiento Facial</h4>
            <div class="row">
                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo1">Foto Facial 1</label>
                        <?php if (!empty($operator['face_photo1'])): ?>
                            <div class="mb-2">
                                <?php
                                // Asegurar que la ruta incluya /server/
                                $photoPath = $operator['face_photo1'];
                                if (strpos($photoPath, '/server/') === false && strpos($photoPath, '/operator-photo/') !== false) {
                                    $photoPath = str_replace('/operator-photo/', '/server/operator-photo/', $photoPath);
                                }
                                ?>
                                <img src="<?php echo htmlspecialchars($photoPath); ?>"
                                    alt="Foto facial 1" class="img-thumbnail" style="max-height: 100px;">
                            </div>
                        <?php endif; ?>
                        <div class="input-group">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo1" name="face_photo1" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo1">Seleccionar archivo</label>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo2">Foto Facial 2</label>
                        <?php if (!empty($operator['face_photo2'])): ?>
                            <div class="mb-2">
                                <?php
                                // Asegurar que la ruta incluya /server/
                                $photoPath = $operator['face_photo2'];
                                if (strpos($photoPath, '/server/') === false && strpos($photoPath, '/operator-photo/') !== false) {
                                    $photoPath = str_replace('/operator-photo/', '/server/operator-photo/', $photoPath);
                                }
                                ?>
                                <img src="<?php echo htmlspecialchars($photoPath); ?>"
                                    alt="Foto facial 1" class="img-thumbnail" style="max-height: 100px;">
                            </div>
                        <?php endif; ?>
                        <div class="input-group">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo2" name="face_photo2" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo2">Seleccionar archivo</label>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="form-group">
                        <label for="face_photo3">Foto Facial 3</label>
                        <?php if (!empty($operator['face_photo3'])): ?>
                            <div class="mb-2">
                                <?php
                                // Asegurar que la ruta incluya /server/
                                $photoPath = $operator['face_photo3'];
                                if (strpos($photoPath, '/server/') === false && strpos($photoPath, '/operator-photo/') !== false) {
                                    $photoPath = str_replace('/operator-photo/', '/server/operator-photo/', $photoPath);
                                }
                                ?>
                                <img src="<?php echo htmlspecialchars($photoPath); ?>"
                                    alt="Foto facial 1" class="img-thumbnail" style="max-height: 100px;">
                            </div>
                        <?php endif; ?>
                        <div class="input-group">
                            <div class="custom-file">
                                <input type="file" class="custom-file-input" id="face_photo3" name="face_photo3" accept="image/jpeg,image/png">
                                <label class="custom-file-label" for="face_photo3">Seleccionar archivo</label>
                            </div>
                        </div>
                    </div>
                </div>

                
            </div>
            <small class="form-text text-muted mb-4">Suba nuevas fotos para reemplazar las existentes o deje en blanco para mantenerlas</small>
            <div class="row">
                <div class="col-12">
                    <button type="submit" class="btn btn-primary">
                        <i class="fas fa-save"></i> Guardar Cambios
                    </button>
                    <a href="view.php" class="btn btn-secondary">
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
<div class="btn-group">
    <a href="view.php?id=' . $operatorId . '" class="btn btn-secondary">
        <i class="fas fa-arrow-left"></i> Volver
    </a>
</div>
';

// JavaScript específico para esta página
$extraJs = '
<script>
  $(function () {
    // BS custom file input
    if (typeof bsCustomFileInput !== "undefined") {
      bsCustomFileInput.init();
    }
    
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
    
    // Previsualización de imágenes seleccionadas
    $("input[type=\'file\']").on("change", function() {
      var fieldName = $(this).attr("name");
      var preview = $(this).closest(".form-group").find("img");
      
      if (this.files && this.files[0]) {
        var reader = new FileReader();
        
        reader.onload = function(e) {
          if (preview.length) {
            preview.attr("src", e.target.result);
          } else {
            $("<div class=\'mb-2\'><img src=\'" + e.target.result + "\' class=\'img-thumbnail\' style=\'max-height: 100px;\'></div>")
              .insertBefore($(this).closest(".input-group"));
          }
        }.bind(this);
        
        reader.readAsDataURL(this.files[0]);
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