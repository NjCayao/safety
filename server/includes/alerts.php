<?php
// Mostrar mensajes de éxito
if (isset($_SESSION['success_message'])) {
    echo '<div class="alert alert-success alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-check"></i> Éxito!</h5>
            ' . $_SESSION['success_message'] . '
          </div>';
    unset($_SESSION['success_message']);
}

// Mostrar mensajes de error
if (isset($_SESSION['error_message'])) {
    echo '<div class="alert alert-danger alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-ban"></i> Error!</h5>
            ' . $_SESSION['error_message'] . '
          </div>';
    unset($_SESSION['error_message']);
}

// Mostrar mensajes de información
if (isset($_SESSION['info_message'])) {
    echo '<div class="alert alert-info alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-info"></i> Información!</h5>
            ' . $_SESSION['info_message'] . '
          </div>';
    unset($_SESSION['info_message']);
}

// Mostrar mensajes de advertencia
if (isset($_SESSION['warning_message'])) {
    echo '<div class="alert alert-warning alert-dismissible fade show">
            <button type="button" class="close" data-dismiss="alert" aria-hidden="true">&times;</button>
            <h5><i class="icon fas fa-exclamation-triangle"></i> Advertencia!</h5>
            ' . $_SESSION['warning_message'] . '
          </div>';
    unset($_SESSION['warning_message']);
}