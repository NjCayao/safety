
<!-- <style> 

.main-footer {
    margin-top: 50px;
}
</style> -->

<!-- /.content-wrapper -->
<footer class="main-footer">
    <strong>Copyright &copy; <?php echo date('Y'); ?> <a href="#">Safety System</a>.</strong>
    Todos los derechos reservados.
    <div class="float-right d-none d-sm-inline-block">
      <b>Versión</b> 1.0.0
    </div>
  </footer>

  <!-- Control Sidebar -->
  <aside class="control-sidebar control-sidebar-dark">
    <!-- Control sidebar content goes here -->
  </aside>
  <!-- /.control-sidebar -->
</div>
<!-- ./wrapper -->

<!-- jQuery -->
<script src="<?php echo ASSETS_URL; ?>/plugins/jquery/jquery.min.js"></script>
<!-- jQuery UI 1.11.4 -->
<script src="<?php echo ASSETS_URL; ?>/plugins/jquery-ui/jquery-ui.min.js"></script>
<!-- Resolve conflict in jQuery UI tooltip with Bootstrap tooltip -->
<script>
  $.widget.bridge('uibutton', $.ui.button)
</script>
<!-- Bootstrap 4 -->
<script src="<?php echo ASSETS_URL; ?>/plugins/bootstrap/js/bootstrap.bundle.min.js"></script>
<!-- ChartJS -->
<script src="<?php echo ASSETS_URL; ?>/plugins/chart.js/Chart.min.js"></script>
<!-- Sparkline -->
<script src="<?php echo ASSETS_URL; ?>/plugins/sparklines/sparkline.js"></script>
<!-- JQVMap -->
<script src="<?php echo ASSETS_URL; ?>/plugins/jqvmap/jquery.vmap.min.js"></script>
<script src="<?php echo ASSETS_URL; ?>/plugins/jqvmap/maps/jquery.vmap.usa.js"></script>
<!-- jQuery Knob Chart -->
<script src="<?php echo ASSETS_URL; ?>/plugins/jquery-knob/jquery.knob.min.js"></script>
<!-- daterangepicker -->
<script src="<?php echo ASSETS_URL; ?>/plugins/moment/moment.min.js"></script>
<script src="<?php echo ASSETS_URL; ?>/plugins/daterangepicker/daterangepicker.js"></script>
<!-- Tempusdominus Bootstrap 4 -->
<script src="<?php echo ASSETS_URL; ?>/plugins/tempusdominus-bootstrap-4/js/tempusdominus-bootstrap-4.min.js"></script>
<!-- Summernote -->
<script src="<?php echo ASSETS_URL; ?>/plugins/summernote/summernote-bs4.min.js"></script>
<!-- overlayScrollbars -->
<script src="<?php echo ASSETS_URL; ?>/plugins/overlayScrollbars/js/jquery.overlayScrollbars.min.js"></script>
<!-- AdminLTE App -->
<script src="<?php echo ASSETS_URL; ?>/dist/js/adminlte.js"></script>


<script src="<?php echo ASSETS_URL; ?>/js/realtime-reports.js"></script>
<script src="<?php echo ASSETS_URL; ?>/js/realtime-alerts.js"></script>
<script src="<?php echo ASSETS_URL; ?>/js/auto-monitor.js"></script>
<script src="<?php echo ASSETS_URL; ?>/js/realtime-alerts-list.js"></script>


<!-- Scripts adicionales específicos de la página -->
<?php if(isset($extraJs)) echo $extraJs; ?>




<script>
// Script para cargar imágenes sin procesar desde /reports/
document.addEventListener('DOMContentLoaded', function() {
    // Función para buscar imágenes sin procesar
    function fetchUnprocessedImages() {
        const container = document.getElementById('unprocessed-images-container');
        const loadingIndicator = document.getElementById('unprocessed-loading');
        
        if (!container) return;
        
        // ID del operador actual
        const operadorId = '<?php echo $alert['operator_id']; ?>';       
        const operadorDni = '<?php echo $alert['operator_dni']; ?>';
        
        // Si no hay datos del operador, mostrar mensaje
        if (!operadorId && !operadorDni) {
            container.innerHTML = '<div class="alert alert-info">No hay información del operador para buscar imágenes relacionadas.</div>';
            return;
        }
        
        // Mostrar indicador de carga
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
        }
        container.style.display = 'none';
        
        // Realizar solicitud AJAX
        fetch('../../api/alerts/report_images.php')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor');
                }
                return response.json();
            })
            .then(data => {
                // Ocultar indicador de carga
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                container.style.display = 'block';
                
                if (data.status === 'success') {
                    // Filtrar imágenes relacionadas con este operador
                    const relatedImages = data.images.filter(img => 
                        (operadorDni && img.operator.dni === operadorDni) || 
                        (operadorId && img.operator.id === operadorId)
                    );
                    
                    // Actualizar contenedor
                    if (relatedImages.length > 0) {
                        const row = document.createElement('div');
                        row.className = 'row';
                        
                        relatedImages.forEach(image => {
                            const col = document.createElement('div');
                            col.className = 'col-md-6 col-lg-4 mb-3';
                            
                            // Determinar ícono según tipo de alerta
                            let icon = 'fas fa-exclamation-triangle';
                            let alertClass = 'secondary';
                            let alertLabel = 'Otro';
                            
                            switch (image.alert_type) {
                                case 'fatigue':
                                    icon = 'fas fa-bed';
                                    alertClass = 'danger';
                                    alertLabel = 'Fatiga';
                                    break;
                                case 'yawn':
                                    icon = 'fas fa-tired';
                                    alertClass = 'warning';
                                    alertLabel = 'Bostezo';
                                    break;
                                case 'phone':
                                    icon = 'fas fa-mobile-alt';
                                    alertClass = 'info';
                                    alertLabel = 'Teléfono';
                                    break;
                                case 'smoking':
                                    icon = 'fas fa-smoking';
                                    alertClass = 'secondary';
                                    alertLabel = 'Fumando';
                                    break;
                                case 'distraction':
                                    icon = 'fas fa-eye-slash';
                                    alertClass = 'primary';
                                    alertLabel = 'Distracción';
                                    break;
                                case 'unauthorized':
                                    icon = 'fas fa-user-slash';
                                    alertClass = 'dark';
                                    alertLabel = 'No Autorizado';
                                    break;
                                case 'behavior':
                                    icon = 'fas fa-exclamation-circle';
                                    alertClass = 'info';
                                    alertLabel = 'Comportamiento';
                                    break;
                            }
                            
                            col.innerHTML = `
                                <div class="card">
                                    <div class="card-header bg-${alertClass} text-white p-2">
                                        <div class="d-flex justify-content-between align-items-center">
                                            <div><i class="${icon}"></i> ${alertLabel}</div>
                                            <small>${image.formatted_time}</small>
                                        </div>
                                    </div>
                                    <a href="../../${image.image_url}" data-toggle="lightbox">
                                        <img src="../../${image.image_url}" class="img-fluid" alt="Imagen sin procesar">
                                    </a>
                                    <div class="card-footer p-2">
                                        <small class="text-muted"><i class="fas fa-sync fa-spin"></i> En procesamiento</small>
                                    </div>
                                </div>
                            `;
                            row.appendChild(col);
                        });
                        
                        container.innerHTML = '';
                        container.appendChild(row);
                        
                        // Inicializar Lightbox para las nuevas imágenes
                        $(document).on('click', '[data-toggle="lightbox"]', function(event) {
                            event.preventDefault();
                            $(this).ekkoLightbox();
                        });
                    } else {
                        container.innerHTML = '<div class="alert alert-info">No hay imágenes en procesamiento para este operador.</div>';
                    }
                } else {
                    container.innerHTML = `<div class="alert alert-danger">Error: ${data.message}</div>`;
                }
            })
            .catch(error => {
                console.error('Error al obtener imágenes:', error);
                if (loadingIndicator) {
                    loadingIndicator.style.display = 'none';
                }
                container.style.display = 'block';
                container.innerHTML = '<div class="alert alert-danger">Error al comunicarse con el servidor. Intente nuevamente.</div>';
            });
    }
    
    // Cargar imágenes al iniciar
    fetchUnprocessedImages();
    
    // Configurar el botón de actualizar
    document.getElementById('refresh-unprocessed')?.addEventListener('click', fetchUnprocessedImages);
    
    // Actualizar automáticamente cada 30 segundos
    setInterval(fetchUnprocessedImages, 30000);
});
</script>



<!-- Script para cargar notificaciones en tiempo real -->
<script>
function cargarNotificaciones() {
    $.ajax({
        url: '<?php echo BASE_URL; ?>/api/notifications.php',        
        type: 'GET',
        dataType: 'json',
        success: function(data) {
            if (data.success) {
                $('#notification-counter').text(data.notifications.length);
                $('#notification-header').text(data.notifications.length + ' Notificaciones');
                $('#notification-container').empty();

                $.each(data.notifications, function(index, notification) {
                    var item = '<a href="pages/alerts/show.php?id=' + notification.id + '" class="dropdown-item">' +
                                 '<i class="fas ' + (notification.type === 'fatigue' ? 'fa-tired' : 
                                                      notification.type === 'distraction' ? 'fa-eye-slash' : 
                                                      notification.type === 'yawn' ? 'fa-comment-dots' : 'fa-exclamation-circle') + 
                                  ' mr-2"></i> ' + notification.message +
                                 '<span class="float-right text-muted text-sm">' + notification.time_ago + '</span>' +
                               '</a>' +
                               '<div class="dropdown-divider"></div>';
                    $('#notification-container').append(item);
                });
            }
        },
        error: function() {
            console.error('Error al cargar las notificaciones');
        }
    });
}

$(document).ready(function() {
    cargarNotificaciones();
    setInterval(cargarNotificaciones, 30000);
});

// Cargar notificaciones al inicio
$(document).ready(function() {
    cargarNotificaciones();
    
    // Actualizar notificaciones cada 30 segundos
    setInterval(cargarNotificaciones, 30000);
});
</script>



</body>
</html>