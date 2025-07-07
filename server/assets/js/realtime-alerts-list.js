/**
 * Sistema de actualización automática del listado de alertas
 * 
 * Este script actualiza automáticamente el listado de alertas
 * cada 10 segundos sin necesidad de recargar la página
 */

document.addEventListener('DOMContentLoaded', function() {
    // Configuración
    const config = {
        refreshInterval: 10000, // 10 segundos
        tableSelector: '.table-alertas', // Selector de la tabla de alertas
        updateUrl: '../../api/alerts/recent.php' // URL para obtener las alertas más recientes
    };

    // Variables
    let lastUpdate = new Date();
    let intervalId = null;
    let isUpdating = false;

    // Inicializar
    function init() {
        // Verificar si existe la tabla
        if (!document.querySelector(config.tableSelector)) {
            console.log('No se encontró la tabla de alertas');
            return;
        }

        // Iniciar actualización automática
        startAutoRefresh();

        // Agregar indicador de actualización a la tabla
        addRefreshIndicator();

        console.log('Sistema de actualización de alertas inicializado');
    }

    // Iniciar actualización automática
    function startAutoRefresh() {
        // Detener intervalo anterior si existe
        if (intervalId) {
            clearInterval(intervalId);
        }

        // Configurar intervalo para actualizaciones periódicas
        intervalId = setInterval(fetchAlerts, config.refreshInterval);
    }

    // Agregar indicador de actualización
    function addRefreshIndicator() {
        const table = document.querySelector(config.tableSelector);
        if (!table) return;

        // Crear indicador
        const indicator = document.createElement('div');
        indicator.className = 'alert-refresh-indicator';
        indicator.innerHTML = `
            <div class="d-flex align-items-center justify-content-end mb-2">
                <span class="mr-2 text-muted" id="last-update-time"></span>
                <div class="refresh-spinner mr-2" id="refresh-spinner" style="display: none;">
                    <i class="fas fa-sync fa-spin"></i>
                </div>
                <button class="btn btn-sm btn-outline-info" id="manual-refresh">
                    <i class="fas fa-sync"></i> Actualizar
                </button>
            </div>
        `;

        // Insertar antes de la tabla
        table.parentNode.insertBefore(indicator, table);

        // Manejar clic en botón de actualización manual
        document.getElementById('manual-refresh').addEventListener('click', function() {
            fetchAlerts(true);
        });

        // Actualizar hora de última actualización
        updateLastUpdateTime();
    }

    // Actualizar tiempo de última actualización
    function updateLastUpdateTime() {
        const timeElement = document.getElementById('last-update-time');
        if (timeElement) {
            timeElement.textContent = `Última actualización: ${lastUpdate.toLocaleTimeString()}`;
        }
    }

    // Consultar alertas nuevas
    function fetchAlerts(showSpinner = false) {
        // Evitar actualizaciones simultáneas
        if (isUpdating) return;
        isUpdating = true;

        // Mostrar indicador de actualización
        if (showSpinner) {
            const spinner = document.getElementById('refresh-spinner');
            if (spinner) spinner.style.display = 'block';
        }

        // Realizar solicitud AJAX
        fetch(config.updateUrl)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success') {
                    updateTable(data.alerts);
                }
                lastUpdate = new Date();
                updateLastUpdateTime();
            })
            .catch(error => {
                console.error('Error al obtener alertas:', error);
            })
            .finally(() => {
                // Ocultar indicador
                const spinner = document.getElementById('refresh-spinner');
                if (spinner) spinner.style.display = 'none';
                isUpdating = false;
            });
    }

    // Actualizar tabla con las nuevas alertas
    function updateTable(alerts) {
        const table = document.querySelector(config.tableSelector);
        if (!table) return;

        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        // Si no hay alertas, mostrar mensaje
        if (alerts.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-center">No se encontraron alertas</td></tr>';
            return;
        }

        // Crear HTML para cada alerta
        let html = '';
        alerts.forEach(alert => {
            // Determinar color del badge según el tipo y estado
            const badgeClass = alert.acknowledged ? 'success' : getBadgeClass(alert.type);
            
            // Agregar clase especial para alertas de dispositivos
            const rowClass = alert.type === 'device_error' ? 'table-warning' : '';
            
            html += `
                <tr class="${rowClass}">
                    <td>${alert.id}</td>
                    <td>
                        <span class="badge badge-${badgeClass}">
                            ${alert.type_label}
                        </span>
                    </td>
                    <td>${alert.operator?.name || 'N/A'}</td>
                    <td>${alert.machine?.name || 'N/A'}</td>
                    <td>
                        ${alert.device_id || 'N/A'}
                        ${alert.device_name ? `<br><small class="text-muted">${alert.device_name}</small>` : ''}
                    </td>
                    <td>${alert.timestamp}</td>
                    <td>
                        ${alert.acknowledged ? `
                            <span class="badge badge-success">Atendida</span>
                            ${alert.acknowledged_by ? `
                                <small class="text-muted">por ${alert.acknowledged_by}</small>
                            ` : ''}
                        ` : `
                            <span class="badge badge-danger">Pendiente</span>
                        `}
                    </td>
                    
                    <td>
                        <div class="btn-group">
                            <a href="view.php?id=${alert.id}" class="btn btn-sm btn-info">
                                <i class="fas fa-eye"></i>
                            </a>
                            ${!alert.acknowledged ? `
                            <a href="acknowledge.php?id=${alert.id}" class="btn btn-sm btn-success">
                                <i class="fas fa-check"></i>
                            </a>
                            ` : ''}
                        </div>
                    </td>
                </tr>
            `;
        });

        // Actualizar contenido de la tabla
        tbody.innerHTML = html;
    }

    // Obtener clase de badge según el tipo de alerta
    function getBadgeClass(type) {
        const classes = {
            'fatigue': 'danger',
            'yawn': 'warning',
            'phone': 'info',
            'smoking': 'secondary',
            'distraction': 'primary',
            'unauthorized': 'dark',
            'behavior': 'info',
            'device_error': 'warning', // NUEVO: para errores de dispositivo
            'other': 'light'
        };
        return classes[type] || 'secondary';
    }

    // Inicializar sistema
    init();
});