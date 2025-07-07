/**
 * Sistema de Alertas en Tiempo Real
 * 
 * Este script consulta periódicamente el API para obtener nuevas alertas
 * y actualiza el dashboard sin necesidad de recargar la página
 */

// Configuración inicial
const RealtimeAlerts = {
    // Configuración
    config: {
        apiUrl: '/safety_system/server/api/alerts/recent.php',  // URL del endpoint de alertas recientes
        reportImagesApiUrl: '/safety_system/server/api/alerts/report_images.php', // URL del endpoint de imágenes de reportes
        pollingInterval: 30000,                                // Intervalo de actualización en milisegundos (30 segundos)
        reportsPollingInterval: 5000,                          // Intervalo para revisar nuevos reportes (5 segundos)
        maxAlerts: 10,                                         // Número máximo de alertas a mostrar
        maxReportImages: 5,                                    // Número máximo de imágenes de reportes a mostrar
        notificationSound: '/safety_system/server/assets/audio/alert.mp3', // Sonido de notificación
        lastId: 0,                                             // ID de la última alerta vista
        containerSelector: '#realtime-alerts-container',       // Selector del contenedor de alertas
        reportsContainerSelector: '#realtime-reports-container', // Selector del contenedor de reportes
        counterSelector: '#alerts-counter',                    // Selector del contador de alertas
        reportsCounterSelector: '#reports-counter',            // Selector del contador de reportes sin procesar
        acknowledgedFilter: 0                                  // 0 = pendientes, 1 = atendidas, -1 = todas
    },
    
    // Estado
    state: {
        isPolling: false,
        isPollingReports: false,
        pollingIntervalId: null,
        reportsPollingIntervalId: null,
        soundEnabled: true,
        alertCount: 0,
        reportCount: 0,
        alertsData: [],
        reportsData: []
    },
    
    // Inicializar el sistema
    init: function(config = {}) {
        // Combinar configuración por defecto con la proporcionada
        this.config = {...this.config, ...config};
        
        // Inicializar el contador de alertas
        this.updateCounter(0);
        
        // Inicializar notificaciones de audio
        this.initAudio();
        
        // Iniciar el polling de alertas
        this.startPolling();
        
        // Iniciar el polling de reportes
        this.startReportsPolling();
        
        // Registrar event listeners
        this.registerEventListeners();
        
        console.log('Sistema de Alertas en Tiempo Real inicializado');
    },
    
    // Inicializar el sistema de audio
    initAudio: function() {
        this.alertSound = new Audio(this.config.notificationSound);
        this.alertSound.load();
    },
    
    // Registrar event listeners
    registerEventListeners: function() {
        // Ejemplo: Botón para habilitar/deshabilitar sonido
        document.querySelector('#toggle-sound')?.addEventListener('click', () => {
            this.state.soundEnabled = !this.state.soundEnabled;
            // Actualizar UI según corresponda
        });
        
        // Ejemplo: Filtrar alertas por estado
        document.querySelector('#filter-acknowledged')?.addEventListener('change', (e) => {
            this.config.acknowledgedFilter = parseInt(e.target.value);
            this.fetchAlerts(true); // Forzar actualización inmediata
        });
    },
    
    // Iniciar el polling
    startPolling: function() {
        if (this.state.isPolling) return;
        
        this.state.isPolling = true;
        
        // Hacer una consulta inicial
        this.fetchAlerts();
        
        // Configurar intervalo para consultas periódicas
        this.state.pollingIntervalId = setInterval(() => {
            this.fetchAlerts();
        }, this.config.pollingInterval);
    },
    
    // Detener el polling
    stopPolling: function() {
        if (!this.state.isPolling) return;
        
        clearInterval(this.state.pollingIntervalId);
        this.state.pollingIntervalId = null;
        this.state.isPolling = false;
    },
    
    // Iniciar el polling de reportes
    startReportsPolling: function() {
        if (this.state.isPollingReports) return;
        
        this.state.isPollingReports = true;
        
        // Hacer una consulta inicial
        this.fetchReportImages();
        
        // Configurar intervalo para consultas periódicas
        this.state.reportsPollingIntervalId = setInterval(() => {
            this.fetchReportImages();
        }, this.config.reportsPollingInterval);
    },
    
    // Detener el polling de reportes
    stopReportsPolling: function() {
        if (!this.state.isPollingReports) return;
        
        clearInterval(this.state.reportsPollingIntervalId);
        this.state.reportsPollingIntervalId = null;
        this.state.isPollingReports = false;
    },
    
    // Consultar alertas recientes
    fetchAlerts: function(forceRefresh = false) {
        // Construir URL con parámetros
        const params = new URLSearchParams({
            last_id: forceRefresh ? 0 : this.config.lastId,
            limit: this.config.maxAlerts,
            acknowledged: this.config.acknowledgedFilter
        });
        
        // Realizar solicitud AJAX
        fetch(`${this.config.apiUrl}?${params.toString()}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success' && data.alerts.length > 0) {
                    this.processNewAlerts(data.alerts);
                }
            })
            .catch(error => {
                console.error('Error al obtener alertas:', error);
            });
    },
    
    // Consultar imágenes de reportes en tiempo real
    fetchReportImages: function() {
        // Construir URL con parámetros
        const params = new URLSearchParams({
            limit: this.config.maxReportImages
        });
        
        // Realizar solicitud AJAX
        fetch(`${this.config.reportImagesApiUrl}?${params.toString()}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Error en la respuesta del servidor');
                }
                return response.json();
            })
            .then(data => {
                if (data.status === 'success' && data.images.length > 0) {
                    this.processNewReportImages(data.images);
                } else if (data.status === 'success' && data.images.length === 0) {
                    // No hay nuevas imágenes, actualizar UI
                    const container = document.querySelector(this.config.reportsContainerSelector);
                    if (container && this.state.reportCount !== 0) {
                        container.innerHTML = '<div class="text-center p-3">No hay reportes nuevos pendientes de procesar</div>';
                        this.updateReportsCounter(0);
                    }
                }
            })
            .catch(error => {
                console.error('Error al obtener imágenes de reportes:', error);
            });
    },
    
    // Procesar nuevas alertas
    processNewAlerts: function(alerts) {
        // Actualizar última ID vista
        if (alerts.length > 0) {
            this.config.lastId = Math.max(...alerts.map(alert => alert.id));
        }
        
        // Actualizar datos de alertas
        this.state.alertsData = alerts;
        
        // Actualizar contador
        this.updateCounter(alerts.length);
        
        // Actualizar interfaz
        this.updateAlertsList(alerts);
        
        // Reproducir sonido si hay nuevas alertas pendientes
        if (this.state.soundEnabled && alerts.some(alert => !alert.acknowledged)) {
            this.playAlertSound();
        }
    },
    
    // Procesar nuevas imágenes de reportes
    processNewReportImages: function(images) {
        // Actualizar datos de reportes
        this.state.reportsData = images;
        
        // Actualizar contador
        this.updateReportsCounter(images.length);
        
        // Actualizar interfaz
        this.updateReportsList(images);
        
        // Reproducir sonido si hay nuevos reportes y está habilitado
        if (this.state.soundEnabled && images.length > 0) {
            this.playAlertSound();
        }
    },
    
    // Actualizar contador de alertas
    updateCounter: function(count) {
        const counter = document.querySelector(this.config.counterSelector);
        if (counter) {
            this.state.alertCount = count;
            counter.textContent = count;
            
            // Destacar el contador si hay alertas
            if (count > 0) {
                counter.classList.add('badge-danger');
            } else {
                counter.classList.remove('badge-danger');
            }
        }
    },
    
    // Actualizar contador de reportes
    updateReportsCounter: function(count) {
        const counter = document.querySelector(this.config.reportsCounterSelector);
        if (counter) {
            this.state.reportCount = count;
            counter.textContent = count;
            
            // Destacar el contador si hay reportes
            if (count > 0) {
                counter.classList.add('badge-warning');
            } else {
                counter.classList.remove('badge-warning');
            }
        }
    },
    
    // Actualizar lista de alertas en la interfaz
    updateAlertsList: function(alerts) {
        const container = document.querySelector(this.config.containerSelector);
        if (!container) return;
        
        // Limpiar contenedor
        container.innerHTML = '';
        
        // Si no hay alertas, mostrar mensaje
        if (alerts.length === 0) {
            container.innerHTML = '<div class="text-center p-3">No hay alertas recientes</div>';
            return;
        }
        
        // Agregar cada alerta al contenedor
        alerts.forEach(alert => {
            const alertElement = this.createAlertElement(alert);
            container.appendChild(alertElement);
        });
    },
    
    // Actualizar lista de reportes en la interfaz
    updateReportsList: function(images) {
        const container = document.querySelector(this.config.reportsContainerSelector);
        if (!container) return;
        
        // Limpiar contenedor
        container.innerHTML = '';
        
        // Si no hay imágenes, mostrar mensaje
        if (images.length === 0) {
            container.innerHTML = '<div class="text-center p-3">No hay reportes nuevos pendientes de procesar</div>';
            return;
        }
        
        // Agregar cada imagen al contenedor
        images.forEach(image => {
            const reportElement = this.createReportElement(image);
            container.appendChild(reportElement);
        });
    },
    
    // Crear elemento HTML para una alerta
    createAlertElement: function(alert) {
        const div = document.createElement('div');
        div.className = `alert ${alert.acknowledged ? 'alert-success' : 'alert-danger'} alert-dismissible fade show`;
        div.setAttribute('role', 'alert');
        div.setAttribute('data-alert-id', alert.id);
        
        // Determinar ícono según tipo de alerta
        let icon = 'fas fa-exclamation-triangle';
        
        switch (alert.type) {
            case 'fatigue':
                icon = 'fas fa-bed';
                break;
            case 'yawn':
                icon = 'fas fa-tired';
                break;
            case 'phone':
                icon = 'fas fa-mobile-alt';
                break;
            case 'smoking':
                icon = 'fas fa-smoking';
                break;
            case 'distraction':
                icon = 'fas fa-eye-slash';
                break;
            case 'unauthorized':
                icon = 'fas fa-user-slash';
                break;
            case 'behavior':
                icon = 'fas fa-exclamation-circle';
                break;
        }
        
        // Construir contenido HTML
        div.innerHTML = `
            <div class="media">
                ${alert.image_url ? `
                <img src="${alert.image_url}" class="mr-3" alt="Alerta" style="width: 64px; height: 64px; object-fit: cover;">
                ` : `
                <div class="mr-3 d-flex align-items-center justify-content-center bg-light" style="width: 64px; height: 64px;">
                    <i class="${icon} fa-2x"></i>
                </div>
                `}
                <div class="media-body">
                    <h5 class="mt-0">
                        <span class="badge badge-${alert.acknowledged ? 'success' : 'danger'}">
                            ${alert.type_label}
                        </span>
                        <small class="text-muted">${alert.timestamp}</small>
                    </h5>
                    <p>
                        <strong>Operador:</strong> ${alert.operator.name} <br>
                        <strong>Máquina:</strong> ${alert.machine.name || 'No asignada'} <br>
                        ${alert.details ? `<small>${alert.details}</small>` : ''}
                    </p>
                    <a href="${alert.view_url}" class="btn btn-sm btn-info">
                        <i class="fas fa-eye"></i> Ver detalles
                    </a>
                    ${!alert.acknowledged ? `
                    <a href="${alert.view_url}" class="btn btn-sm btn-success">
                        <i class="fas fa-check"></i> Atender
                    </a>
                    ` : ''}
                </div>
            </div>
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                <span aria-hidden="true">&times;</span>
            </button>
        `;
        
        return div;
    },
    
    // Crear elemento HTML para un reporte
    createReportElement: function(report) {
        const div = document.createElement('div');
        div.className = `alert alert-warning alert-dismissible fade show`;
        div.setAttribute('role', 'alert');
        div.setAttribute('data-report-filename', report.base_filename);
        
        // Determinar ícono según tipo de alerta
        let icon = 'fas fa-exclamation-triangle';
        
        switch (report.alert_type) {
            case 'fatigue':
                icon = 'fas fa-bed';
                break;
            case 'yawn':
                icon = 'fas fa-tired';
                break;
            case 'phone':
                icon = 'fas fa-mobile-alt';
                break;
            case 'smoking':
                icon = 'fas fa-smoking';
                break;
            case 'distraction':
                icon = 'fas fa-eye-slash';
                break;
            case 'unauthorized':
                icon = 'fas fa-user-slash';
                break;
            case 'behavior':
                icon = 'fas fa-exclamation-circle';
                break;
        }
        
        // Construir contenido HTML
        div.innerHTML = `
            <div class="media">
                <img src="${report.image_data}" class="mr-3" alt="Reporte" style="width: 80px; height: 64px; object-fit: cover;">
                <div class="media-body">
                    <h5 class="mt-0">
                        <span class="badge badge-warning">
                            <i class="${icon}"></i> ${report.alert_type_label}
                        </span>
                        <small class="text-muted">${report.formatted_time}</small>
                        <span class="badge badge-info">Procesando</span>
                    </h5>
                    <p>
                        <strong>Operador:</strong> ${report.operator.name} (DNI: ${report.operator.dni})<br>
                        ${report.txt_content ? `<small>Detalles: ${report.txt_content.substring(0, 100)}${report.txt_content.length > 100 ? '...' : ''}</small>` : ''}
                    </p>
                </div>
            </div>
            <button type="button" class="close" data-dismiss="alert" aria-label="Close">
                <span aria-hidden="true">&times;</span>
            </button>
        `;
        
        return div;
    },
    
    // Reproducir sonido de alerta
    playAlertSound: function() {
        if (this.alertSound && this.state.soundEnabled) {
            this.alertSound.play().catch(e => {
                // Manejar error de reproducción (por ejemplo, el usuario no ha interactuado con la página)
                console.warn('No se pudo reproducir el sonido de alerta:', e);
            });
        }
    }
};

// Inicializar cuando el DOM esté cargado
document.addEventListener('DOMContentLoaded', function() {
    RealtimeAlerts.init();
});