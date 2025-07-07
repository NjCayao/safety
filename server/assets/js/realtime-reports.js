/**
 * Sistema de Monitoreo de Reportes en Tiempo Real
 * 
 * Este script consulta periódicamente el API para obtener nuevas imágenes
 * desde la carpeta /reports/ y actualiza la interfaz cada 10 segundos
 */

document.addEventListener('DOMContentLoaded', function() {
    // Configuración inicial
    const RealtimeReports = {
        // Configuración
        config: {
            apiUrl: '../../api/alerts/report_images.php',  // URL del endpoint de imágenes de reportes
            executeMonitorUrl: '../../scripts/monitor_reports.php', // URL para ejecutar el monitor
            refreshInterval: 10000,                       // Intervalo de actualización en milisegundos (10 segundos)
            maxReports: 10,                               // Número máximo de reportes a mostrar
            containerSelector: '#reportes-container',     // Selector del contenedor de reportes
            counterSelector: '.reportes-counter',         // Selector del contador de reportes
            autoUpdateCheckboxSelector: '#auto-checkbox', // Selector del checkbox de actualización automática
            updateButtonSelector: '#actualizar-ahora',    // Selector del botón de actualización manual
            countdownSelector: '.countdown-timer'         // Selector del contador de tiempo
        },
        
        // Estado
        state: {
            isPolling: false,
            pollingIntervalId: null,
            autoUpdateEnabled: true,
            reportCount: 0,
            reportsData: [],
            lastUpdateTime: null,
            countdownIntervalId: null,
            countdownValue: 10,
            isExecutingMonitor: false
        },
        
        // Inicializar el sistema
        init: function() {
            console.log('Inicializando sistema de monitoreo de reportes en tiempo real');
            
            // Obtener referencias a elementos DOM
            this.container = document.querySelector(this.config.containerSelector);
            this.counter = document.querySelector(this.config.counterSelector);
            this.updateButton = document.querySelector(this.config.updateButtonSelector);
            this.autoUpdateCheckbox = document.querySelector(this.config.autoUpdateCheckboxSelector);
            
            // Verificar si los elementos existen
            if (!this.container) {
                console.error('No se encontró el contenedor de reportes:', this.config.containerSelector);
                return;
            }
            
            // Inicializar el contador
            this.updateCounter(0);
            
            // Registrar event listeners
            this.registerEventListeners();
            
            // Iniciar actualización automática
            this.startAutoUpdate();
            
            // Ejecutar el monitor de reportes inicialmente
            this.executeMonitor();
            
            console.log('Sistema de monitoreo de reportes inicializado');
        },
        
        // Registrar event listeners
        registerEventListeners: function() {
            // Botón de actualización manual
            if (this.updateButton) {
                this.updateButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.executeMonitor();
                    this.fetchReports(true);
                });
            }
            
            // Checkbox de actualización automática
            if (this.autoUpdateCheckbox) {
                this.autoUpdateCheckbox.addEventListener('change', (e) => {
                    this.state.autoUpdateEnabled = e.target.checked;
                    if (this.state.autoUpdateEnabled) {
                        this.startAutoUpdate();
                    } else {
                        this.stopAutoUpdate();
                    }
                });
            }
        },
        
        // Iniciar actualización automática
        startAutoUpdate: function() {
            if (this.state.isPolling) {
                return;
            }
            
            this.state.isPolling = true;
            this.state.autoUpdateEnabled = true;
            
            // Actualizar UI
            if (this.autoUpdateCheckbox) {
                this.autoUpdateCheckbox.checked = true;
            }
            
            // Hacer una consulta inicial
            this.fetchReports();
            
            // Configurar intervalo para consultas periódicas
            this.state.pollingIntervalId = setInterval(() => {
                this.fetchReports();
            }, this.config.refreshInterval);
            
            // Iniciar contador regresivo
            this.startCountdown();
            
            console.log('Actualización automática iniciada');
        },
        
        // Detener actualización automática
        stopAutoUpdate: function() {
            if (!this.state.isPolling) {
                return;
            }
            
            clearInterval(this.state.pollingIntervalId);
            this.state.pollingIntervalId = null;
            this.state.isPolling = false;
            
            // Detener contador regresivo
            this.stopCountdown();
            
            console.log('Actualización automática detenida');
        },
        
        // Iniciar contador regresivo
        startCountdown: function() {
            // Detener contador anterior si existe
            this.stopCountdown();
            
            // Inicializar contador
            this.state.countdownValue = this.config.refreshInterval / 1000;
            this.updateCountdown();
            
            // Iniciar intervalo de actualización
            this.state.countdownIntervalId = setInterval(() => {
                this.state.countdownValue--;
                this.updateCountdown();
                
                if (this.state.countdownValue <= 0) {
                    this.state.countdownValue = this.config.refreshInterval / 1000;
                }
            }, 1000);
        },
        
        // Detener contador regresivo
        stopCountdown: function() {
            if (this.state.countdownIntervalId) {
                clearInterval(this.state.countdownIntervalId);
                this.state.countdownIntervalId = null;
            }
            
            const countdownElement = document.querySelector(this.config.countdownSelector);
            if (countdownElement) {
                countdownElement.textContent = '';
            }
        },
        
        // Actualizar contador regresivo
        updateCountdown: function() {
            const countdownElement = document.querySelector(this.config.countdownSelector);
            if (countdownElement) {
                countdownElement.textContent = this.state.countdownValue;
            }
        },
        
        // Ejecutar el monitor de reportes
        executeMonitor: function() {
            // Evitar ejecuciones simultáneas
            if (this.state.isExecutingMonitor) {
                console.log('Ya hay una ejecución del monitor en curso, esperando...');
                return;
            }
            
            this.state.isExecutingMonitor = true;
            
            // Realizar solicitud AJAX
            fetch(this.config.executeMonitorUrl)
                .then(response => {
                    return response.text(); // Capturar la salida como texto
                })
                .then(data => {
                    console.log('Monitor ejecutado, respuesta:', data);
                    
                    // Actualizar la vista después de ejecutar el monitor
                    this.fetchReports(true);
                })
                .catch(error => {
                    console.error('Error al ejecutar monitor:', error);
                })
                .finally(() => {
                    this.state.isExecutingMonitor = false;
                });
        },
        
        // Consultar reportes nuevos
        fetchReports: function(forceUpdate = false) {
            // Mostrar indicador de actualización
            if (this.container) {
                this.container.classList.add('actualizando');
            }
            
            // Construir URL con parámetros
            const params = new URLSearchParams({
                limit: this.config.maxReports
            });
            
            // Mostrar mensaje de actualización
            const loadingElement = document.querySelector('#actualizando-mensaje');
            if (loadingElement) {
                loadingElement.style.display = 'block';
            }
            
            // Realizar solicitud AJAX
            fetch(`${this.config.apiUrl}?${params.toString()}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Error en la respuesta del servidor');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.status === 'success') {
                        this.processReports(data.images);
                    } else {
                        console.error('Error en la respuesta:', data.message);
                    }
                })
                .catch(error => {
                    console.error('Error al obtener reportes:', error);
                })
                .finally(() => {
                    // Ocultar indicador de actualización
                    if (this.container) {
                        this.container.classList.remove('actualizando');
                    }
                    
                    // Ocultar mensaje de actualización
                    if (loadingElement) {
                        loadingElement.style.display = 'none';
                    }
                    
                    // Actualizar hora de última actualización
                    this.state.lastUpdateTime = new Date();
                    
                    // Reiniciar contador regresivo
                    if (this.state.isPolling) {
                        this.startCountdown();
                    }
                });
        },
        
        // Procesar reportes
        processReports: function(reports) {
            // Verificar si hay cambios en los reportes
            const hasChanges = JSON.stringify(this.state.reportsData) !== JSON.stringify(reports);
            
            // Actualizar datos solo si hay cambios o es la primera carga
            if (hasChanges || this.state.reportsData.length === 0) {
                // Guardar datos actualizados
                this.state.reportsData = reports;
                
                // Actualizar contador
                this.updateCounter(reports.length);
                
                // Actualizar interfaz
                this.updateReportsList(reports);
            }
        },
        
        // Actualizar contador de reportes
        updateCounter: function(count) {
            this.state.reportCount = count;
            
            // Actualizar contador en la UI
            if (this.counter) {
                this.counter.textContent = count;
            }
        },
        
        // Actualizar lista de reportes en la interfaz
        updateReportsList: function(reports) {
            if (!this.container) return;
            
            // Si no hay reportes, mostrar mensaje
            if (reports.length === 0) {
                this.container.innerHTML = '<div class="text-center p-3">No hay reportes pendientes de procesar</div>';
                return;
            }
            
            // Crear contenido HTML
            let html = '';
            
            // Agregar cada reporte
            reports.forEach(report => {
                // Obtener tipo de alerta con formato
                const alertType = this.getFormattedAlertType(report.alert_type, report.alert_type_label, report.alert_type_class);
                
                // Crear HTML para el reporte
                html += `
                    <div class="alert alert-warning mb-3">
                        <div class="row">
                            <div class="col-md-3 text-center">
                                <img src="${report.image_url}" class="img-fluid img-thumbnail mb-2" alt="Reporte" style="max-height: 150px;">
                            </div>
                            <div class="col-md-9">
                                <div class="d-flex justify-content-between align-items-start">
                                    <div>
                                        <div class="mb-2">${alertType}</div>
                                        <strong>Operador:</strong> ${report.operator.name} 
                                        <small class="text-muted">(DNI: ${report.operator.dni})</small><br>
                                        <strong>Máquina:</strong> ${report.machine.name}<br>
                                        <strong>Fecha:</strong> ${report.formatted_time}
                                    </div>
                                    <span class="badge badge-warning">
                                        <i class="fas fa-sync fa-spin"></i> Procesando
                                    </span>
                                </div>
                                ${report.txt_content ? `
                                <div class="mt-2">
                                    <small class="text-muted">${this.formatTxtContent(report.txt_content)}</small>
                                </div>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                `;
            });
            
            // Actualizar el contenedor
            this.container.innerHTML = html;
        },
        
        // Formatear el contenido del archivo TXT
        formatTxtContent: function(content) {
            // Escapar HTML para evitar problemas de seguridad
            const escaped = this.escapeHtml(content);
            // Reemplazar saltos de línea con <br>
            return escaped.replace(/\n/g, '<br>');
        },
        
        // Escapar HTML
        escapeHtml: function(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },
        
        // Obtener formato HTML para un tipo de alerta
        getFormattedAlertType: function(type, label, cssClass) {
            // Definir íconos para cada tipo de alerta
            const icons = {
                'fatigue': 'fas fa-bed',
                'yawn': 'fas fa-tired',
                'phone': 'fas fa-mobile-alt',
                'smoking': 'fas fa-smoking',
                'distraction': 'fas fa-eye-slash',
                'unauthorized': 'fas fa-user-slash',
                'behavior': 'fas fa-exclamation-circle',
                'other': 'fas fa-question-circle'
            };
            
            // Obtener ícono correspondiente o usar el predeterminado
            const icon = icons[type] || icons.other;
            
            // Devolver formato HTML
            return `<span class="badge badge-${cssClass || 'secondary'}"><i class="${icon}"></i> ${label || type}</span>`;
        }
    };
    
    // Verificar si el contenedor existe antes de inicializar
    if (document.querySelector('#reportes-container')) {
        RealtimeReports.init();
    }
});