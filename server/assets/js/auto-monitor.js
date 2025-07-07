/**
 * Sistema de monitoreo automático de reportes
 * 
 * Este script ejecuta periódicamente el monitor de reportes y muestra el resultado
 * También permite ejecutar el monitor manualmente mediante un botón
 */

document.addEventListener('DOMContentLoaded', function() {
    // Configuración inicial
    const AutoMonitor = {
        // Configuración
        config: {
            apiUrl: '../../api/alerts/execute_monitor.php',  // URL del endpoint para ejecutar el monitor
            refreshInterval: 60000,                     // Intervalo para ejecución automática (60 segundos)
            enableAutoExecute: true,                        // Habilitar ejecución automática
            executeButtonSelector: '#ejecutar-ahora',      // Selector del botón para ejecutar manualmente
            statusContainerSelector: '#monitor-status-container', // Selector del contenedor de estado
            toggleAutoSelector: '#detener-auto'      // Selector del botón para activar/desactivar ejecución automática
        },
        
        // Estado
        state: {
            isExecuting: false,
            lastExecutionTime: null,
            autoExecuteIntervalId: null,
            isAutoExecuteEnabled: true
        },
        
        // Inicializar el sistema
        init: function() {
            // Registrar event listeners
            this.registerEventListeners();
            
            // Iniciar ejecución automática si está habilitada
            if (this.config.enableAutoExecute) {
                this.startAutoExecute();
            }
            
            console.log('Sistema de monitoreo automático inicializado');
        },
        
        // Registrar event listeners
        registerEventListeners: function() {
            // Botón para ejecutar manualmente
            const executeButton = document.querySelector(this.config.executeButtonSelector);
            if (executeButton) {
                executeButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    this.executeMonitor();
                });
            }
            
            // Botón para activar/desactivar ejecución automática
            const toggleAutoButton = document.querySelector(this.config.toggleAutoSelector);
            if (toggleAutoButton) {
                toggleAutoButton.addEventListener('click', (e) => {
                    e.preventDefault();
                    if (this.state.isAutoExecuteEnabled) {
                        this.stopAutoExecute();
                        toggleAutoButton.classList.remove('btn-danger');
                        toggleAutoButton.classList.add('btn-success');
                        toggleAutoButton.querySelector('i').classList.remove('fa-pause');
                        toggleAutoButton.querySelector('i').classList.add('fa-play');
                        toggleAutoButton.querySelector('span').textContent = ' Activar Auto';
                    } else {
                        this.startAutoExecute();
                        toggleAutoButton.classList.remove('btn-success');
                        toggleAutoButton.classList.add('btn-danger');
                        toggleAutoButton.querySelector('i').classList.remove('fa-play');
                        toggleAutoButton.querySelector('i').classList.add('fa-pause');
                        toggleAutoButton.querySelector('span').textContent = ' Detener Auto';
                    }
                });
            }
        },
        
        // Iniciar ejecución automática
        startAutoExecute: function() {
            if (this.state.autoExecuteIntervalId) {
                clearInterval(this.state.autoExecuteIntervalId);
            }
            
            this.state.isAutoExecuteEnabled = true;
            
            // Ejecutar una vez al iniciar
            this.executeMonitor();
            
            // Configurar intervalo para ejecuciones periódicas
            this.state.autoExecuteIntervalId = setInterval(() => {
                this.executeMonitor();
            }, this.config.refreshInterval);
            
            console.log('Ejecución automática iniciada');
        },
        
        // Detener ejecución automática
        stopAutoExecute: function() {
            if (this.state.autoExecuteIntervalId) {
                clearInterval(this.state.autoExecuteIntervalId);
                this.state.autoExecuteIntervalId = null;
            }
            
            this.state.isAutoExecuteEnabled = false;
            
            console.log('Ejecución automática detenida');
        },
        
        // Ejecutar monitor de reportes
        executeMonitor: function() {
            // Evitar ejecuciones simultáneas
            if (this.state.isExecuting) {
                console.log('Ya hay una ejecución en curso, esperando...');
                return;
            }
            
            this.state.isExecuting = true;
            
            // Actualizar UI para mostrar que está ejecutando
            this.updateStatusUI('executing');
            
            // Realizar solicitud AJAX
            fetch(this.config.apiUrl)
                .then(response => {
                    // Verificar si la respuesta es válida
                    if (!response.ok) {
                        throw new Error(`Error HTTP: ${response.status} - ${response.statusText}`);
                    }
                    
                    // Verificar el tipo de contenido
                    const contentType = response.headers.get('content-type');
                    if (!contentType || !contentType.includes('application/json')) {
                        // Leer el texto de la respuesta para depuración
                        return response.text().then(text => {
                            throw new Error(`La respuesta no es JSON válido: ${text.substring(0, 100)}...`);
                        });
                    }
                    
                    // Si todo está bien, interpretar como JSON
                    return response.json();
                })
                .then(data => {
                    // Guardar timestamp de última ejecución
                    this.state.lastExecutionTime = new Date();
                    
                    // Actualizar UI con resultado
                    if (data.status === 'success') {
                        this.updateStatusUI('success', data);
                        
                        // Si hay imágenes procesadas, actualizar también los listados
                        if (data.data && data.data.processed > 0) {
                            this.refreshLists();
                        }
                    } else {
                        this.updateStatusUI('error', data);
                    }
                })
                .catch(error => {
                    console.error('Error al ejecutar monitor:', error);
                    this.updateStatusUI('error', { message: error.message });
                })
                .finally(() => {
                    // Restablecer estado después de un tiempo
                    setTimeout(() => {
                        this.state.isExecuting = false;
                    }, 2000); // Pequeño retraso para evitar múltiples clics
                });
        },
        
        // Actualizar listados de reportes y alertas
        refreshLists: function() {
            // Actualizar lista de alertas si existe la función
            if (typeof fetchAlerts === 'function') {
                fetchAlerts(true);
            } else if (window.RealtimeAlerts && typeof window.RealtimeAlerts.fetchAlerts === 'function') {
                window.RealtimeAlerts.fetchAlerts(true);
            }
            
            // Actualizar lista de reportes si existe la función
            if (typeof fetchReports === 'function') {
                fetchReports(true);
            } else if (window.RealtimeReports && typeof window.RealtimeReports.fetchReports === 'function') {
                window.RealtimeReports.fetchReports(true);
            }
        },
        
        // Actualizar UI con estado de ejecución
        updateStatusUI: function(status, data = {}) {
            const statusContainer = document.querySelector(this.config.statusContainerSelector);
            if (!statusContainer) return;
            
            // Obtener texto de último proceso
            let lastExecutionText = this.state.lastExecutionTime ? 
                `Última ejecución: ${this.state.lastExecutionTime.toLocaleTimeString()}` : 
                'Primera ejecución';
            
            // Actualizar contenido según estado
            switch (status) {
                case 'executing':
                    statusContainer.innerHTML = `
                        <div class="alert alert-info">
                            <i class="fas fa-sync fa-spin mr-2"></i>
                            <span>Procesando reportes...</span>
                            <div class="mt-1 small">${lastExecutionText}</div>
                        </div>
                    `;
                    break;
                    
                case 'success':
                    const processed = data.data ? data.data.processed : (data.processed || 0);
                    const failed = data.data ? data.data.failed : (data.failed || 0);
                    const message = data.message || 'Proceso completado correctamente';
                    const executionTime = data.execution_time || data.data?.execution_time || '';
                    
                    statusContainer.innerHTML = `
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle mr-2"></i>
                            <span>Proceso completado ${executionTime ? `(${executionTime})` : ''}</span>
                            <div class="mt-1">
                                Procesados: <strong>${processed}</strong> | 
                                Fallidos: <strong>${failed}</strong>
                            </div>
                            <div class="mt-1 small">${message}</div>
                        </div>
                    `;
                    
                    // Ocultar después de 10 segundos si auto-execute está activado
                    if (this.state.isAutoExecuteEnabled) {
                        setTimeout(() => {
                            statusContainer.innerHTML = `
                                <div class="alert alert-light">
                                    <i class="fas fa-info-circle mr-2"></i>
                                    <span>Monitoreo automático activo</span>
                                    <div class="mt-1 small">${lastExecutionText}</div>
                                </div>
                            `;
                        }, 10000);
                    }
                    break;
                    
                case 'error':
                    statusContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle mr-2"></i>
                            <span>Error al procesar reportes</span>
                            <div class="mt-1 small">${data.message || 'Error desconocido'}</div>
                            <div class="mt-1 small">${lastExecutionText}</div>
                        </div>
                    `;
                    break;
            }
            
            // Actualizar botón de ejecución manual
            const executeButton = document.querySelector(this.config.executeButtonSelector);
            if (executeButton) {
                executeButton.disabled = (status === 'executing');
            }
        }
    };
    
    // Inicializar si existe el contenedor
    if (document.querySelector('#monitor-status-container')) {
        AutoMonitor.init();
    }
});