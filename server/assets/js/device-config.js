/**
 * device-config.js - Sistema de validación en tiempo real para configuración de dispositivos
 * Validación dinámica, previsualización de cambios, y UX optimizada
 */

class DeviceConfigValidator {
    constructor(validationRules, defaultConfig) {
        this.validationRules = validationRules || {};
        this.defaultConfig = defaultConfig || {};
        this.currentConfig = {};
        this.originalConfig = {};
        this.pendingChanges = {};
        this.validationErrors = {};
        
        // Estado de la UI
        this.isValidating = false;
        this.hasUnsavedChanges = false;
        this.autoSaveEnabled = true;
        this.lastAutoSave = null;
        
        // Callbacks
        this.onConfigChange = null;
        this.onValidationError = null;
        this.onValidationSuccess = null;
        
        // Timers
        this.validationTimer = null;
        this.autoSaveTimer = null;
        this.previewTimer = null;
        
        this.init();
    }
    
    init() {
        console.log('🔧 DeviceConfigValidator inicializado');
        this.setupEventListeners();
        this.loadCurrentConfig();
        this.initializeValidation();
        this.setupAutoSave();
        this.initializePreview();
    }
    
    /**
     * 🎯 CONFIGURACIÓN INICIAL
     */
    setupEventListeners() {
        // Validación en tiempo real para todos los inputs
        $(document).on('input change', '.config-input', (e) => {
            this.handleInputChange(e.target);
        });
        
        // Validación para sliders
        $(document).on('input', '.config-slider', (e) => {
            this.handleSliderChange(e.target);
        });
        
        // Validación para checkboxes
        $(document).on('change', '.config-checkbox', (e) => {
            this.handleCheckboxChange(e.target);
        });
        
        // Validación para selects
        $(document).on('change', '.config-select', (e) => {
            this.handleSelectChange(e.target);
        });
        
        // Prevenir salida sin guardar
        $(window).on('beforeunload', (e) => {
            if (this.hasUnsavedChanges) {
                const message = 'Tienes cambios sin guardar. ¿Estás seguro de que quieres salir?';
                e.returnValue = message;
                return message;
            }
        });
        
        // Shortcuts de teclado
        $(document).on('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
        
        // Reset de configuración
        $(document).on('click', '.reset-config-section', (e) => {
            const section = $(e.target).data('section');
            this.resetSection(section);
        });
        
        // Aplicar valores por defecto
        $(document).on('click', '.apply-default', (e) => {
            const field = $(e.target).data('field');
            this.applyDefaultValue(field);
        });
    }
    
    loadCurrentConfig() {
        // Cargar configuración actual del formulario
        this.currentConfig = this.extractConfigFromForm();
        this.originalConfig = JSON.parse(JSON.stringify(this.currentConfig));
        console.log('📋 Configuración actual cargada:', this.currentConfig);
    }
    
    extractConfigFromForm() {
        const config = {};
        
        // Extraer configuración de todos los campos del formulario
        $('.config-input, .config-slider, .config-select').each((index, element) => {
            const $element = $(element);
            const name = $element.attr('name');
            
            if (name) {
                const [section, param] = name.split('_', 2);
                
                if (!config[section]) {
                    config[section] = {};
                }
                
                let value = $element.val();
                
                // Convertir tipos según el tipo de input
                if ($element.attr('type') === 'number' || $element.hasClass('config-slider')) {
                    value = parseFloat(value);
                } else if ($element.attr('type') === 'range') {
                    value = parseFloat(value);
                }
                
                config[section][param] = value;
            }
        });
        
        // Extraer checkboxes
        $('.config-checkbox').each((index, element) => {
            const $element = $(element);
            const name = $element.attr('name');
            
            if (name) {
                const [section, param] = name.split('_', 2);
                
                if (!config[section]) {
                    config[section] = {};
                }
                
                config[section][param] = $element.is(':checked');
            }
        });
        
        return config;
    }
    
    /**
     * 🔍 VALIDACIÓN EN TIEMPO REAL
     */
    initializeValidation() {
        // Validar configuración inicial
        this.validateAllFields();
        this.updateValidationStatus();
    }
    
    handleInputChange(input) {
        clearTimeout(this.validationTimer);
        
        this.validationTimer = setTimeout(() => {
            this.validateField(input);
            this.updateConfigState();
            this.updatePreview();
        }, 300); // Debounce de 300ms
    }
    
    handleSliderChange(slider) {
        const $slider = $(slider);
        const valueDisplay = $slider.siblings('.slider-value, .badge');
        
        // Actualizar display del valor inmediatamente
        let displayValue = $slider.val();
        const field = $slider.attr('name');
        
        // Formatear según el tipo de campo
        if (field && field.includes('threshold')) {
            displayValue += (field.includes('time') || field.includes('duration')) ? 's' : '';
        }
        
        valueDisplay.text(displayValue);
        
        // Validar después de un delay
        clearTimeout(this.validationTimer);
        this.validationTimer = setTimeout(() => {
            this.validateField(slider);
            this.updateConfigState();
            this.updatePreview();
        }, 100); // Delay más corto para sliders
    }
    
    handleCheckboxChange(checkbox) {
        this.validateField(checkbox);
        this.updateConfigState();
        this.updatePreview();
        
        // Manejar dependencias entre checkboxes
        this.handleCheckboxDependencies(checkbox);
    }
    
    handleSelectChange(select) {
        this.validateField(select);
        this.updateConfigState();
        this.updatePreview();
        
        // Manejar cambios de resolución
        if ($(select).attr('name') === 'camera_width') {
            this.updateResolutionHeight(select);
        }
    }
    
    validateField(field) {
        const $field = $(field);
        const name = $field.attr('name');
        
        if (!name) return;
        
        const [section, param] = name.split('_', 2);
        let value = $field.val();
        
        // Convertir tipo según el campo
        if ($field.attr('type') === 'number' || $field.hasClass('config-slider')) {
            value = parseFloat(value);
        } else if ($field.attr('type') === 'checkbox') {
            value = $field.is(':checked');
        }
        
        const validation = this.validateValue(section, param, value);
        
        if (validation.isValid) {
            this.clearFieldError($field);
            delete this.validationErrors[name];
        } else {
            this.showFieldError($field, validation.error);
            this.validationErrors[name] = validation.error;
        }
        
        // Actualizar configuración actual
        if (!this.currentConfig[section]) {
            this.currentConfig[section] = {};
        }
        this.currentConfig[section][param] = value;
        
        return validation.isValid;
    }
    
    validateValue(section, param, value) {
        const rules = this.validationRules[section]?.[param];
        
        if (!rules) {
            return { isValid: true };
        }
        
        // Validar tipo
        if (rules.type === 'int' && (!Number.isInteger(value) || isNaN(value))) {
            return { isValid: false, error: 'Debe ser un número entero' };
        }
        
        if (rules.type === 'float' && (isNaN(value) || typeof value !== 'number')) {
            return { isValid: false, error: 'Debe ser un número válido' };
        }
        
        if (rules.type === 'bool' && typeof value !== 'boolean') {
            return { isValid: false, error: 'Debe ser verdadero o falso' };
        }
        
        // Validar rango
        if (rules.min !== undefined && value < rules.min) {
            return { isValid: false, error: `Valor mínimo: ${rules.min}` };
        }
        
        if (rules.max !== undefined && value > rules.max) {
            return { isValid: false, error: `Valor máximo: ${rules.max}` };
        }
        
        // Validar valores específicos
        if (rules.values && !rules.values.includes(value)) {
            return { isValid: false, error: `Valores válidos: ${rules.values.join(', ')}` };
        }
        
        return { isValid: true };
    }
    
    validateAllFields() {
        let allValid = true;
        this.validationErrors = {};
        
        $('.config-input, .config-slider, .config-select, .config-checkbox').each((index, field) => {
            if (!this.validateField(field)) {
                allValid = false;
            }
        });
        
        return allValid;
    }
    
    /**
     * 🎨 INTERFAZ DE USUARIO Y FEEDBACK
     */
    showFieldError(field, message) {
        const $field = $(field);
        
        // Remover errores anteriores
        this.clearFieldError($field);
        
        // Agregar clase de error
        $field.addClass('is-invalid');
        
        // Agregar mensaje de error
        const errorDiv = $('<div class="invalid-feedback"></div>').text(message);
        $field.after(errorDiv);
        
        // Agregar icono de error si es un slider o input especial
        if ($field.hasClass('config-slider')) {
            $field.siblings('.slider-value, .badge').addClass('text-danger');
        }
    }
    
    clearFieldError(field) {
        const $field = $(field);
        
        $field.removeClass('is-invalid');
        $field.siblings('.invalid-feedback').remove();
        
        // Remover estilos de error de elementos relacionados
        if ($field.hasClass('config-slider')) {
            $field.siblings('.slider-value, .badge').removeClass('text-danger');
        }
    }
    
    updateConfigState() {
        // Detectar cambios
        this.pendingChanges = this.detectChanges();
        this.hasUnsavedChanges = Object.keys(this.pendingChanges).length > 0;
        
        // Actualizar UI
        this.updateSaveButton();
        this.updateChangesSummary();
        this.updateTabIndicators();
        
        // Callback si está definido
        if (this.onConfigChange) {
            this.onConfigChange(this.currentConfig, this.pendingChanges);
        }
    }
    
    detectChanges() {
        const changes = {};
        
        for (const section in this.currentConfig) {
            for (const param in this.currentConfig[section]) {
                const currentValue = this.currentConfig[section][param];
                const originalValue = this.originalConfig[section]?.[param];
                
                if (currentValue !== originalValue) {
                    if (!changes[section]) {
                        changes[section] = {};
                    }
                    changes[section][param] = {
                        from: originalValue,
                        to: currentValue
                    };
                }
            }
        }
        
        return changes;
    }
    
    updateSaveButton() {
        const $saveBtn = $('.btn-save-config, button[type="submit"]');
        const hasErrors = Object.keys(this.validationErrors).length > 0;
        
        if (this.hasUnsavedChanges && !hasErrors) {
            $saveBtn.removeClass('btn-secondary')
                   .addClass('btn-primary')
                   .prop('disabled', false)
                   .html('<i class="fas fa-save"></i> Guardar Cambios');
        } else if (hasErrors) {
            $saveBtn.removeClass('btn-primary btn-secondary')
                   .addClass('btn-danger')
                   .prop('disabled', true)
                   .html('<i class="fas fa-exclamation-triangle"></i> Errores de Validación');
        } else {
            $saveBtn.removeClass('btn-primary btn-danger')
                   .addClass('btn-secondary')
                   .prop('disabled', true)
                   .html('<i class="fas fa-check"></i> Sin Cambios');
        }
    }
    
    updateChangesSummary() {
        const changesCount = Object.keys(this.pendingChanges).length;
        let $summary = $('.changes-summary');
        
        if ($summary.length === 0) {
            $summary = $('<div class="changes-summary alert alert-info mt-3" style="display: none;"></div>');
            $('.card-footer').prepend($summary);
        }
        
        if (changesCount > 0) {
            const summaryText = this.generateChangesSummaryText();
            $summary.html(`
                <h6><i class="fas fa-info-circle"></i> Cambios Pendientes (${changesCount})</h6>
                <div class="changes-list">${summaryText}</div>
            `).slideDown();
        } else {
            $summary.slideUp();
        }
    }
    
    generateChangesSummaryText() {
        const summaryItems = [];
        
        for (const section in this.pendingChanges) {
            for (const param in this.pendingChanges[section]) {
                const change = this.pendingChanges[section][param];
                const friendlyName = this.getFriendlyParameterName(section, param);
                
                summaryItems.push(`
                    <div class="change-item">
                        <strong>${friendlyName}:</strong>
                        <span class="text-muted">${change.from}</span>
                        <i class="fas fa-arrow-right mx-1"></i>
                        <span class="text-primary">${change.to}</span>
                    </div>
                `);
            }
        }
        
        return summaryItems.join('');
    }
    
    updateTabIndicators() {
        // Agregar indicadores visuales en las tabs que tienen cambios
        $('.nav-link').each((index, tab) => {
            const $tab = $(tab);
            const tabId = $tab.attr('href');
            
            // Remover indicadores existentes
            $tab.find('.change-indicator').remove();
            
            // Verificar si hay cambios en esta tab
            const hasChanges = this.tabHasChanges(tabId);
            
            if (hasChanges) {
                $tab.append(' <span class="change-indicator badge badge-warning">•</span>');
            }
        });
    }
    
    tabHasChanges(tabId) {
        const sectionMapping = {
            '#camera-tab': 'camera',
            '#fatigue-tab': 'fatigue',
            '#yawn-tab': 'yawn',
            '#distraction-tab': 'distraction',
            '#behavior-tab': 'behavior',
            '#audio-tab': 'audio',
            '#system-tab': 'system',
            '#sync-tab': 'sync'
        };
        
        const section = sectionMapping[tabId];
        return section && this.pendingChanges[section];
    }
    
    /**
     * 💾 AUTO-GUARDADO Y PERSISTENCIA
     */
    setupAutoSave() {
        if (!this.autoSaveEnabled) return;
        
        // Auto-guardar en localStorage cada 30 segundos
        this.autoSaveTimer = setInterval(() => {
            if (this.hasUnsavedChanges) {
                this.saveToLocalStorage();
            }
        }, 30000);
        
        // Cargar del localStorage al iniciar
        this.loadFromLocalStorage();
    }
    
    saveToLocalStorage() {
        try {
            const deviceId = this.getDeviceId();
            const autoSaveData = {
                config: this.currentConfig,
                timestamp: new Date().toISOString(),
                pendingChanges: this.pendingChanges
            };
            
            localStorage.setItem(`device_config_autosave_${deviceId}`, JSON.stringify(autoSaveData));
            this.lastAutoSave = new Date();
            
            console.log('💾 Auto-guardado realizado');
        } catch (error) {
            console.warn('Error en auto-guardado:', error);
        }
    }
    
    loadFromLocalStorage() {
        try {
            const deviceId = this.getDeviceId();
            const saved = localStorage.getItem(`device_config_autosave_${deviceId}`);
            
            if (saved) {
                const autoSaveData = JSON.parse(saved);
                const savedTime = new Date(autoSaveData.timestamp);
                const timeDiff = new Date() - savedTime;
                
                // Solo cargar si es de hace menos de 1 hora
                if (timeDiff < 3600000) {
                    this.showAutoSaveRecoveryDialog(autoSaveData);
                }
            }
        } catch (error) {
            console.warn('Error cargando auto-guardado:', error);
        }
    }
    
    showAutoSaveRecoveryDialog(autoSaveData) {
        const savedTime = new Date(autoSaveData.timestamp).toLocaleString();
        
        const modal = $(`
            <div class="modal fade" id="autoSaveRecoveryModal" tabindex="-1">
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">
                                <i class="fas fa-history"></i> Recuperar Cambios
                            </h5>
                        </div>
                        <div class="modal-body">
                            <p>Se encontraron cambios no guardados del <strong>${savedTime}</strong>.</p>
                            <p>¿Deseas recuperar estos cambios?</p>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-dismiss="modal">
                                Descartar
                            </button>
                            <button type="button" class="btn btn-primary" id="recoverChanges">
                                <i class="fas fa-undo"></i> Recuperar
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `);
        
        $('body').append(modal);
        modal.modal('show');
        
        $('#recoverChanges').on('click', () => {
            this.applyConfigToForm(autoSaveData.config);
            modal.modal('hide');
            this.showNotification('success', 'Cambios recuperados exitosamente');
        });
        
        modal.on('hidden.bs.modal', () => {
            modal.remove();
        });
    }
    
    /**
     * 🔄 PREVISUALIZACIÓN DE CAMBIOS
     */
    initializePreview() {
        // Crear panel de previsualización si no existe
        if ($('.config-preview').length === 0) {
            this.createPreviewPanel();
        }
    }
    
    createPreviewPanel() {
        const previewPanel = $(`
            <div class="config-preview card">
                <div class="card-header">
                    <h6>
                        <i class="fas fa-eye"></i> Previsualización
                        <button class="btn btn-sm btn-outline-secondary float-right" onclick="deviceConfigValidator.togglePreview()">
                            <i class="fas fa-eye-slash"></i>
                        </button>
                    </h6>
                </div>
                <div class="card-body">
                    <div class="preview-content">
                        <p class="text-muted">No hay cambios para previsualizar</p>
                    </div>
                </div>
            </div>
        `);
        
        // Agregar al sidebar o crear uno
        if ($('.col-md-4').length > 0) {
            $('.col-md-4').append(previewPanel);
        }
    }
    
    updatePreview() {
        clearTimeout(this.previewTimer);
        
        this.previewTimer = setTimeout(() => {
            const $preview = $('.preview-content');
            
            if (Object.keys(this.pendingChanges).length === 0) {
                $preview.html('<p class="text-muted">No hay cambios para previsualizar</p>');
                return;
            }
            
            const previewHtml = this.generatePreviewHtml();
            $preview.html(previewHtml);
        }, 500);
    }
    
    generatePreviewHtml() {
        let html = '<div class="preview-changes">';
        
        for (const section in this.pendingChanges) {
            html += `<div class="preview-section">`;
            html += `<h6 class="text-primary">${this.getFriendlySectionName(section)}</h6>`;
            
            for (const param in this.pendingChanges[section]) {
                const change = this.pendingChanges[section][param];
                const friendlyName = this.getFriendlyParameterName(section, param);
                
                html += `
                    <div class="preview-item">
                        <small class="text-muted">${friendlyName}</small><br>
                        <span class="old-value text-muted">${change.from}</span>
                        <i class="fas fa-arrow-right mx-1"></i>
                        <span class="new-value text-success"><strong>${change.to}</strong></span>
                    </div>
                `;
            }
            
            html += '</div>';
        }
        
        html += '</div>';
        
        // Agregar estimación de impacto
        html += this.generateImpactEstimation();
        
        return html;
    }
    
    generateImpactEstimation() {
        const criticalParams = ['camera.fps', 'camera.width', 'camera.height'];
        const hasCriticalChanges = Object.keys(this.pendingChanges).some(section => 
            Object.keys(this.pendingChanges[section]).some(param => 
                criticalParams.includes(`${section}.${param}`)
            )
        );
        
        if (hasCriticalChanges) {
            return `
                <div class="impact-estimation mt-3 p-2 bg-warning text-dark rounded">
                    <small>
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>Impacto:</strong> Estos cambios pueden afectar el rendimiento del sistema.
                    </small>
                </div>
            `;
        }
        
        return `
            <div class="impact-estimation mt-3 p-2 bg-success text-white rounded">
                <small>
                    <i class="fas fa-check"></i>
                    <strong>Impacto:</strong> Cambios menores, aplicación segura.
                </small>
            </div>
        `;
    }
    
    /**
     * ⌨️ SHORTCUTS Y FUNCIONES AUXILIARES
     */
    handleKeyboardShortcuts(e) {
        // Ctrl+S para guardar
        if (e.ctrlKey && e.key === 's') {
            e.preventDefault();
            if (this.hasUnsavedChanges && Object.keys(this.validationErrors).length === 0) {
                $('form').submit();
            }
        }
        
        // Ctrl+R para resetear
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            this.resetAllFields();
        }
        
        // Esc para cancelar cambios
        if (e.key === 'Escape' && this.hasUnsavedChanges) {
            this.showCancelConfirmation();
        }
    }
    
    handleCheckboxDependencies(checkbox) {
        const $checkbox = $(checkbox);
        const name = $checkbox.attr('name');
        
        // Manejar dependencias específicas
        switch (name) {
            case 'fatigue_enable_night_mode':
                const nightModeEnabled = $checkbox.is(':checked');
                $('.night-mode-dependent').prop('disabled', !nightModeEnabled);
                break;
                
            case 'audio_enabled':
                const audioEnabled = $checkbox.is(':checked');
                $('.audio-dependent').prop('disabled', !audioEnabled);
                break;
                
            case 'sync_enabled':
                const syncEnabled = $checkbox.is(':checked');
                $('.sync-dependent').prop('disabled', !syncEnabled);
                break;
        }
    }
    
    updateResolutionHeight(widthSelect) {
        const $widthSelect = $(widthSelect);
        const selectedOption = $widthSelect.find('option:selected');
        const height = selectedOption.data('height');
        
        if (height) {
            $('input[name="camera_height"]').val(height);
        }
    }
    
    /**
     * 🔧 FUNCIONES DE UTILIDAD
     */
    resetSection(section) {
        if (confirm(`¿Resetear toda la sección de ${this.getFriendlySectionName(section)} a valores por defecto?`)) {
            const defaultSectionConfig = this.defaultConfig[section] || {};
            
            for (const param in defaultSectionConfig) {
                this.applyValueToField(section, param, defaultSectionConfig[param]);
            }
            
            this.validateAllFields();
            this.updateConfigState();
        }
    }
    
    applyDefaultValue(fieldName) {
        const [section, param] = fieldName.split('_', 2);
        const defaultValue = this.defaultConfig[section]?.[param];
        
        if (defaultValue !== undefined) {
            this.applyValueToField(section, param, defaultValue);
            this.validateField($(`[name="${fieldName}"]`)[0]);
            this.updateConfigState();
        }
    }
    
    applyValueToField(section, param, value) {
        const fieldName = `${section}_${param}`;
        const $field = $(`[name="${fieldName}"]`);
        
        if ($field.attr('type') === 'checkbox') {
            $field.prop('checked', !!value);
        } else {
            $field.val(value);
            
            // Actualizar display de sliders
            if ($field.hasClass('config-slider')) {
                $field.siblings('.slider-value, .badge').text(value);
            }
        }
    }
    
    applyConfigToForm(config) {
        for (const section in config) {
            for (const param in config[section]) {
                this.applyValueToField(section, param, config[section][param]);
            }
        }
        
        this.loadCurrentConfig();
        this.validateAllFields();
        this.updateConfigState();
    }
    
    resetAllFields() {
        if (confirm('¿Resetear toda la configuración a los valores originales?')) {
            this.applyConfigToForm(this.originalConfig);
            this.showNotification('info', 'Configuración reseteada');
        }
    }
    
    showCancelConfirmation() {
        if (confirm('¿Descartar todos los cambios no guardados?')) {
            this.resetAllFields();
        }
    }
    
    togglePreview() {
        $('.config-preview .card-body').slideToggle();
        const $icon = $('.config-preview .fa-eye-slash, .config-preview .fa-eye');
        $icon.toggleClass('fa-eye fa-eye-slash');
    }
    
    getFriendlySectionName(section) {
        const names = {
            camera: 'Cámara',
            fatigue: 'Detección de Fatiga',
            yawn: 'Detección de Bostezos',
            distraction: 'Detección de Distracción',
            behavior: 'Comportamientos',
            audio: 'Audio',
            system: 'Sistema',
            sync: 'Sincronización'
        };
        
        return names[section] || section;
    }
    
    getFriendlyParameterName(section, param) {
        const names = {
            // Camera
            'camera.fps': 'FPS',
            'camera.width': 'Ancho',
            'camera.height': 'Alto',
            'camera.brightness': 'Brillo',
            'camera.contrast': 'Contraste',
            
            // Fatigue
            'fatigue.eye_closed_threshold': 'Umbral Ojos Cerrados',
            'fatigue.ear_threshold': 'Umbral EAR',
            'fatigue.enable_night_mode': 'Modo Nocturno',
            
            // Audio
            'audio.enabled': 'Audio Habilitado',
            'audio.volume': 'Volumen',
            
            // System
            'system.enable_gui': 'Interfaz Gráfica',
            'system.log_level': 'Nivel de Log'
        };
        
        return names[`${section}.${param}`] || param.replace(/_/g, ' ');
    }
    
    getDeviceId() {
        // Extraer device ID de la URL o elemento del DOM
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('id') || $('[data-device-id]').data('device-id') || 'unknown';
    }
    
    updateValidationStatus() {
        const hasErrors = Object.keys(this.validationErrors).length > 0;
        
        if (this.onValidationError && hasErrors) {
            this.onValidationError(this.validationErrors);
        } else if (this.onValidationSuccess && !hasErrors) {
            this.onValidationSuccess();
        }
    }
    
    showNotification(type, message) {
        // Usar la función global si existe
        if (typeof showNotification === 'function') {
            showNotification(type, message);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }
    
    /**
     * 📋 API PÚBLICA
     */
    getValidationErrors() {
        return this.validationErrors;
    }
    
    hasErrors() {
        return Object.keys(this.validationErrors).length > 0;
    }
    
    getPendingChanges() {
        return this.pendingChanges;
    }
    
    getCurrentConfig() {
        return this.currentConfig;
    }
    
    clearAutoSave() {
        try {
            const deviceId = this.getDeviceId();
            localStorage.removeItem(`device_config_autosave_${deviceId}`);
        } catch (error) {
            console.warn('Error limpiando auto-guardado:', error);
        }
    }
    
    destroy() {
        // Limpiar timers
        clearTimeout(this.validationTimer);
        clearTimeout(this.previewTimer);
        clearInterval(this.autoSaveTimer);
        
        // Remover event listeners
        $(document).off('.deviceConfig');
        $(window).off('beforeunload');
        
        console.log('🔧 DeviceConfigValidator destruido');
    }
}

// Variable global para la instancia
let deviceConfigValidator = null;

// Inicialización automática cuando el DOM esté listo
$(document).ready(function() {
    // Solo inicializar si estamos en una página de configuración
    if ($('.config-input, .config-slider, .config-select, .config-checkbox').length > 0) {
        
        // Obtener reglas de validación del servidor (deben estar en la página)
        const validationRules = window.validationRules || {};
        const defaultConfig = window.defaultConfig || {};
        
        // Crear instancia global
        deviceConfigValidator = new DeviceConfigValidator(validationRules, defaultConfig);
        
        console.log('✅ Sistema de validación de configuración inicializado');
    }
});

// Limpiar al salir
$(window).on('unload', function() {
    if (deviceConfigValidator) {
        deviceConfigValidator.destroy();
    }
});