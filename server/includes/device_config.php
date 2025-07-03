<?php
/**
 * Funciones para gestión de configuración de dispositivos
 * server/includes/device_config.php
 */

class DeviceConfigManager {
    
    /**
     * Configuración por defecto para dispositivos
     */
    private static $defaultConfig = [
        'camera' => [
            'fps' => 15,
            'width' => 640,
            'height' => 480,
            'brightness' => 0,
            'contrast' => 0,
            'saturation' => 0,
            'exposure' => -1,
            'buffer_size' => 1,
            'use_threading' => true,
            'warmup_time' => 2
        ],
        'fatigue' => [
            'eye_closed_threshold' => 1.5,
            'ear_threshold' => 0.25,
            'ear_night_adjustment' => 0.03,
            'window_size' => 600,
            'frames_to_confirm' => 2,
            'calibration_period' => 30,
            'alarm_cooldown' => 5,
            'multiple_fatigue_threshold' => 3,
            'night_mode_threshold' => 50,
            'enable_night_mode' => true
        ],
        'yawn' => [
            'mouth_threshold' => 0.7,
            'duration_threshold' => 2.5,
            'window_size' => 600,
            'frames_to_confirm' => 3,
            'alert_cooldown' => 5.0,
            'max_yawns_before_alert' => 3,
            'report_delay' => 2.0,
            'enable_auto_calibration' => true,
            'calibration_frames' => 60,
            'calibration_factor' => 0.4,
            'enable_sounds' => true
        ],
        'distraction' => [
            'rotation_threshold_day' => 2.6,
            'rotation_threshold_night' => 2.8,
            'extreme_rotation_threshold' => 2.5,
            'level1_time' => 3,
            'level2_time' => 5,
            'visibility_threshold' => 15,
            'frames_without_face_limit' => 5,
            'confidence_threshold' => 0.7,
            'prediction_buffer_size' => 10,
            'distraction_window' => 600,
            'min_frames_for_reset' => 10,
            'audio_enabled' => true,
            'level1_volume' => 0.8,
            'level2_volume' => 1.0,
            'camera_fps' => 4
        ],
        'behavior' => [
            'confidence_threshold' => 0.4,
            'night_confidence_threshold' => 0.35,
            'night_mode_threshold' => 50,
            'night_image_alpha' => 1.3,
            'night_image_beta' => 40,
            'phone_alert_threshold_1' => 3,
            'phone_alert_threshold_2' => 7,
            'cigarette_pattern_window' => 30,
            'cigarette_pattern_threshold' => 3,
            'cigarette_continuous_threshold' => 7,
            'face_proximity_factor' => 2,
            'detection_timeout' => 1.0,
            'audio_enabled' => true
        ],
        'audio' => [
            'enabled' => true,
            'volume' => 1.0,
            'frequency' => 44100,
            'size' => -16,
            'channels' => 2,
            'buffer' => 2048
        ],
        'system' => [
            'enable_gui' => false,
            'log_level' => 'INFO',
            'debug_mode' => false,
            'performance_monitoring' => true,
            'auto_optimization' => true,
            'startup_timeout' => 30,
            'module_init_timeout' => 10
        ],
        'sync' => [
            'enabled' => true,
            'auto_sync_interval' => 300,
            'batch_size' => 50,
            'connection_timeout' => 10,
            'read_timeout' => 30,
            'max_retries' => 3,
            'retry_delay' => 5,
            'max_local_events' => 10000,
            'cleanup_days' => 30
        ]
    ];

    /**
     * Rangos válidos para validación de parámetros
     */
    private static $validationRules = [
        'camera' => [
            'fps' => ['min' => 1, 'max' => 30, 'type' => 'int'],
            'width' => ['values' => [320, 640, 800, 1024, 1280], 'type' => 'int'],
            'height' => ['values' => [240, 480, 600, 768, 720], 'type' => 'int'],
            'brightness' => ['min' => -100, 'max' => 100, 'type' => 'int'],
            'contrast' => ['min' => -100, 'max' => 100, 'type' => 'int'],
            'saturation' => ['min' => -100, 'max' => 100, 'type' => 'int'],
            'buffer_size' => ['min' => 1, 'max' => 5, 'type' => 'int'],
            'warmup_time' => ['min' => 0, 'max' => 10, 'type' => 'int']
        ],
        'fatigue' => [
            'eye_closed_threshold' => ['min' => 0.5, 'max' => 5.0, 'type' => 'float'],
            'ear_threshold' => ['min' => 0.1, 'max' => 0.5, 'type' => 'float'],
            'ear_night_adjustment' => ['min' => 0.01, 'max' => 0.1, 'type' => 'float'],
            'frames_to_confirm' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'calibration_period' => ['min' => 10, 'max' => 120, 'type' => 'int'],
            'alarm_cooldown' => ['min' => 1, 'max' => 30, 'type' => 'int'],
            'multiple_fatigue_threshold' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'night_mode_threshold' => ['min' => 10, 'max' => 100, 'type' => 'int']
        ],
        'yawn' => [
            'mouth_threshold' => ['min' => 0.3, 'max' => 1.0, 'type' => 'float'],
            'duration_threshold' => ['min' => 1.0, 'max' => 5.0, 'type' => 'float'],
            'frames_to_confirm' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'alert_cooldown' => ['min' => 1.0, 'max' => 30.0, 'type' => 'float'],
            'max_yawns_before_alert' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'report_delay' => ['min' => 0.5, 'max' => 10.0, 'type' => 'float'],
            'calibration_frames' => ['min' => 30, 'max' => 120, 'type' => 'int'],
            'calibration_factor' => ['min' => 0.1, 'max' => 1.0, 'type' => 'float']
        ],
        'distraction' => [
            'rotation_threshold_day' => ['min' => 1.0, 'max' => 5.0, 'type' => 'float'],
            'rotation_threshold_night' => ['min' => 1.0, 'max' => 5.0, 'type' => 'float'],
            'extreme_rotation_threshold' => ['min' => 1.0, 'max' => 5.0, 'type' => 'float'],
            'level1_time' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'level2_time' => ['min' => 2, 'max' => 15, 'type' => 'int'],
            'visibility_threshold' => ['min' => 5, 'max' => 50, 'type' => 'int'],
            'frames_without_face_limit' => ['min' => 1, 'max' => 20, 'type' => 'int'],
            'confidence_threshold' => ['min' => 0.1, 'max' => 1.0, 'type' => 'float'],
            'level1_volume' => ['min' => 0.1, 'max' => 1.0, 'type' => 'float'],
            'level2_volume' => ['min' => 0.1, 'max' => 1.0, 'type' => 'float']
        ],
        'behavior' => [
            'confidence_threshold' => ['min' => 0.1, 'max' => 0.9, 'type' => 'float'],
            'night_confidence_threshold' => ['min' => 0.1, 'max' => 0.9, 'type' => 'float'],
            'night_mode_threshold' => ['min' => 10, 'max' => 100, 'type' => 'int'],
            'phone_alert_threshold_1' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'phone_alert_threshold_2' => ['min' => 2, 'max' => 20, 'type' => 'int'],
            'cigarette_pattern_window' => ['min' => 10, 'max' => 60, 'type' => 'int'],
            'cigarette_pattern_threshold' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'cigarette_continuous_threshold' => ['min' => 3, 'max' => 30, 'type' => 'int'],
            'detection_timeout' => ['min' => 0.5, 'max' => 5.0, 'type' => 'float']
        ],
        'audio' => [
            'volume' => ['min' => 0.0, 'max' => 1.0, 'type' => 'float'],
            'frequency' => ['values' => [22050, 44100, 48000], 'type' => 'int'],
            'channels' => ['values' => [1, 2], 'type' => 'int'],
            'buffer' => ['values' => [1024, 2048, 4096], 'type' => 'int']
        ],
        'system' => [
            'startup_timeout' => ['min' => 10, 'max' => 120, 'type' => 'int'],
            'module_init_timeout' => ['min' => 5, 'max' => 60, 'type' => 'int']
        ],
        'sync' => [
            'auto_sync_interval' => ['min' => 60, 'max' => 3600, 'type' => 'int'],
            'batch_size' => ['min' => 10, 'max' => 500, 'type' => 'int'],
            'connection_timeout' => ['min' => 5, 'max' => 60, 'type' => 'int'],
            'read_timeout' => ['min' => 10, 'max' => 300, 'type' => 'int'],
            'max_retries' => ['min' => 1, 'max' => 10, 'type' => 'int'],
            'retry_delay' => ['min' => 1, 'max' => 30, 'type' => 'int'],
            'max_local_events' => ['min' => 1000, 'max' => 50000, 'type' => 'int'],
            'cleanup_days' => ['min' => 1, 'max' => 365, 'type' => 'int']
        ]
    ];

    /**
     * Obtiene la configuración actual de un dispositivo
     */
    public static function getDeviceConfig($device_id) {
        $device = db_fetch_one(
            "SELECT config_json, config_version, config_pending, config_applied FROM devices WHERE device_id = ?",
            [$device_id]
        );
        
        if (!$device) {
            return null;
        }
        
        $config = $device['config_json'] ? json_decode($device['config_json'], true) : null;
        
        // Si no hay configuración, usar la por defecto
        if (!$config) {
            $config = self::$defaultConfig;
        }
        
        // Asegurar que todos los campos estén presentes
        $config = self::mergeWithDefaults($config);
        
        return [
            'config' => $config,
            'version' => $device['config_version'],
            'pending' => (bool)$device['config_pending'],
            'last_applied' => $device['config_applied']
        ];
    }

    /**
     * Actualiza la configuración de un dispositivo
     */
    public static function updateDeviceConfig($device_id, $new_config, $user_id, $change_summary = null) {
        try {
            // Validar la nueva configuración
            $validation_result = self::validateConfig($new_config);
            if ($validation_result !== true) {
                return ['success' => false, 'error' => $validation_result];
            }

            // Obtener configuración actual
            $current = self::getDeviceConfig($device_id);
            if (!$current) {
                return ['success' => false, 'error' => 'Dispositivo no encontrado'];
            }

            // Fusionar con configuración por defecto para asegurar completitud
            $merged_config = self::mergeWithDefaults($new_config);
            $config_json = json_encode($merged_config, JSON_PRETTY_PRINT);

            // Iniciar transacción
            $db = new Database();
            $pdo = $db->getConnection();
            $pdo->beginTransaction();

            try {
                // Actualizar configuración del dispositivo
                $update_result = db_update(
                    'devices',
                    [
                        'config_json' => $config_json,
                        'config_pending' => 1
                    ],
                    'device_id = ?',
                    [$device_id]
                );

                if ($update_result === false) {
                    throw new Exception('Error al actualizar configuración del dispositivo');
                }

                // Registrar en historial
                $history_data = [
                    'device_id' => $device_id,
                    'changed_by' => $user_id,
                    'change_type' => 'manual',
                    'config_before' => $current['config'] ? json_encode($current['config']) : null,
                    'config_after' => $config_json,
                    'changes_summary' => $change_summary ?: self::generateChangeSummary($current['config'], $merged_config),
                    'applied_successfully' => null
                ];

                $history_id = db_insert('device_config_history', $history_data);
                if (!$history_id) {
                    throw new Exception('Error al registrar historial de cambios');
                }

                // Registrar en log del sistema
                log_system_message(
                    'info',
                    "Configuración actualizada para dispositivo {$device_id}",
                    null,
                    "Cambios: " . $history_data['changes_summary']
                );

                $pdo->commit();

                return [
                    'success' => true,
                    'message' => 'Configuración actualizada correctamente',
                    'history_id' => $history_id
                ];

            } catch (Exception $e) {
                $pdo->rollBack();
                throw $e;
            }

        } catch (Exception $e) {
            return ['success' => false, 'error' => $e->getMessage()];
        }
    }

    /**
     * Valida una configuración completa
     */
    public static function validateConfig($config) {
        foreach (self::$validationRules as $section => $rules) {
            if (!isset($config[$section])) {
                continue; // Sección opcional
            }

            foreach ($rules as $param => $rule) {
                if (!isset($config[$section][$param])) {
                    continue; // Parámetro opcional
                }

                $value = $config[$section][$param];
                
                // Validar tipo
                if ($rule['type'] === 'int' && !is_int($value)) {
                    return "Parámetro {$section}.{$param} debe ser un número entero";
                }
                if ($rule['type'] === 'float' && !is_numeric($value)) {
                    return "Parámetro {$section}.{$param} debe ser un número";
                }
                if ($rule['type'] === 'bool' && !is_bool($value)) {
                    return "Parámetro {$section}.{$param} debe ser verdadero o falso";
                }

                // Validar rango
                if (isset($rule['min']) && $value < $rule['min']) {
                    return "Parámetro {$section}.{$param} debe ser mayor o igual a {$rule['min']}";
                }
                if (isset($rule['max']) && $value > $rule['max']) {
                    return "Parámetro {$section}.{$param} debe ser menor o igual a {$rule['max']}";
                }
                
                // Validar valores específicos
                if (isset($rule['values']) && !in_array($value, $rule['values'])) {
                    $valid_values = implode(', ', $rule['values']);
                    return "Parámetro {$section}.{$param} debe ser uno de: {$valid_values}";
                }
            }
        }

        return true;
    }

    /**
     * Fusiona configuración con valores por defecto
     */
    private static function mergeWithDefaults($config) {
        $merged = self::$defaultConfig;
        
        foreach ($config as $section => $params) {
            if (isset($merged[$section]) && is_array($params)) {
                $merged[$section] = array_merge($merged[$section], $params);
            } else {
                $merged[$section] = $params;
            }
        }
        
        return $merged;
    }

    /**
     * Genera un resumen legible de los cambios
     */
    private static function generateChangeSummary($old_config, $new_config) {
        $changes = [];
        
        foreach ($new_config as $section => $params) {
            if (!isset($old_config[$section])) {
                $changes[] = "Agregada sección {$section}";
                continue;
            }
            
            foreach ($params as $param => $value) {
                $old_value = $old_config[$section][$param] ?? null;
                if ($old_value !== $value) {
                    $changes[] = "{$section}.{$param}: {$old_value} → {$value}";
                }
            }
        }
        
        return empty($changes) ? 'Sin cambios' : implode(', ', $changes);
    }

    /**
     * Confirma que la configuración fue aplicada exitosamente
     */
    public static function confirmConfigApplied($device_id, $config_version, $history_id = null) {
        try {
            $db = new Database();
            $pdo = $db->getConnection();
            $pdo->beginTransaction();

            // Actualizar dispositivo
            $update_result = db_update(
                'devices',
                [
                    'config_applied' => date('Y-m-d H:i:s'),
                    'config_pending' => 0,
                    'last_config_check' => date('Y-m-d H:i:s')
                ],
                'device_id = ? AND config_version = ?',
                [$device_id, $config_version]
            );

            // Actualizar historial
            if ($history_id) {
                db_update(
                    'device_config_history',
                    [
                        'applied_successfully' => 1,
                        'applied_at' => date('Y-m-d H:i:s')
                    ],
                    'id = ?',
                    [$history_id]
                );
            }

            $pdo->commit();
            return true;

        } catch (Exception $e) {
            $pdo->rollBack();
            return false;
        }
    }

    /**
     * Reporta error en aplicación de configuración
     */
    public static function reportConfigError($device_id, $error_message, $history_id = null) {
        try {
            // Actualizar historial
            if ($history_id) {
                db_update(
                    'device_config_history',
                    [
                        'applied_successfully' => 0,
                        'error_message' => $error_message,
                        'applied_at' => date('Y-m-d H:i:s')
                    ],
                    'id = ?',
                    [$history_id]
                );
            }

            // Registrar en log del sistema
            log_system_message(
                'error',
                "Error aplicando configuración en dispositivo {$device_id}: {$error_message}",
                null,
                "Device ID: {$device_id}"
            );

            return true;

        } catch (Exception $e) {
            return false;
        }
    }

    /**
     * Obtiene todos los dispositivos con su estado de configuración
     */
    public static function getAllDevicesConfigStatus() {
        return db_fetch_all("SELECT * FROM device_config_status ORDER BY device_name");
    }

    /**
     * Obtiene historial de configuración de un dispositivo
     */
    public static function getConfigHistory($device_id, $limit = 50) {
        return db_fetch_all(
            "SELECT dch.*, u.username, u.name as user_name 
             FROM device_config_history dch 
             LEFT JOIN users u ON dch.changed_by = u.id 
             WHERE dch.device_id = ? 
             ORDER BY dch.created_at DESC 
             LIMIT ?",
            [$device_id, $limit]
        );
    }

    /**
     * Obtiene perfiles de configuración disponibles
     */
    public static function getConfigProfiles($device_type = null) {
        $sql = "SELECT * FROM device_config_profiles WHERE 1=1";
        $params = [];
        
        if ($device_type) {
            $sql .= " AND (device_type = ? OR device_type IS NULL)";
            $params[] = $device_type;
        }
        
        $sql .= " ORDER BY is_default DESC, name ASC";
        
        return db_fetch_all($sql, $params);
    }

    /**
     * Aplica un perfil de configuración a un dispositivo
     */
    public static function applyConfigProfile($device_id, $profile_id, $user_id) {
        // Obtener perfil
        $profile = db_fetch_one(
            "SELECT * FROM device_config_profiles WHERE id = ?",
            [$profile_id]
        );
        
        if (!$profile) {
            return ['success' => false, 'error' => 'Perfil no encontrado'];
        }

        $config = json_decode($profile['config_json'], true);
        if (!$config) {
            return ['success' => false, 'error' => 'Perfil tiene configuración inválida'];
        }

        return self::updateDeviceConfig(
            $device_id, 
            $config, 
            $user_id, 
            "Aplicado perfil: {$profile['name']}"
        );
    }

    /**
     * Obtiene las reglas de validación para el frontend
     */
    public static function getValidationRules() {
        return self::$validationRules;
    }

    /**
     * Obtiene la configuración por defecto
     */
    public static function getDefaultConfig() {
        return self::$defaultConfig;
    }
}
?>