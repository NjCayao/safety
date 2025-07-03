<?php
// utils/password_helper.php

/**
 * Genera un hash seguro para una contraseña
 */
function generate_password_hash($password) {
    return password_hash($password, PASSWORD_DEFAULT);
}

/**
 * Verifica si una contraseña coincide con un hash
 */
function verify_password($password, $hash) {
    return password_verify($password, $hash);
}

/**
 * Genera una contraseña aleatoria
 */
function generate_random_password($length = 10) {
    $chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()';
    $password = '';
    
    for ($i = 0; $i < $length; $i++) {
        $password .= $chars[random_int(0, strlen($chars) - 1)];
    }
    
    return $password;
}

/**
 * Valida la fortaleza de una contraseña
 * Retorna true si es válida, o un mensaje de error si no lo es
 */
function validate_password_strength($password) {
    // Mínimo 8 caracteres
    if (strlen($password) < 8) {
        return 'La contraseña debe tener al menos 8 caracteres.';
    }
    
    // Debe tener al menos un número
    if (!preg_match('/[0-9]/', $password)) {
        return 'La contraseña debe incluir al menos un número.';
    }
    
    // Debe tener al menos una letra mayúscula
    if (!preg_match('/[A-Z]/', $password)) {
        return 'La contraseña debe incluir al menos una letra mayúscula.';
    }
    
    // Debe tener al menos una letra minúscula
    if (!preg_match('/[a-z]/', $password)) {
        return 'La contraseña debe incluir al menos una letra minúscula.';
    }
    
    // Opcional: verificar caracteres especiales
    if (!preg_match('/[!@#$%^&*()\-_=+{};:,<.>]/', $password)) {
        return 'La contraseña debe incluir al menos un carácter especial.';
    }
    
    return true;
}