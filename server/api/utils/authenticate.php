<?php
// server/api/utils/authenticate.php

require_once __DIR__ . '/JwtHandler.php';
require_once __DIR__ . '/Response.php';

function authenticate() {
    $headers = getallheaders();
    $authHeader = isset($headers['Authorization']) ? $headers['Authorization'] : '';
    
    // Verificar si existe el encabezado de autorización
    if (!$authHeader || !preg_match('/Bearer\s(\S+)/', $authHeader, $matches)) {
        echo Response::error('No autorizado: Token no proporcionado', 401);
        exit();
    }

    $jwt = $matches[1];
    $jwtHandler = new JwtHandler();
    $payload = $jwtHandler->decode($jwt);
    
    if (!$payload) {
        echo Response::error('No autorizado: Token inválido o expirado', 401);
        exit();
    }
    
    // Devolver la información del dispositivo para uso en los endpoints
    return $payload['data'];
}
?>