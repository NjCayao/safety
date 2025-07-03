<?php
// server/api/utils/JwtHandler.php

class JwtHandler {
    protected $jwt_secret;
    protected $token;
    protected $issuedAt;
    protected $expire;
    protected $jwt;

    public function __construct() {
        // Cargamos el secreto desde config o .env
        $this->jwt_secret = isset($_ENV['JWT_SECRET']) ? $_ENV['JWT_SECRET'] : 'tu_secreto_seguro_aqui';
        $this->issuedAt = time();
        
        // Token válido por 12 horas (ajustable según necesidades)
        $this->expire = $this->issuedAt + 3600 * 12;
    }

    public function generateToken($data) {
        $this->token = array(
            // Header
            "iat" => $this->issuedAt,
            "exp" => $this->expire,
            // Payload
            "data" => $data
        );

        $this->jwt = $this->encode($this->token);
        
        return $this->jwt;
    }

    private function encode($payload) {
        // Implementación básica de JWT
        $header = json_encode(['typ' => 'JWT', 'alg' => 'HS256']);
        $payload = json_encode($payload);
        
        $header_encoded = $this->base64UrlEncode($header);
        $payload_encoded = $this->base64UrlEncode($payload);
        
        $signature = hash_hmac('sha256', "$header_encoded.$payload_encoded", $this->jwt_secret, true);
        $signature_encoded = $this->base64UrlEncode($signature);
        
        return "$header_encoded.$payload_encoded.$signature_encoded";
    }

    public function decode($jwt) {
        $tokenParts = explode('.', $jwt);
        if (count($tokenParts) != 3) {
            return false;
        }

        list($header_encoded, $payload_encoded, $signature_encoded) = $tokenParts;
        
        $signature = $this->base64UrlDecode($signature_encoded);
        $raw_signature = hash_hmac('sha256', "$header_encoded.$payload_encoded", $this->jwt_secret, true);
        
        if (hash_equals($raw_signature, $signature)) {
            $payload = json_decode($this->base64UrlDecode($payload_encoded), true);
            
            // Verificar expiración
            if (isset($payload['exp']) && $payload['exp'] < time()) {
                return false; // Token expirado
            }
            
            return $payload;
        }
        
        return false;
    }

    private function base64UrlEncode($data) {
        return rtrim(strtr(base64_encode($data), '+/', '-_'), '=');
    }

    private function base64UrlDecode($data) {
        return base64_decode(strtr($data, '-_', '+/'));
    }
}
?>