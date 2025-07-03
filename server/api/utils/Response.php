<?php
// server/api/utils/Response.php

class Response {
    public static function success($data = null, $message = "Operación exitosa", $code = 200) {
        http_response_code($code);
        return json_encode([
            'status' => 'success',
            'message' => $message,
            'data' => $data
        ]);
    }

    public static function error($message = "Ha ocurrido un error", $code = 400) {
        http_response_code($code);
        return json_encode([
            'status' => 'error',
            'message' => $message
        ]);
    }
}
?>