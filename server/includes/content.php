<?php
// Verificar si se ha definido un título de página
if (!isset($pageTitle)) {
    $pageTitle = 'Dashboard';
}
?>
<!-- Content Wrapper. Contains page content -->
<div class="content-wrapper">
    <!-- Content Header (Page header) -->
    <div class="content-header">
        <div class="container-fluid">
            <div class="row mb-2">
                <div class="col-sm-6">
                    <h1 class="m-0"><?php echo $pageTitle; ?></h1>
                </div><!-- /.col -->
                <div class="col-sm-6">
                    <ol class="breadcrumb float-sm-right">
                        <li class="breadcrumb-item"><a href="index.php">Inicio</a></li>
                        <?php if (isset($breadcrumbs) && is_array($breadcrumbs)): ?>
                            <?php foreach ($breadcrumbs as $label => $url): ?>
                                <?php if ($url): ?>
                                    <li class="breadcrumb-item"><a href="<?php echo $url; ?>"><?php echo $label; ?></a></li>
                                <?php else: ?>
                                    <li class="breadcrumb-item active"><?php echo $label; ?></li>
                                <?php endif; ?>
                            <?php endforeach; ?>
                        <?php else: ?>
                            <li class="breadcrumb-item active"><?php echo $pageTitle; ?></li>
                        <?php endif; ?>
                    </ol>
                </div><!-- /.col -->
            </div><!-- /.row -->
        </div><!-- /.container-fluid -->
    </div>
    <!-- /.content-header -->

    <!-- Main content -->
    <section class="content">
        <div class="container-fluid">
            <!-- Aquí irá el contenido específico de cada página -->
            <?php if (isset($pageContent)): ?>
                <?php echo $pageContent; ?>
            <?php else: ?>
                <!-- Contenido por defecto si no se ha definido $pageContent -->
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h3 class="card-title">Contenido predeterminado</h3>
                            </div>
                            <div class="card-body">
                                <p>Esta es una página de contenido predeterminada. Define la variable $pageContent para personalizar este contenido.</p>
                            </div>
                        </div>
                    </div>
                </div>
            <?php endif; ?>
        </div><!-- /.container-fluid -->
    </section>
    <!-- /.content -->
</div>