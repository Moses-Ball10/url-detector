<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UrlScannerController;

/**
 * =========================================================================
 * STEP 5: LARAVEL INTEGRATION - Routes
 * =========================================================================
 * This file contains the route definitions you need to add to your Laravel
 * project's routes/web.php file.
 * =========================================================================
 */

// Route to display the scanner dashboard (GET request)
Route::get('/', [UrlScannerController::class, 'index'])->name('scanner.index');

// Route to handle the form submission and AI processing (POST request)
Route::post('/scan', [UrlScannerController::class, 'scan'])->name('scanner.scan');

