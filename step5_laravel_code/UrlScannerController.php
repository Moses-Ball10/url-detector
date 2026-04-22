<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;

/**
 * =========================================================================
 * STEP 5: LARAVEL INTEGRATION - URL Scanner Controller
 * =========================================================================
 * Project : Multi-Class Malicious URL Detection System
 * Purpose : This controller serves as the bridge between the Laravel web
 *           frontend and the FastAPI AI microservice.
 *           It takes user input, sanitizes it, forwards it to the Python API,
 *           and formats the response back for the UI.
 * =========================================================================
 */
class UrlScannerController extends Controller
{
    /**
     * URL of the running FastAPI Python Microservice (from Step 4)
     */
    private string $aiMicroserviceUrl = 'http://127.0.0.1:8000/predict';

    /**
     * Display the main URL scanner dashboard
     */
    public function index()
    {
        return view('scanner.dashboard');
    }

    /**
     * Handle the URL submission and communicate with the Python API
     */
    public function scan(Request $request)
    {
        // 1. Validate the user input
        // Ensure they submitted a 'url' field and that it resembles a URL string
        $validated = $request->validate([
            'url' => 'required|string|min:4'
        ]);

        $targetUrl = $validated['url'];

        try {
            // 2. HTTP POST request to the Python FastAPI microservice
            // Using Laravel's built-in Http client with a timeout
            $response = Http::timeout(10)->post($this->aiMicroserviceUrl, [
                'url' => $targetUrl
            ]);

            // 3. Handle response from the AI Engine
            if ($response->successful()) {
                $aiData = $response->json();
                
                // Return to the dashboard view with the AI prediction data
                return view('scanner.dashboard', [
                    'scanResult' => $aiData,
                    'scannedUrl' => $targetUrl
                ]);
            } else {
                // If Python API returns 500 error or similar
                Log::error('AI Microservice Error', [
                    'status' => $response->status(),
                    'body' => $response->body()
                ]);
                
                return back()
                    ->with('error', 'The AI engine returned an error. Please ensure the Python API is running.')
                    ->withInput();
            }

        } catch (\Exception $e) {
            // If the Python server is offline/unreachable
            Log::error('AI Microservice Connection Failed', [
                'message' => $e->getMessage()
            ]);

            return back()
                ->with('error', 'Unable to connect to the AI engine. Is the Python FastAPI server running on port 8000?')
                ->withInput();
        }
    }
}
