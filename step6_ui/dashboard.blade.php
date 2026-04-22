<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Malicious URL Scanner</title>
    <!-- Tailwind CSS (via CDN for simplicity in this example) -->
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        primary: '#1e3a8a',
                        secondary: '#3b82f6',
                        dark: '#0f172a',
                    }
                }
            }
        }
    </script>
</head>
<body class="bg-gray-50 min-h-screen font-sans text-gray-800">

    <!-- Top Navigation Bar -->
    <nav class="bg-dark text-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16 items-center">
                <div class="flex items-center gap-3">
                    <i class="fa-solid fa-shield-halved text-secondary text-2xl"></i>
                    <span class="font-bold text-xl tracking-tight">CyberShield AI Engine</span>
                </div>
                <div>
                    <span class="text-sm text-gray-300">Deep Learning URL Analysis</span>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content Area -->
    <main class="max-w-4xl mx-auto px-4 py-12">
        
        <!-- Header Section -->
        <div class="text-center mb-10">
            <h1 class="text-4xl font-extrabold text-gray-900 mb-4">Malicious URL Scanner</h1>
            <p class="text-lg text-gray-600">
                Powered by a Deep Neural Network trained on 79 lexical features. 
                Enter a URL to detect Phishing, Malware, Spam, Defacement, or Benign sites.
            </p>
        </div>

        <!-- Scanner Form Card -->
        <div class="bg-white rounded-2xl shadow-xl p-8 mb-8 border border-gray-100">
            
            <!-- Error Alert -->
            @if(session('error'))
            <div class="bg-red-50 border-l-4 border-red-500 p-4 mb-6 rounded-r-md">
                <div class="flex">
                    <div class="flex-shrink-0">
                        <i class="fa-solid fa-circle-exclamation text-red-500"></i>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm text-red-700 font-medium">{{ session('error') }}</p>
                    </div>
                </div>
            </div>
            @endif

            <form action="{{ route('scanner.scan') }}" method="POST">
                @csrf
                <div class="flex flex-col md:flex-row gap-4">
                    <div class="flex-grow">
                        <div class="relative">
                            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                                <i class="fa-solid fa-globe text-gray-400"></i>
                            </div>
                            <input 
                                type="url" 
                                name="url" 
                                id="url" 
                                required
                                value="{{ old('url', $scannedUrl ?? '') }}"
                                class="pl-10 block w-full rounded-xl border-gray-300 shadow-sm focus:ring-secondary focus:border-secondary sm:text-lg p-4 border bg-gray-50 focus:bg-white transition-colors" 
                                placeholder="https://example.com/suspicious-link"
                            >
                        </div>
                        @error('url')
                            <p class="mt-2 text-sm text-red-600"><i class="fa-solid fa-triangle-exclamation mr-1"></i>{{ $message }}</p>
                        @enderror
                    </div>
                    <button type="submit" class="inline-flex justify-center items-center px-8 py-4 border border-transparent text-lg font-medium rounded-xl shadow-sm text-white bg-secondary hover:bg-primary focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary transition-all">
                        <i class="fa-solid fa-radar mr-2"></i> Scan Target
                    </button>
                </div>
            </form>
        </div>

        <!-- Results Section (Only displays if AI returns data) -->
        @if(isset($scanResult))
            @php
                $isSafe = $scanResult['risk_level'] === 'Safe';
                $prediction = $scanResult['prediction'];
                
                // Determine styling based on AI prediction
                $statusColor = $isSafe ? 'green' : 'red';
                $iconClass = $isSafe ? 'fa-shield-check' : 'fa-triangle-exclamation';
                
                // Color mapping for specific malicious types
                if (!$isSafe) {
                    if ($prediction === 'Phishing') $statusColor = 'orange';
                    if ($prediction === 'Malware') $statusColor = 'red';
                    if ($prediction === 'Spam') $statusColor = 'yellow';
                    if ($prediction === 'Defacement') $statusColor = 'purple';
                }
            @endphp
            
            <div class="bg-white rounded-2xl shadow-xl overflow-hidden border border-gray-100 animate-fade-in-up">
                
                <!-- Result Banner -->
                <div class="bg-{{ $statusColor }}-600 px-6 py-8 text-white text-center">
                    <i class="fa-solid {{ $iconClass }} text-5xl mb-4 opacity-90 mx-auto block"></i>
                    <h2 class="text-3xl font-bold mb-2">
                        {{ $isSafe ? 'Safe URL Detected' : 'Threat Detected: ' . $prediction }}
                    </h2>
                    <p class="text-{{ $statusColor }}-100 text-lg opacity-90">
                        AI Confidence Level: <span class="font-bold">{{ $scanResult['confidence'] }}</span>
                    </p>
                </div>

                <!-- Detailed Breakdown -->
                <div class="p-8">
                    <h3 class="text-lg font-semibold text-gray-900 mb-6 border-b pb-2">Analysis Breakdown</h3>
                    
                    <div class="space-y-5">
                        @foreach($scanResult['all_probabilities'] as $class => $probStr)
                            @php
                                // Strip % and convert to float for the progress bar width
                                $probFloat = floatval(str_replace('%', '', $probStr));
                            @endphp
                            <div>
                                <div class="flex justify-between mb-1">
                                    <span class="text-sm font-medium text-gray-700">{{ $class }}</span>
                                    <span class="text-sm font-semibold text-gray-900">{{ $probStr }}</span>
                                </div>
                                <div class="w-full bg-gray-200 rounded-full h-2.5">
                                    <div class="h-2.5 rounded-full {{ $class === $prediction ? 'bg-'.$statusColor.'-500' : 'bg-gray-400' }}" style="width: {{ $probFloat }}%"></div>
                                </div>
                            </div>
                        @endforeach
                    </div>
                    
                    <div class="mt-8 pt-6 border-t text-sm text-gray-500 flex justify-between items-center">
                        <div>
                            <i class="fa-solid fa-microchip mr-1"></i> Processed via FastAPI Deep Neural Network
                        </div>
                        <div>
                            <span class="bg-gray-100 text-gray-700 px-3 py-1 rounded-full font-mono text-xs">Latency: < 100ms</span>
                        </div>
                    </div>
                </div>
            </div>
        @endif

    </main>

    <style>
        /* Simple Tailwind Animation utility */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade-in-up {
            animation: fadeInUp 0.5s ease-out forwards;
        }
    </style>
</body>
</html>
