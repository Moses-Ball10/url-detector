# How to Run the Project

This system operates in two parts: the **AI Python Microservice (Backend)** and the **Laravel Web App (Frontend)**. You must run both for the system to work.

---

### Part 1: Start the AI FastAPI Server (Python)

1. **Open your terminal** and navigate to your project directory:
   ```bash
   cd /Users/mac/Development/ds
   ```
2. **Activate your virtual environment**:
   ```bash
   source venv/bin/activate
   ```
3. **Run the FastAPI server**:
   ```bash
   uvicorn step4_api:app --host 127.0.0.1 --port 8000 --reload
   ```
4. **Verify it is running**: You should see output saying `Application startup complete`. DO NOT close this terminal window.

---

### Part 2: Setup & Run Laravel (PHP Front-End)

*(Assuming you have a Laravel project ready or are spinning up a new one)*

1. **Copy the PHP/Blade files** from the generated folders into your existing Laravel project:
   * Copy `step5_laravel_code/UrlScannerController.php` ➔ `app/Http/Controllers/UrlScannerController.php`
   * Copy the code inside `step5_laravel_code/routes.php` ➔ paste it into your `routes/web.php` file.
   * Copy `step6_ui/dashboard.blade.php` ➔ `resources/views/scanner/dashboard.blade.php` *(You will need to create the `scanner` folder inside the `views` directory)*.

2. **Open a SECOND terminal window**, navigate to your **Laravel project directory**, and start the PHP development server:
   ```bash
   php artisan serve
   ```
   *(By default, this will run on `http://127.0.0.1:8000`. If you face a port conflict with FastAPI, run Laravel on port 8001 instead: `php artisan serve --port=8001`)*

3. **Use the Application**:
   Open your web browser and go to `http://localhost:8000` (or 8001). You will see the Tailwind Dashboard. Enter a URL, click "Scan Target", and watch exactly how Laravel passes the data to Python and renders the threat prediction!

---

### Troubleshooting
* **"Unable to connect to the AI engine" error in browser**: This means your Python FastAPI server is not running, or it crashed. Check your first terminal window.
* **Missing Python modules**: Make sure you activated your `venv` before running the `uvicorn` command.
* **Missing Laravel views missing error**: Ensure you placed `dashboard.blade.php` exactly inside a folder named `scanner` inside your `resources/views/` folder.
