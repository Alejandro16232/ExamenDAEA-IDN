using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;

namespace WebApi.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class PythonController : ControllerBase
    {
        [HttpGet("run-script")]
        public IActionResult RunPythonScript()
        {
            try
            {
                // Ruta al script Python dentro del contenedor
                string scriptPath = "/WebApi/a.py";
                
                // Configuración del proceso para ejecutar el script Python
                var startInfo = new ProcessStartInfo
                {
                    FileName = "python3",
                    Arguments = scriptPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using (var process = Process.Start(startInfo))
                {
                    using (var reader = process.StandardOutput)
                    {
                        string result = reader.ReadToEnd();
                        return Ok(result);
                    }
                }
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }
    }
}
