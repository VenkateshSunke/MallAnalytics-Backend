using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;

namespace videoxpert.Controllers;

[ApiController]
[Route("api/[controller]")]
public class HealthController : ControllerBase
{
    private readonly ILogger<HealthController> _logger;

    public HealthController(ILogger<HealthController> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Health check endpoint to verify API is running
    /// </summary>
    /// <returns>API status and version information</returns>
    [HttpGet]
    public IActionResult GetHealth()
    {
        try
        {
            var response = new
            {
                status = "healthy",
                timestamp = DateTime.UtcNow,
                version = "1.0.0",
                service = "VideoXpert API"
            };

            return Ok(response);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Health check failed");
            return StatusCode(500, new { status = "unhealthy", error = ex.Message });
        }
    }
} 