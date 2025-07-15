using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using videoxpert.Domain;
using videoxpert.Services;

namespace videoxpert.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ExportsController : ControllerBase
{
    private readonly IVideoXpertService _videoXpertService;
    private readonly ILogger<ExportsController> _logger;

    public ExportsController(IVideoXpertService videoXpertService, ILogger<ExportsController> logger)
    {
        _videoXpertService = videoXpertService;
        _logger = logger;
    }

    /// <summary>
    /// Creates a video export for the specified camera and time range
    /// </summary>
    /// <param name="request">Export request containing camera ID, start time, and end time</param>
    /// <returns>Export creation result with export ID</returns>
    [HttpPost]
    public async Task<IActionResult> CreateExport([FromBody] ExportRequest request)
    {
        try
        {
            if (request == null)
            {
                return BadRequest(new { success = false, error = "Request body is required" });
            }

            if (string.IsNullOrWhiteSpace(request.CameraId))
            {
                return BadRequest(new { success = false, error = "Camera ID is required" });
            }

            if (string.IsNullOrWhiteSpace(request.StartTime))
            {
                return BadRequest(new { success = false, error = "Start time is required" });
            }

            if (string.IsNullOrWhiteSpace(request.EndTime))
            {
                return BadRequest(new { success = false, error = "End time is required" });
            }

            _logger.LogInformation("Received request to create export for camera: {CameraId}", request.CameraId);
            var result = await _videoXpertService.CreateExportAsync(request);
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in CreateExport endpoint");
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }

    /// <summary>
    /// Creates a video export and returns a direct download URL
    /// </summary>
    /// <param name="request">Export request containing camera ID, start time, and end time</param>
    /// <returns>Direct download URL for the export</returns>
    [HttpPost("create-download-url")]
    public async Task<IActionResult> CreateAndGetDownloadUrl([FromBody] ExportRequest request)
    {
        try
        {
            if (request == null)
            {
                return BadRequest(new { success = false, error = "Request body is required" });
            }

            if (string.IsNullOrWhiteSpace(request.CameraId))
            {
                return BadRequest(new { success = false, error = "Camera ID is required" });
            }

            if (string.IsNullOrWhiteSpace(request.StartTime))
            {
                return BadRequest(new { success = false, error = "Start time is required" });
            }

            if (string.IsNullOrWhiteSpace(request.EndTime))
            {
                return BadRequest(new { success = false, error = "End time is required" });
            }

            _logger.LogInformation("Received request to create export and get download URL for camera: {CameraId}", request.CameraId);
            var result = await _videoXpertService.CreateAndGetDownloadUrlAsync(request);
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in CreateAndGetDownloadUrl endpoint");
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }

    /// <summary>
    /// Gets the download URL for an existing export
    /// </summary>
    /// <param name="exportId">The export ID</param>
    /// <returns>Direct download URL for the export</returns>
    [HttpGet("{exportId}/download-url")]
    public async Task<IActionResult> GetExportDownloadUrl(string exportId)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(exportId))
            {
                return BadRequest(new { success = false, error = "Export ID is required" });
            }

            _logger.LogInformation("Received request to get download URL for export: {ExportId}", exportId);
            var result = await _videoXpertService.GetExportDownloadUrlAsync(exportId);
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in GetExportDownloadUrl endpoint for export {ExportId}", exportId);
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }

    /// <summary>
    /// Gets the authentication headers needed for downloading exports
    /// </summary>
    /// <returns>Base64-encoded authentication headers</returns>
    [HttpGet("auth-headers")]
    public IActionResult GetAuthHeaders()
    {
        try
        {
            var username = Environment.GetEnvironmentVariable("VIDEOXPERT_USERNAME") ?? "";
            var password = Environment.GetEnvironmentVariable("VIDEOXPERT_PASSWORD") ?? "";
            
            if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(password))
            {
                return BadRequest(new { success = false, error = "Authentication credentials not configured" });
            }

            var result = new
            {
                success = true,
                headers = new
                {
                    XSerenityUser = Convert.ToBase64String(System.Text.Encoding.ASCII.GetBytes(username)),
                    XSerenityPassword = Convert.ToBase64String(System.Text.Encoding.ASCII.GetBytes(password))
                },
                instructions = "Include these headers when downloading: 'X-Serenity-User' and 'X-Serenity-Password'"
            };

            return Ok(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error generating auth headers");
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }

    /// <summary>
    /// Gets the status of an existing export
    /// </summary>
    /// <param name="exportId">The export ID</param>
    /// <returns>Export status information</returns>
    [HttpGet("{exportId}/status")]
    public async Task<IActionResult> GetExportStatus(string exportId)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(exportId))
            {
                return BadRequest(new { success = false, error = "Export ID is required" });
            }

            _logger.LogInformation("Received request to get export status: {ExportId}", exportId);
            var result = await _videoXpertService.GetExportStatusAsync(exportId);
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in GetExportStatus endpoint for export {ExportId}", exportId);
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }
} 