using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using videoxpert.Services;

namespace videoxpert.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CamerasController : ControllerBase
{
    private readonly IVideoXpertService _videoXpertService;
    private readonly ILogger<CamerasController> _logger;

    public CamerasController(IVideoXpertService videoXpertService, ILogger<CamerasController> logger)
    {
        _videoXpertService = videoXpertService;
        _logger = logger;
    }

    /// <summary>
    /// Lists all available cameras in the VideoXpert system
    /// </summary>
    /// <returns>List of cameras with their details</returns>
    [HttpGet]
    public async Task<IActionResult> ListCameras()
    {
        try
        {
            _logger.LogInformation("Received request to list cameras");
            var result = await _videoXpertService.ListCamerasAsync();
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ListCameras endpoint");
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }

    /// <summary>
    /// Lists all clips for a specific camera
    /// </summary>
    /// <param name="cameraId">The camera ID or name</param>
    /// <returns>List of video clips with their time ranges and summary information</returns>
    [HttpGet("{cameraId}/clips")]
    public async Task<IActionResult> ListClips(string cameraId)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(cameraId))
            {
                return BadRequest(new { success = false, error = "Camera ID is required" });
            }

            _logger.LogInformation("Received request to list clips for camera: {CameraId}", cameraId);
            var result = await _videoXpertService.ListClipsAsync(cameraId);
            
            if (result.Success)
            {
                return Ok(result);
            }
            
            return BadRequest(result);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error in ListClips endpoint for camera {CameraId}", cameraId);
            return StatusCode(500, new { success = false, error = "Internal server error" });
        }
    }
} 