using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;
using System.IO.Compression;
using System.Net;
using videoxpert.Domain;
using VxSdkNet;

namespace videoxpert.Services;

public class VideoXpertService : IVideoXpertService, IDisposable
{
    private readonly VideoXpertConfiguration _config;
    private readonly ILogger<VideoXpertService> _logger;
    private VXSystem? _currentSystem;
    private readonly object _sdkLock = new();
    private DateTime _lastConnectionTime = DateTime.MinValue;
    private readonly TimeSpan _connectionTimeout = TimeSpan.FromMinutes(30);

    public VideoXpertService(IOptions<VideoXpertConfiguration> config, ILogger<VideoXpertService> logger)
    {
        _config = config.Value;
        _logger = logger;
        
        // Debug logging to see what configuration is loaded
        _logger.LogInformation("VideoXpert Configuration Debug:");
        _logger.LogInformation("  ServerIp: '{ServerIp}'", _config.ServerIp);
        _logger.LogInformation("  ServerPort: {ServerPort}", _config.ServerPort);
        _logger.LogInformation("  Username: '{Username}'", _config.Username);
        _logger.LogInformation("  Password: '{Password}'", string.IsNullOrEmpty(_config.Password) ? "EMPTY" : "***SET***");
        _logger.LogInformation("  LicenseKey: '{LicenseKey}'", string.IsNullOrEmpty(_config.LicenseKey) ? "EMPTY" : $"***{_config.LicenseKey.Length} chars***");
        _logger.LogInformation("  UseSsl: {UseSsl}", _config.UseSsl);
        _logger.LogInformation("  TimeoutMinutes: {TimeoutMinutes}", _config.TimeoutMinutes);
        _logger.LogInformation("  OutputDirectory: '{OutputDirectory}'", _config.OutputDirectory);
    }

    public async Task<CameraResponse> ListCamerasAsync()
    {
        try
        {
            var connectionResult = await EnsureConnectionAsync();
            if (!connectionResult.Success)
            {
                return new CameraResponse { Success = false, Error = connectionResult.Error };
            }

            _logger.LogInformation("Listing available cameras...");

            var allSources = new List<DataSource>(_currentSystem!.DataSources);
            var cameraList = allSources.Select(camera => new Camera
            {
                Id = camera.Id,
                Name = camera.Name,
                State = camera.State.ToString(),
                Type = camera.Type.ToString()
            }).ToList();

            _logger.LogInformation("Listed {Count} cameras", cameraList.Count);

            return new CameraResponse
            {
                Success = true,
                TotalCameras = cameraList.Count,
                Cameras = cameraList
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error listing cameras");
            return new CameraResponse
            {
                Success = false,
                Error = ex.Message,
                Cameras = new List<Camera>()
            };
        }
    }

    public async Task<ClipsResponse> ListClipsAsync(string cameraId)
    {
        try
        {
            var connectionResult = await EnsureConnectionAsync();
            if (!connectionResult.Success)
            {
                return new ClipsResponse { Success = false, Error = connectionResult.Error };
            }

            _logger.LogInformation("Listing clips for camera: {CameraId}", cameraId);

            var camera = FindCamera(cameraId);
            if (camera == null)
            {
                var error = $"Camera not found: {cameraId}";
                _logger.LogError(error);
                return new ClipsResponse
                {
                    Success = false,
                    Error = error,
                    Clips = new List<VideoClip>()
                };
            }

            _logger.LogInformation("Found camera: {CameraName}", camera.Name);

            var recordings = camera.Clips;
            var clipsList = new List<VideoClip>();

            if (recordings != null && recordings.Count > 0)
            {
                foreach (var clip in recordings.OrderBy(c => c.StartTime))
                {
                    var duration = clip.EndTime - clip.StartTime;
                    clipsList.Add(new VideoClip
                    {
                        StartTime = clip.StartTime.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"),
                        EndTime = clip.EndTime.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"),
                        StartTimeUtc = clip.StartTime.ToString("yyyy-MM-dd HH:mm:ss"),
                        EndTimeUtc = clip.EndTime.ToString("yyyy-MM-dd HH:mm:ss"),
                        DurationMinutes = Math.Round(duration.TotalMinutes, 1),
                        DurationHours = Math.Round(duration.TotalHours, 2)
                    });
                }
            }

            // Calculate overall time span
            DateTime? earliestStart = null;
            DateTime? latestEnd = null;
            double totalRecordingHours = 0;

            if (clipsList.Count > 0 && recordings != null)
            {
                earliestStart = recordings.Min(r => r.StartTime).ToLocalTime();
                latestEnd = recordings.Max(r => r.EndTime).ToLocalTime();
                totalRecordingHours = recordings.Sum(r => (r.EndTime - r.StartTime).TotalHours);
            }

            var result = new ClipsResponse
            {
                Success = true,
                Camera = new Camera
                {
                    Id = camera.Id,
                    Name = camera.Name,
                    State = camera.State.ToString(),
                    Type = camera.Type.ToString()
                },
                Summary = new ClipsSummary
                {
                    TotalClips = clipsList.Count,
                    EarliestRecording = earliestStart?.ToString("yyyy-MM-dd HH:mm:ss"),
                    LatestRecording = latestEnd?.ToString("yyyy-MM-dd HH:mm:ss"),
                    TotalRecordingHours = Math.Round(totalRecordingHours, 2),
                    TotalSpanDays = earliestStart.HasValue && latestEnd.HasValue ?
                        Math.Round((latestEnd.Value - earliestStart.Value).TotalDays, 1) : 0
                },
                Clips = clipsList
            };

            _logger.LogInformation("Listed {Count} clips for camera {CameraName}", clipsList.Count, camera.Name);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error listing clips for camera {CameraId}", cameraId);
            return new ClipsResponse
            {
                Success = false,
                Error = ex.Message,
                Camera = null,
                Clips = new List<VideoClip>()
            };
        }
    }

    public async Task<ExportResponse> CreateExportAsync(ExportRequest request)
    {
        try
        {
            var connectionResult = await EnsureConnectionAsync();
            if (!connectionResult.Success)
            {
                return new ExportResponse { Success = false, Error = connectionResult.Error };
            }

            // Validate time parameters
            if (!DateTime.TryParse(request.StartTime, out DateTime startTime))
            {
                return new ExportResponse { Success = false, Error = $"Invalid start_time format: {request.StartTime}. Use 'yyyy-MM-dd HH:mm:ss'" };
            }

            if (!DateTime.TryParse(request.EndTime, out DateTime endTime))
            {
                return new ExportResponse { Success = false, Error = $"Invalid end_time format: {request.EndTime}. Use 'yyyy-MM-dd HH:mm:ss'" };
            }

            if (startTime >= endTime)
            {
                return new ExportResponse { Success = false, Error = "start_time must be before end_time" };
            }

            var camera = FindCamera(request.CameraId);
            if (camera == null)
            {
                return new ExportResponse { Success = false, Error = $"Camera not found: {request.CameraId}" };
            }

            _logger.LogInformation("Creating export for camera {CameraName} from {StartTime} to {EndTime}", 
                camera.Name, startTime, endTime);

            // Check video availability
            var hasVideo = await CheckVideoAvailabilityAsync(camera, startTime, endTime);
            if (!hasVideo)
            {
                return new ExportResponse { Success = false, Error = "No video data available for the specified time range" };
            }

            var export = await CreateExportInternalAsync(camera, startTime, endTime, request.ExportPassword);
            if (export == null)
            {
                return new ExportResponse { Success = false, Error = "Failed to create export" };
            }

            return new ExportResponse
            {
                Success = true,
                ExportId = export.Id,
                ExportName = export.Name,
                Status = export.Status.ToString(),
                FileSizeKb = export.FileSizeKb
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating export");
            return new ExportResponse { Success = false, Error = ex.Message };
        }
    }

    public async Task<ExportResponse> CreateAndDownloadExportAsync(ExportRequest request)
    {
        try
        {
            // First create the export
            var createResult = await CreateExportAsync(request);
            if (!createResult.Success || createResult.ExportId == null)
            {
                return createResult;
            }

            // Wait for export completion
            var completedExport = await WaitForExportCompletionAsync(createResult.ExportId, _config.TimeoutMinutes);
            if (completedExport == null)
            {
                return new ExportResponse { Success = false, Error = "Export failed or timed out" };
            }

            if (completedExport.Status == "Failed")
            {
                return new ExportResponse { Success = false, Error = $"Export failed: {completedExport.StatusReason}" };
            }

            // Download the export
            var zipPath = await DownloadExportAsync(completedExport);
            if (string.IsNullOrEmpty(zipPath))
            {
                return new ExportResponse { Success = false, Error = "Failed to download export" };
            }

            return new ExportResponse
            {
                Success = true,
                ExportId = completedExport.Id,
                ExportName = completedExport.Name,
                Status = completedExport.Status,
                FilePath = zipPath,
                FileSizeKb = completedExport.FileSizeKb
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating and downloading export");
            return new ExportResponse { Success = false, Error = ex.Message };
        }
    }

    public async Task<ExportDownloadResponse> CreateAndGetDownloadUrlAsync(ExportRequest request)
    {
        try
        {
            // First create the export
            var createResult = await CreateExportAsync(request);
            if (!createResult.Success || createResult.ExportId == null)
            {
                return new ExportDownloadResponse 
                { 
                    Success = false, 
                    Error = createResult.Error ?? "Failed to create export" 
                };
            }

            // Wait for export completion
            var completedExport = await WaitForExportCompletionAsync(createResult.ExportId, _config.TimeoutMinutes);
            if (completedExport == null)
            {
                return new ExportDownloadResponse 
                { 
                    Success = false, 
                    Error = "Export failed or timed out" 
                };
            }

            if (completedExport.Status == "Failed")
            {
                return new ExportDownloadResponse 
                { 
                    Success = false, 
                    Error = $"Export failed: {completedExport.StatusReason}" 
                };
            }

            // Return the download URL
            return new ExportDownloadResponse
            {
                Success = true,
                ExportId = completedExport.Id,
                ExportName = completedExport.Name,
                DownloadUrl = completedExport.DataUri,
                FileSizeKb = completedExport.FileSizeKb,
                Instructions = "Use this URL to download the export file directly. Include the following headers: X-Serenity-User (base64 encoded username) and X-Serenity-Password (base64 encoded password)."
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating export and getting download URL");
            return new ExportDownloadResponse { Success = false, Error = ex.Message };
        }
    }

    public async Task<ExportDownloadResponse> GetExportDownloadUrlAsync(string exportId)
    {
        try
        {
            var connectionResult = await EnsureConnectionAsync();
            if (!connectionResult.Success)
            {
                return new ExportDownloadResponse { Success = false, Error = connectionResult.Error };
            }

            Export? export = null;
            lock (_sdkLock)
            {
                var exports = _currentSystem!.Exports;
                export = exports.FirstOrDefault(exp => exp.Id == exportId);
            }

            if (export == null)
            {
                return new ExportDownloadResponse { Success = false, Error = "Export not found" };
            }

            if (export.Status != Export.States.Successful)
            {
                return new ExportDownloadResponse 
                { 
                    Success = false, 
                    Error = $"Export is not ready for download. Status: {export.Status}" 
                };
            }

            if (string.IsNullOrEmpty(export.DataUri))
            {
                return new ExportDownloadResponse 
                { 
                    Success = false, 
                    Error = "Export download URL is not available" 
                };
            }

            return new ExportDownloadResponse
            {
                Success = true,
                ExportId = export.Id,
                ExportName = export.Name,
                DownloadUrl = export.DataUri,
                FileSizeKb = export.FileSizeKb,
                Instructions = "Use this URL to download the export file directly. Include the following headers: X-Serenity-User (base64 encoded username) and X-Serenity-Password (base64 encoded password)."
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting export download URL");
            return new ExportDownloadResponse { Success = false, Error = ex.Message };
        }
    }

    public async Task<ExportResponse> GetExportStatusAsync(string exportId)
    {
        try
        {
            var connectionResult = await EnsureConnectionAsync();
            if (!connectionResult.Success)
            {
                return new ExportResponse { Success = false, Error = connectionResult.Error };
            }

            Export? export = null;
            lock (_sdkLock)
            {
                var exports = _currentSystem!.Exports;
                export = exports.FirstOrDefault(exp => exp.Id == exportId);
            }

            if (export == null)
            {
                return new ExportResponse { Success = false, Error = "Export not found" };
            }

            return new ExportResponse
            {
                Success = true,
                ExportId = export.Id,
                ExportName = export.Name,
                Status = export.Status.ToString(),
                FileSizeKb = export.FileSizeKb,
                StatusReason = export.StatusReason.ToString()
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting export status");
            return new ExportResponse { Success = false, Error = ex.Message };
        }
    }

    private async Task<(bool Success, string? Error)> EnsureConnectionAsync()
    {
        try
        {
            // Check if we need to reconnect
            if (_currentSystem == null || DateTime.Now - _lastConnectionTime > _connectionTimeout)
            {
                _currentSystem?.Dispose();
                
                _logger.LogInformation("Connecting to VideoXpert system...");
                _logger.LogInformation("Server: {ServerIp}:{ServerPort}, SSL: {UseSsl}", _config.ServerIp, _config.ServerPort, _config.UseSsl);
                _logger.LogInformation("Username: {Username}", _config.Username);
                _logger.LogInformation("License Key Length: {LicenseKeyLength}, First 10 chars: {LicenseKeyPrefix}", 
                    _config.LicenseKey?.Length ?? 0, 
                    _config.LicenseKey?.Substring(0, Math.Min(10, _config.LicenseKey?.Length ?? 0)) ?? "NULL");
                
                _currentSystem = new VXSystem(_config.ServerIp, _config.ServerPort, _config.UseSsl, _config.LicenseKey);
                
                var result = _currentSystem.Login(_config.Username, _config.Password);
                
                if (result == Results.Value.SdkLicenseGracePeriodActive)
                {
                    var expirationTime = _currentSystem.GraceLicenseExpirationTime;
                    _logger.LogWarning("License grace period active. Expires on {ExpirationTime}", expirationTime.ToLocalTime());
                }
                else if (result != Results.Value.OK)
                {
                    if (result == Results.Value.SdkLicenseGracePeriodExpired)
                    {
                        _logger.LogError("License grace period expired. System must be licensed to proceed.");
                    }
                    else
                    {
                        _logger.LogError("Login failed: {Result}", result);
                    }
                    return (false, $"Login failed: {result}");
                }
                
                var user = _currentSystem.Currentuser;
                var sysDevice = _currentSystem.HostDevice;
                var sysName = sysDevice?.Name ?? _config.ServerIp;
                _logger.LogInformation("Successfully connected to {SystemName} as {UserName}", sysName, user.Name);
                
                _lastConnectionTime = DateTime.Now;
            }
            
            return (true, null);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Connection failed");
            return (false, $"Connection failed: {ex.Message}");
        }
    }

    private DataSource? FindCamera(string cameraId)
    {
        try
        {
            var allSources = new List<DataSource>(_currentSystem!.DataSources);

            // Try exact ID match first
            var exactMatch = allSources.FirstOrDefault(ds => ds.Id == cameraId);
            if (exactMatch != null)
                return exactMatch;

            // Try name match
            var nameMatch = allSources.FirstOrDefault(ds =>
                ds.Name.Equals(cameraId, StringComparison.OrdinalIgnoreCase));
            if (nameMatch != null)
                return nameMatch;

            // Try partial name match
            var partialMatch = allSources.FirstOrDefault(ds =>
                ds.Name.IndexOf(cameraId, StringComparison.OrdinalIgnoreCase) >= 0);

            return partialMatch;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error finding camera");
            return null;
        }
    }

    private async Task<bool> CheckVideoAvailabilityAsync(DataSource camera, DateTime startTime, DateTime endTime)
    {
        try
        {
            _logger.LogInformation("Checking video availability...");

            var recordings = camera.Clips;
            if (recordings == null || recordings.Count == 0)
            {
                _logger.LogError("No recordings found for this camera");
                return false;
            }

            var utcStartTime = startTime.ToUniversalTime();
            var utcEndTime = endTime.ToUniversalTime();

            foreach (var recording in recordings)
            {
                var recordingStart = recording.StartTime;
                var recordingEnd = recording.EndTime;

                // Check if there's any overlap between recording and requested time range
                bool overlaps = recordingStart < utcEndTime && recordingEnd > utcStartTime;

                if (overlaps)
                {
                    _logger.LogInformation("Video data is available for the requested time range");
                    return true;
                }
            }

            _logger.LogError("No video data available for the requested time range");
            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error checking video availability");
            // If we can't check availability, assume it's available and let the export process handle it
            return true;
        }
    }

    private async Task<Export?> CreateExportInternalAsync(DataSource camera, DateTime startTime, DateTime endTime, string? exportPassword)
    {
        try
        {
            _logger.LogInformation("Creating export...");

            var exportClip = new NewExportClip
            {
                DataSourceId = camera.Id,
                StartTime = startTime.ToUniversalTime(),
                EndTime = endTime.ToUniversalTime()
            };

            var newExport = new NewExport();
            newExport.Clips.Add(exportClip);
            newExport.Format = Export.Formats.MkvZip;
            newExport.Name = $"API_Export_{camera.Name}_{startTime:yyyyMMdd_HHmmss}_{endTime:yyyyMMdd_HHmmss}";

            if (!string.IsNullOrEmpty(exportPassword))
            {
                newExport.Password = exportPassword;
            }

            // Get export size estimate and submit export (synchronized)
            lock (_sdkLock)
            {
                var estimate = _currentSystem!.GetExportEstimate(newExport);
                if (estimate != null && estimate.Size >= 0)
                {
                    _logger.LogInformation("Export size estimate: {Size}", FormatFileSize(estimate.Size * 1024));
                    if (estimate.IsTooLarge)
                    {
                        _logger.LogWarning("Export may be too large");
                    }
                }

                _currentSystem.AddExport(newExport);
                _logger.LogInformation("Export submitted successfully");
            }

            // Wait for export to appear in the system
            await Task.Delay(30000); // Wait 30 seconds

            Export? createdExport = null;
            lock (_sdkLock)
            {
                var exports = _currentSystem!.Exports;
                createdExport = exports
                    .Where(export => export.Name.Contains(camera.Name) && export.Status != Export.States.Failed)
                    .OrderByDescending(export => export.Name)
                    .FirstOrDefault();
            }

            if (createdExport != null)
            {
                _logger.LogInformation("Export created: {ExportName}", createdExport.Name);
            }

            return createdExport;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error creating export");
            return null;
        }
    }

    private async Task<ExportStatus?> WaitForExportCompletionAsync(string exportId, int timeoutMinutes)
    {
        try
        {
            _logger.LogInformation("Waiting for export completion...");

            var timeout = TimeSpan.FromMinutes(timeoutMinutes);
            var startTime = DateTime.Now;

            while (DateTime.Now - startTime < timeout)
            {
                Export? currentExport = null;
                lock (_sdkLock)
                {
                    var updatedExports = _currentSystem!.Exports;
                    currentExport = updatedExports.FirstOrDefault(exp => exp.Id == exportId);
                }

                if (currentExport == null)
                {
                    _logger.LogError("Export not found");
                    return null;
                }

                _logger.LogInformation("Export status: {Status}", currentExport.Status);

                if (currentExport.Status == Export.States.Successful)
                {
                    _logger.LogInformation("Export completed successfully");
                    return new ExportStatus
                    {
                        Id = currentExport.Id,
                        Name = currentExport.Name,
                        Status = currentExport.Status.ToString(),
                        StatusReason = currentExport.StatusReason.ToString(),
                        FileSizeKb = currentExport.FileSizeKb,
                        DataUri = currentExport.DataUri
                    };
                }
                else if (currentExport.Status == Export.States.Failed)
                {
                    _logger.LogError("Export failed: {StatusReason}", currentExport.StatusReason);
                    return new ExportStatus
                    {
                        Id = currentExport.Id,
                        Name = currentExport.Name,
                        Status = currentExport.Status.ToString(),
                        StatusReason = currentExport.StatusReason.ToString(),
                        FileSizeKb = currentExport.FileSizeKb
                    };
                }

                await Task.Delay(10000); // Check every 10 seconds
            }

            _logger.LogError("Export timed out after {TimeoutMinutes} minutes", timeoutMinutes);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error waiting for export completion");
            return null;
        }
    }

    private async Task<string?> DownloadExportAsync(ExportStatus export)
    {
        try
        {
            if (string.IsNullOrEmpty(export.DataUri))
            {
                _logger.LogError("Export data URI is not available");
                return null;
            }

            // Create output directory
            if (!Directory.Exists(_config.OutputDirectory))
            {
                Directory.CreateDirectory(_config.OutputDirectory);
            }

            var fileName = $"{export.Name}.zip";
            var downloadPath = Path.Combine(_config.OutputDirectory, fileName);

            _logger.LogInformation("Downloading export to: {DownloadPath}", downloadPath);
            _logger.LogInformation("File size: {FileSize}", FormatFileSize(export.FileSizeKb * 1024));

            // Set security protocol
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            ServicePointManager.ServerCertificateValidationCallback = delegate { return true; };

            // Try download with HttpClient first
            bool downloadSuccess = false;
            try
            {
                await DownloadWithHttpClient(export.DataUri, downloadPath);
                downloadSuccess = true;
            }
            catch (Exception httpEx)
            {
                _logger.LogError(httpEx, "HttpClient download failed");
                _logger.LogInformation("Trying with WebClient...");

                // Fallback to WebClient
                try
                {
                    using (var webClient = new WebClient())
                    {
                        webClient.Headers.Add("X-Serenity-User", EncodeToBase64(_config.Username));
                        webClient.Headers.Add("X-Serenity-Password", EncodeToBase64(_config.Password));

                        var downloadUri = new Uri(export.DataUri);
                        await webClient.DownloadFileTaskAsync(downloadUri, downloadPath);
                        downloadSuccess = true;
                    }
                }
                catch (Exception webEx)
                {
                    _logger.LogError(webEx, "WebClient download also failed");
                    _logger.LogError("Both download methods failed. HttpClient: {HttpError}, WebClient: {WebError}", httpEx.Message, webEx.Message);
                    return null;
                }
            }

            if (downloadSuccess)
            {
                _logger.LogInformation("Download completed successfully");
                return downloadPath;
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Download failed");
            return null;
        }
    }

    private async Task DownloadWithHttpClient(string dataUri, string downloadPath)
    {
        using (var httpClient = new HttpClient())
        {
            httpClient.DefaultRequestHeaders.Add("X-Serenity-User", EncodeToBase64(_config.Username));
            httpClient.DefaultRequestHeaders.Add("X-Serenity-Password", EncodeToBase64(_config.Password));
            httpClient.Timeout = TimeSpan.FromMinutes(30);

            using (var response = await httpClient.GetAsync(dataUri, HttpCompletionOption.ResponseHeadersRead))
            {
                response.EnsureSuccessStatusCode();

                using (var contentStream = await response.Content.ReadAsStreamAsync())
                using (var fileStream = new FileStream(downloadPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true))
                {
                    await contentStream.CopyToAsync(fileStream);
                }
            }
        }
    }

    private static string EncodeToBase64(string toEncode)
    {
        var toEncodeAsBytes = System.Text.Encoding.ASCII.GetBytes(toEncode);
        return Convert.ToBase64String(toEncodeAsBytes);
    }

    private static string FormatFileSize(long bytes)
    {
        if (bytes < 1024)
            return bytes + " bytes";
        else if (bytes < 1048576)
            return Math.Round((double)bytes / 1024, 1) + " KB";
        else if (bytes < 1073741824)
            return Math.Round((double)bytes / 1024 / 1024, 1) + " MB";
        else
            return Math.Round((double)bytes / 1024 / 1024 / 1024, 1) + " GB";
    }

    public void Dispose()
    {
        _currentSystem?.Dispose();
    }
} 