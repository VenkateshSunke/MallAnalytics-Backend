using videoxpert.Domain;

namespace videoxpert.Services;

public interface IVideoXpertService
{
    Task<CameraResponse> ListCamerasAsync();
    Task<ClipsResponse> ListClipsAsync(string cameraId);
    Task<ExportResponse> CreateExportAsync(ExportRequest request);
    Task<ExportResponse> CreateAndDownloadExportAsync(ExportRequest request);
    Task<ExportDownloadResponse> CreateAndGetDownloadUrlAsync(ExportRequest request);
    Task<ExportResponse> GetExportStatusAsync(string exportId);
    Task<ExportDownloadResponse> GetExportDownloadUrlAsync(string exportId);
} 