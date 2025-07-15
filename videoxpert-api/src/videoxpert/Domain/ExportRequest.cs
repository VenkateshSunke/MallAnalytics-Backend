namespace videoxpert.Domain;

public class ExportRequest
{
    public string CameraId { get; set; } = string.Empty;
    public string StartTime { get; set; } = string.Empty;
    public string EndTime { get; set; } = string.Empty;
    public string? ExportPassword { get; set; }
    public bool ZipOnly { get; set; } = false;
}

public class ExportResponse
{
    public bool Success { get; set; }
    public string? ExportId { get; set; }
    public string? ExportName { get; set; }
    public string? Status { get; set; }
    public string? FilePath { get; set; }
    public long? FileSizeKb { get; set; }
    public string? Error { get; set; }
    public string? StatusReason { get; set; }
    public string? DownloadUrl { get; set; }
}

public class ExportDownloadResponse
{
    public bool Success { get; set; }
    public string? ExportId { get; set; }
    public string? ExportName { get; set; }
    public string? DownloadUrl { get; set; }
    public long? FileSizeKb { get; set; }
    public string? Error { get; set; }
    public string? Instructions { get; set; }
}

public class ExportStatus
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public string? StatusReason { get; set; }
    public long FileSizeKb { get; set; }
    public string? DataUri { get; set; }
} 