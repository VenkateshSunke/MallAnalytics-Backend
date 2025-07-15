namespace videoxpert.Domain;

public class CameraResponse
{
    public bool Success { get; set; }
    public int TotalCameras { get; set; }
    public List<Camera> Cameras { get; set; } = new();
    public string? Error { get; set; }
}

public class Camera
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;
    public string Type { get; set; } = string.Empty;
} 