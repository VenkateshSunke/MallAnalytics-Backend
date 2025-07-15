namespace videoxpert.Domain;

public class VideoXpertConfiguration
{
    public string ServerIp { get; set; } = string.Empty;
    public int ServerPort { get; set; }
    public string Username { get; set; } = string.Empty;
    public string Password { get; set; } = string.Empty;
    public string LicenseKey { get; set; } = string.Empty;
    public bool UseSsl { get; set; }
    public int TimeoutMinutes { get; set; }
    public string OutputDirectory { get; set; } = string.Empty;
} 