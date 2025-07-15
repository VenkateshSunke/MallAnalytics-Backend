namespace videoxpert.Domain;

public class ClipsResponse
{
    public bool Success { get; set; }
    public Camera? Camera { get; set; }
    public ClipsSummary? Summary { get; set; }
    public List<VideoClip> Clips { get; set; } = new();
    public string? Error { get; set; }
}

public class ClipsSummary
{
    public int TotalClips { get; set; }
    public string? EarliestRecording { get; set; }
    public string? LatestRecording { get; set; }
    public double TotalRecordingHours { get; set; }
    public double TotalSpanDays { get; set; }
}

public class VideoClip
{
    public string StartTime { get; set; } = string.Empty;
    public string EndTime { get; set; } = string.Empty;
    public string StartTimeUtc { get; set; } = string.Empty;
    public string EndTimeUtc { get; set; } = string.Empty;
    public double DurationMinutes { get; set; }
    public double DurationHours { get; set; }
} 