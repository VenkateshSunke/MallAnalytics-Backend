# VideoXpert API

A REST API for VideoXpert video export functionality, built with .NET 8. This API provides endpoints for listing cameras, viewing video clips, and creating/downloading video exports from a VideoXpert system.

## Features

- **Camera Management**: List all available cameras in the VideoXpert system
- **Clip Viewing**: View all available video clips for any camera with time ranges and duration information
- **Video Export**: Create video exports for specific time ranges
- **Download URLs**: Generate direct download URLs for exports (recommended for large files)
- **Export Status**: Check the status of ongoing exports

## Prerequisites

- .NET 8 SDK
- Access to a VideoXpert system
- Valid VideoXpert SDK license

## Configuration

### Environment Variables (Recommended)

Copy `.env.example` to `.env` and configure your VideoXpert system details:

```env
VIDEOXPERT_SERVER_IP=your-videoxpert-server-ip
VIDEOXPERT_SERVER_PORT=443
VIDEOXPERT_USERNAME=your-username
VIDEOXPERT_PASSWORD=your-password
VIDEOXPERT_LICENSE_KEY=your-license-key
VIDEOXPERT_USE_SSL=true
VIDEOXPERT_TIMEOUT_MINUTES=30
VIDEOXPERT_OUTPUT_DIRECTORY=./exports
```

### Legacy Configuration (appsettings.json)

You can also use `appsettings.json` with environment variable references:

```json
{
  "VideoXpert": {
    "ServerIp": "${VIDEOXPERT_SERVER_IP}",
    "ServerPort": "${VIDEOXPERT_SERVER_PORT}",
    "Username": "${VIDEOXPERT_USERNAME}",
    "Password": "${VIDEOXPERT_PASSWORD}",
    "LicenseKey": "${VIDEOXPERT_LICENSE_KEY}",
    "UseSsl": "${VIDEOXPERT_USE_SSL}",
    "TimeoutMinutes": "${VIDEOXPERT_TIMEOUT_MINUTES}",
    "OutputDirectory": "${VIDEOXPERT_OUTPUT_DIRECTORY}"
  }
}
```

## API Endpoints

### Health Check

#### `GET /api/health`
Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "service": "VideoXpert API"
}
```

### Cameras

#### `GET /api/cameras`
List all available cameras in the VideoXpert system.

**Response:**
```json
{
  "success": true,
  "totalCameras": 5,
  "cameras": [
    {
      "id": "camera-001",
      "name": "Front Door Camera",
      "state": "Online",
      "type": "IpCamera"
    }
  ]
}
```

#### `GET /api/cameras/{cameraId}/clips`
List all video clips for a specific camera.

**Parameters:**
- `cameraId` (string): Camera ID or name

**Response:**
```json
{
  "success": true,
  "camera": {
    "id": "camera-001",
    "name": "Front Door Camera",
    "state": "Online",
    "type": "IpCamera"
  },
  "summary": {
    "totalClips": 10,
    "earliestRecording": "2024-01-01 00:00:00",
    "latestRecording": "2024-01-15 23:59:59",
    "totalRecordingHours": 150.5,
    "totalSpanDays": 14.0
  },
  "clips": [
    {
      "startTime": "2024-01-15 08:00:00",
      "endTime": "2024-01-15 09:00:00",
      "startTimeUtc": "2024-01-15 08:00:00",
      "endTimeUtc": "2024-01-15 09:00:00",
      "durationMinutes": 60.0,
      "durationHours": 1.0
    }
  ]
}
```

### Exports

#### `POST /api/exports`
Create a video export for a specific camera and time range.

**Request Body:**
```json
{
  "cameraId": "camera-001",
  "startTime": "2024-01-15 08:00:00",
  "endTime": "2024-01-15 09:00:00",
  "exportPassword": "optional-password",
  "zipOnly": false
}
```

**Response:**
```json
{
  "success": true,
  "exportId": "export-123",
  "exportName": "API_Export_FrontDoor_20240115_080000_20240115_090000",
  "status": "InProgress",
  "fileSizeKb": 1024000
}
```

#### `POST /api/exports/create-download-url` (Recommended)
Create a video export and get a direct download URL.

**Request Body:**
```json
{
  "cameraId": "camera-001",
  "startTime": "2024-01-15 08:00:00",
  "endTime": "2024-01-15 09:00:00",
  "exportPassword": "optional-password",
  "zipOnly": true
}
```

**Response:**
```json
{
  "success": true,
  "exportId": "export-123",
  "exportName": "API_Export_FrontDoor_20240115_080000_20240115_090000",
  "downloadUrl": "https://videoxpert-server/api/exports/export-123/download",
  "fileSizeKb": 1024000,
  "instructions": "Use this URL to download the export file directly. Include the following headers: X-Serenity-User (base64 encoded username) and X-Serenity-Password (base64 encoded password)."
}
```

#### `GET /api/exports/{exportId}/download-url`
Get the download URL for an existing export.

**Parameters:**
- `exportId` (string): Export ID returned from create export

**Response:**
```json
{
  "success": true,
  "exportId": "export-123",
  "exportName": "API_Export_FrontDoor_20240115_080000_20240115_090000",
  "downloadUrl": "https://videoxpert-server/api/exports/export-123/download",
  "fileSizeKb": 1024000,
  "instructions": "Use this URL to download the export file directly. Include the following headers: X-Serenity-User (base64 encoded username) and X-Serenity-Password (base64 encoded password)."
}
```

#### `GET /api/exports/auth-headers`
Get the authentication headers needed for downloading exports.

**Response:**
```json
{
  "success": true,
  "headers": {
    "xSerenityUser": "dXNlcm5hbWU=",
    "xSerenityPassword": "cGFzc3dvcmQ="
  },
  "instructions": "Include these headers when downloading: 'X-Serenity-User' and 'X-Serenity-Password'"
}
```

#### `GET /api/exports/{exportId}/status`
Get the status of an existing export.

**Parameters:**
- `exportId` (string): Export ID returned from create export

**Response:**
```json
{
  "success": true,
  "exportId": "export-123",
  "exportName": "API_Export_FrontDoor_20240115_080000_20240115_090000",
  "status": "Successful",
  "fileSizeKb": 1024000,
  "statusReason": null
}
```

## Downloading Large Files

For large export files (hundreds of GB), use the download URL approach:

1. **Create Export with Download URL**:
   ```bash
   curl -X POST "http://localhost:5183/api/exports/create-download-url" \
     -H "Content-Type: application/json" \
     -d '{
       "cameraId": "camera-001",
       "startTime": "2024-01-15 08:00:00",
       "endTime": "2024-01-15 09:00:00"
     }'
   ```

2. **Get Authentication Headers**:
   ```bash
   curl "http://localhost:5183/api/exports/auth-headers"
   ```

3. **Download the File**:
   ```bash
   curl -H "X-Serenity-User: dXNlcm5hbWU=" \
        -H "X-Serenity-Password: cGFzc3dvcmQ=" \
        -o export.zip \
        "https://videoxpert-server/api/exports/export-123/download"
   ```

## Time Format

All time parameters should be in the format: `yyyy-MM-dd HH:mm:ss`

Examples:
- `2024-01-15 08:30:00`
- `2024-12-31 23:59:59`

## Error Responses

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error description"
}
```

## Running the API

1. Clone the repository
2. Copy `.env.example` to `.env` and configure your VideoXpert settings
3. Run the application:

```bash
cd videoxpert-api/src/videoxpert
dotnet run -p:Platform=x64
```

4. Access the API at `http://localhost:5183` (or the configured port)
5. View API documentation at `http://localhost:5183/swagger`

## Development

For development, you can override settings in `appsettings.Development.json`:

```json
{
  "VideoXpert": {
    "OutputDirectory": "./exports-dev"
  }
}
```

## Security

- The `.env` file contains sensitive information and should never be committed to version control
- Use the `.env.example` file as a template for other developers
- Authentication credentials are base64 encoded for download URLs

## Dependencies

- **VxSdk.NET**: VideoXpert SDK for .NET
- **Swashbuckle.AspNetCore**: Swagger/OpenAPI documentation
- **DotNetEnv**: Environment variable loading from .env files
- **System.Text.Json**: JSON serialization 