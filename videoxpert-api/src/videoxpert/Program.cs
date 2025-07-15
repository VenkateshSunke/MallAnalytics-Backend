using DotNetEnv;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using videoxpert.Domain;
using videoxpert.Services;

// Load environment variables from .env file
Env.Load();

var builder = WebApplication.CreateBuilder(args);

// Add services to the container
builder.Services.AddControllers();

// Configure VideoXpert settings using environment variables
builder.Services.Configure<VideoXpertConfiguration>(options =>
{
    options.ServerIp = Environment.GetEnvironmentVariable("VIDEOXPERT_SERVER_IP") ?? "localhost";
    options.ServerPort = int.Parse(Environment.GetEnvironmentVariable("VIDEOXPERT_SERVER_PORT") ?? "443");
    options.Username = Environment.GetEnvironmentVariable("VIDEOXPERT_USERNAME") ?? "";
    options.Password = Environment.GetEnvironmentVariable("VIDEOXPERT_PASSWORD") ?? "";
    options.LicenseKey = Environment.GetEnvironmentVariable("VIDEOXPERT_LICENSE_KEY") ?? "";
    options.UseSsl = bool.Parse(Environment.GetEnvironmentVariable("VIDEOXPERT_USE_SSL") ?? "true");
    options.TimeoutMinutes = int.Parse(Environment.GetEnvironmentVariable("VIDEOXPERT_TIMEOUT_MINUTES") ?? "30");
    options.OutputDirectory = Environment.GetEnvironmentVariable("VIDEOXPERT_OUTPUT_DIRECTORY") ?? "./exports";
});

// Register VideoXpert service
builder.Services.AddScoped<IVideoXpertService, VideoXpertService>();

// Add API Explorer and Swagger
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new Microsoft.OpenApi.Models.OpenApiInfo
    {
        Title = "VideoXpert API",
        Version = "v1",
        Description = "REST API for VideoXpert video export functionality"
    });
});

// Add CORS policy for frontend access
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyMethod()
              .AllowAnyHeader();
    });
});

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "VideoXpert API v1");
    });
}

app.UseHttpsRedirection();
app.UseCors();
app.UseAuthorization();
app.MapControllers();

app.Run();

