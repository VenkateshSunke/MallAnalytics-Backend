# Use official .NET 9 SDK image for building on Windows Nano Server 2025
FROM mcr.microsoft.com/dotnet/sdk:9.0-nanoserver-ltsc2025 AS build

# Set shell to cmd to avoid user issues
SHELL ["cmd", "/S", "/C"]

# Set working directory
WORKDIR /app

# Copy solution and project files
COPY videoxpertAPI.sln ./
COPY src/videoxpert/*.csproj ./src/videoxpert/

# Restore dependencies
RUN dotnet restore

# Copy source code
COPY src/ ./src/

# Build the application in Release mode
RUN dotnet build src/videoxpert/videoxpert.csproj -c Release -o /app/build --no-restore

# Publish the application
RUN dotnet publish src/videoxpert/videoxpert.csproj -c Release -o /app/publish --no-restore

# Use official .NET 9 runtime image for production on Windows Nano Server 2025
FROM mcr.microsoft.com/dotnet/aspnet:9.0-nanoserver-ltsc2025 AS runtime

# Set shell to cmd
SHELL ["cmd", "/S", "/C"]

# Set working directory
WORKDIR /app

# Copy published application from build stage
COPY --from=build /app/publish .

# Create exports directory
RUN mkdir exports

# Expose port
EXPOSE 5183

# Run the application
CMD ["dotnet", "videoxpert.dll"]