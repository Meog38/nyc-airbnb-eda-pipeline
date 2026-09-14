$directories = @(
    "data/raw",
    "reports/figures",
    "src/nyc_airbnb_eda",
    "tests"
)

foreach ($directory in $directories) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

Write-Host "Project directories are ready."
