# PowerShell script to push to GitHub
# Usage: .\push_to_github.ps1 -RepoUrl "https://github.com/username/repo.git"

param(
    [Parameter(Mandatory=$true)]
    [string]$RepoUrl
)

Write-Host "Setting up GitHub remote..." -ForegroundColor Green

# Check if remote already exists
$existingRemote = git remote get-url origin 2>$null
if ($existingRemote) {
    Write-Host "Remote 'origin' already exists: $existingRemote" -ForegroundColor Yellow
    $overwrite = Read-Host "Overwrite? (y/n)"
    if ($overwrite -eq "y") {
        git remote set-url origin $RepoUrl
    } else {
        Write-Host "Keeping existing remote. Exiting." -ForegroundColor Yellow
        exit
    }
} else {
    git remote add origin $RepoUrl
}

Write-Host "Remote added: $RepoUrl" -ForegroundColor Green

# Ensure we're on main branch
git branch -M main

Write-Host "Pushing to GitHub..." -ForegroundColor Green
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
} else {
    Write-Host "Push failed. You may need to:" -ForegroundColor Red
    Write-Host "1. Create the repository on GitHub first" -ForegroundColor Yellow
    Write-Host "2. Authenticate (use Personal Access Token or SSH)" -ForegroundColor Yellow
    Write-Host "3. Check the repository URL" -ForegroundColor Yellow
}

