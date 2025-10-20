# pull.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Write-Host "🔄 正在从远程仓库拉取最新内容..." -ForegroundColor Cyan

# 确保在脚本所在目录执行
Set-Location $PSScriptRoot

# 检查是否存在 Git 仓库
if (!(Test-Path ".git")) {
    Write-Host "⚠️ 当前目录不是 Git 仓库，请确认路径正确。" -ForegroundColor Yellow
    exit
}

# 自动检测当前分支
$branch = git rev-parse --abbrev-ref HEAD
git pull origin $branch

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 拉取完成，博客已同步到最新版本！" -ForegroundColor Green
} else {
    Write-Host "❌ 拉取失败，请检查网络或仓库设置。" -ForegroundColor Red
}
