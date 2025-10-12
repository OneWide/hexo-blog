# push_auto.ps1
$commitMsg = "Auto update on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
git add .
git commit -m "$commitMsg"
git push origin main
Write-Host "✅ 推送完成！" -ForegroundColor Green
