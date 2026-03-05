# IdeaAgent 快速启动脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  IdeaAgent 安装脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python 版本
Write-Host "检查 Python 版本..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：未找到 Python 或 Python 版本低于 3.11" -ForegroundColor Red
    Write-Host "请安装 Python 3.11 或更高版本" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
Write-Host ""

# 创建虚拟环境
Write-Host "创建虚拟环境..." -ForegroundColor Yellow
if (Test-Path ".venv") {
    Write-Host "虚拟环境已存在，跳过创建" -ForegroundColor Yellow
} else {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误：创建虚拟环境失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ 虚拟环境创建成功" -ForegroundColor Green
}
Write-Host ""

# 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Yellow
. .\.venv\Scripts\Activate.ps1
Write-Host "✓ 虚拟环境已激活" -ForegroundColor Green
Write-Host ""

# 安装依赖
Write-Host "安装项目依赖..." -ForegroundColor Yellow
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "警告：安装依赖时出现问题，但继续执行" -ForegroundColor Yellow
}
Write-Host "✓ 依赖安装完成" -ForegroundColor Green
Write-Host ""

# 检查 .env 文件
Write-Host "检查环境配置..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env 文件已存在" -ForegroundColor Green
} else {
    Write-Host "创建 .env 文件..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "✓ .env 文件已创建" -ForegroundColor Green
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  重要提示" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "请编辑 .env 文件并设置以下变量：" -ForegroundColor Yellow
    Write-Host "  - OPENAI_API_KEY: 你的 OpenAI API 密钥" -ForegroundColor White
    Write-Host "  - MONGODB_URI: MongoDB 连接 URI（可选）" -ForegroundColor White
    Write-Host ""
    Write-Host "编辑完成后，运行以下命令启动 IdeaAgent：" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host "  IdeaAgent" -ForegroundColor White
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  安装完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

if (Test-Path ".env") {
    Write-Host "启动 IdeaAgent..." -ForegroundColor Yellow
    Write-Host ""
    IdeaAgent
}
