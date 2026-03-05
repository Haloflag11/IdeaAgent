#!/bin/bash

echo "========================================"
echo "  IdeaAgent 安装脚本"
echo "========================================"
echo ""

# 检查 Python 版本
echo "检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    echo "错误：未找到 Python"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ $PYTHON_VERSION"
echo ""

# 创建虚拟环境
echo "创建虚拟环境..."
if [ -d ".venv" ]; then
    echo "虚拟环境已存在，跳过创建"
else
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo "错误：创建虚拟环境失败"
        exit 1
    fi
    echo "✓ 虚拟环境创建成功"
fi
echo ""

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate
echo "✓ 虚拟环境已激活"
echo ""

# 安装依赖
echo "安装项目依赖..."
pip install -e .
if [ $? -ne 0 ]; then
    echo "警告：安装依赖时出现问题，但继续执行"
fi
echo "✓ 依赖安装完成"
echo ""

# 检查 .env 文件
echo "检查环境配置..."
if [ -f ".env" ]; then
    echo "✓ .env 文件已存在"
else
    echo "创建 .env 文件..."
    cp .env.example .env
    echo "✓ .env 文件已创建"
    echo ""
    echo "========================================"
    echo "  重要提示"
    echo "========================================"
    echo "请编辑 .env 文件并设置以下变量："
    echo "  - OPENAI_API_KEY: 你的 OpenAI API 密钥"
    echo "  - MONGODB_URI: MongoDB 连接 URI（可选）"
    echo ""
    echo "编辑完成后，运行以下命令启动 IdeaAgent："
    echo "  source .venv/bin/activate"
    echo "  IdeaAgent"
    echo "========================================"
    echo ""
fi

echo ""
echo "========================================"
echo "  安装完成！"
echo "========================================"
echo ""

if [ -f ".env" ]; then
    echo "启动 IdeaAgent..."
    echo ""
    IdeaAgent
fi
