#!/bin/bash
# 启动两个DExperts服务

echo "启动服务1 (端口8402, GPU 0-3)..."
nohup python runapi.py config_service1.yaml > service1.log 2>&1 &
echo "服务1 PID: $!"

sleep 2

echo "启动服务2 (端口8403, GPU 4-7)..."
nohup python runapi.py config_service2.yaml > service2.log 2>&1 &
echo "服务2 PID: $!"

echo ""
echo "✅ 两个服务已启动！"
echo ""
echo "检查状态："
echo "  curl http://localhost:8402/health"
echo "  curl http://localhost:8403/health"
echo ""
echo "查看日志："
echo "  tail -f service1.log"
echo "  tail -f service2.log"
echo ""
echo "停止服务："
echo "  pkill -f 'python runapi.py'"
