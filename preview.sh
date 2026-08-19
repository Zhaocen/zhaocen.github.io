#!/usr/bin/env bash
# ============================================================
#  本地预览管理脚本
#  用法: ./preview.sh {start|stop|restart|status|logs|build|rebuild-image}
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="kb-material:local"
CONTAINER="mkdocs-preview"
PORT="${MKDOCS_PORT:-8000}"

cd "$PROJECT_DIR"

# 确保扩展镜像存在（含 git-revision-date-localized 与 jieba 插件）
ensure_image() {
    if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
        echo "==> 首次运行，构建扩展镜像 $IMAGE ..."
        docker build -t "$IMAGE" -f docker/Dockerfile .
    fi
}

case "${1:-start}" in
    start)
        ensure_image
        if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
            echo "预览已在运行: http://$(hostname -I | awk '{print $1}'):${PORT}"
            exit 0
        fi
        docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
        docker run -d --name "$CONTAINER" \
            -p "${PORT}:8000" \
            -v "${PROJECT_DIR}:/docs" \
            "$IMAGE" \
            serve --dev-addr 0.0.0.0:8000
        echo "==> 预览已启动"
        echo "    本机:  http://127.0.0.1:${PORT}"
        echo "    公网:  http://localhost:${PORT}   (需在安全组放行 ${PORT} 端口)"
        echo "    日志:  ./preview.sh logs"
        ;;
    stop)
        docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "==> 预览已停止" || echo "预览未在运行"
        ;;
    restart)
        "$0" stop; "$0" start
        ;;
    status)
        docker ps -a --filter "name=${CONTAINER}" \
            --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
        ;;
    logs)
        docker logs -f "$CONTAINER"
        ;;
    build)
        # 以 CI 相同的严格模式构建，用于本地验证
        ensure_image
        docker run --rm -v "${PROJECT_DIR}:/docs" "$IMAGE" build --strict
        echo "==> 构建成功，产物位于 ${PROJECT_DIR}/site"
        ;;
    rebuild-image)
        docker build --no-cache -t "$IMAGE" -f docker/Dockerfile .
        echo "==> 镜像已重建"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs|build|rebuild-image}"
        exit 1
        ;;
esac
