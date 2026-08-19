---
tags:
  - Docker
  - 部署
---

# 容器与部署

## 本站的预览环境

本知识库在阿里云机器上通过官方镜像做实时预览：

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v ~/knowledge-base:/docs \
  squidfunk/mkdocs-material:latest
```

保存任意 Markdown 文件后，浏览器会自动热重载。

## 常用命令速查

=== "容器"

    ```bash
    docker ps -a              # 查看所有容器
    docker logs -f <name>     # 跟踪日志
    docker exec -it <name> sh # 进入容器
    docker rm -f <name>       # 强制删除
    ```

=== "镜像"

    ```bash
    docker images             # 列出镜像
    docker pull <image>       # 拉取镜像
    docker rmi <image>        # 删除镜像
    docker system df          # 查看磁盘占用
    ```

=== "清理"

    ```bash
    docker system prune -a    # 清理未使用的镜像与缓存
    docker volume prune       # 清理悬空卷
    ```

!!! warning "小内存机器注意"

    2 核 1.6G 的实例上，同时运行多个容器容易触发 OOM。
    预览容器用完记得停掉：`docker stop mkdocs-preview`。
