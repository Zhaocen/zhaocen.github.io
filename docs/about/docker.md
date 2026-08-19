---
tags:
  - Docker
  - 部署
---

# 容器与部署

## 本站的预览环境

本知识库通过 Docker 镜像做实时预览，在项目根目录下执行：

```bash
docker run --rm -it \
  -p 8000:8000 \
  -v "$(pwd):/docs" \
  squidfunk/mkdocs-material:latest
```

保存任意 Markdown 文件后，浏览器会自动热重载。

!!! note "为什么本项目用的是自建镜像"

    官方镜像不包含 `git-revision-date-localized`（页面更新时间）与 `jieba`（中文搜索分词）
    两个插件，因此 `docker/Dockerfile` 在官方镜像基础上补装了它们，
    由 `preview.sh` 自动构建，无需手动干预。

## 国内环境加速

在国内网络下构建镜像时，直连 PyPI 官方源往往很慢，建议在 Dockerfile 中先换源：

```dockerfile
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
 && pip config set global.trusted-host mirrors.aliyun.com
```

同样的构建过程，换源后耗时通常从数分钟降到半分钟内。

!!! tip "CI 环境不需要换源"

    GitHub Actions 运行在海外网络，直连官方源反而更快，因此
    `requirements.txt` 保持默认源即可。

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

    在内存有限的机器上同时运行多个容器容易触发 OOM，
    预览用完记得停掉：`./preview.sh stop`。
