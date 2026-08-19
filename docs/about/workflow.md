# 写作与发布流程

本页是日常更新知识库的操作手册。

## 一、环境说明

| 位置 | 路径 / 地址 | 说明 |
|------|-------------|------|
| 阿里云工作副本 | `~/knowledge-base` | 写作与预览的主目录 |
| 预览地址 | `http://localhost:8000` | 需在安全组放行 8000 端口 |
| 线上站点 | <https://zhaocen.github.io/> | GitHub Actions 自动发布 |

## 二、启动预览

登录阿里云机器后执行：

```bash
cd ~/knowledge-base
./preview.sh start
```

浏览器打开 `http://localhost:8000`，改动 Markdown 保存后页面自动刷新。

其他子命令：

```bash
./preview.sh stop     # 停止预览容器
./preview.sh logs     # 查看容器日志
./preview.sh status   # 查看运行状态
```

!!! warning "预览完记得关掉"

    机器只有 1.6G 内存，长期挂着预览容器会挤占资源。

## 三、新增一篇文章

1. 在 `docs/` 下对应板块创建 Markdown 文件，例如 `docs/ai/kv-cache.md`
2. 文件顶部可选地写 front matter 加标签：

    ```yaml
    ---
    tags:
      - vLLM
      - 显存优化
    ---
    ```

3. 在 `mkdocs.yml` 的 `nav:` 中登记，让它出现在侧边栏：

    ```yaml
    nav:
      - AI 推理:
          - ai/index.md
          - vLLM 笔记: ai/vllm.md
          - KV Cache 管理: ai/kv-cache.md   # 新增这行
    ```

!!! tip "不登记会怎样"

    页面仍会被构建、也能被搜索到，但不出现在侧边栏导航中。

## 四、发布上线

```bash
cd ~/knowledge-base
git add -A
git commit -m "docs: 新增 KV Cache 管理笔记"
git push
```

推送后 GitHub Actions 自动构建，约 1 分钟后线上生效。查看构建状态：

<https://github.com/Zhaocen/zhaocen.github.io/actions>

## 五、常用写法速查

=== "提示框"

    ```markdown
    !!! note "标题"

        缩进四个空格的内容。

    !!! warning "警告"
    !!! tip "技巧"
    !!! danger "危险"

    ??? note "默认折叠"
    ```

=== "代码与标签页"

    ````markdown
    ```python title="示例" hl_lines="2"
    def hello():
        print("高亮这行")
    ```

    === "标签 A"
        内容 A
    === "标签 B"
        内容 B
    ````

=== "图表"

    ````markdown
    ```mermaid
    graph LR
        A --> B
    ```
    ````

=== "其他"

    ```markdown
    ++ctrl+c++              按键样式
    - [x] 已完成            任务清单
    ==高亮== ~~删除~~       文本标记
    $E = mc^2$              行内公式（需 \( \) 包裹）
    ```

## 六、构建失败排查

CI 使用 `mkdocs build --strict`，任何警告都会导致失败。最常见原因：

| 报错 | 原因 | 处理 |
|------|------|------|
| `is not found among documentation files` | `nav` 里写了不存在的文件 | 检查路径拼写 |
| `contains a link ... not found` | 内部链接指向的文件不存在 | 用相对路径指向真实 `.md` 文件 |
| `Config value 'plugins'` | 依赖未安装 | 检查 `requirements.txt` |

本地复现 CI 的构建结果：

```bash
docker run --rm -v ~/knowledge-base:/docs \
  squidfunk/mkdocs-material:latest build --strict
```
