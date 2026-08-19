# zhaocen.github.io

基于 [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) 搭建的个人 AI 知识库。

线上地址：<https://zhaocen.github.io/>

## 目录结构

```
.
├── docs/                     # 所有 Markdown 内容
│   ├── index.md              # 首页
│   ├── llm/                  # 大模型
│   ├── ai/                   # AI 推理
│   ├── benchmark/            # 模型评测
│   ├── tags.md               # 标签索引
│   ├── about/                # 关于、写作指南、容器与部署
│   ├── stylesheets/          # 自定义 CSS
│   └── javascripts/          # 自定义 JS（MathJax 配置）
├── mkdocs.yml                # 站点配置与导航
├── requirements.txt          # 构建依赖
├── preview.sh                # 本地预览管理脚本
├── docker/Dockerfile         # 预览用扩展镜像
└── .github/workflows/        # GitHub Actions 自动部署
```

## 本地预览

```bash
./preview.sh start     # 启动，访问 http://localhost:8000
./preview.sh stop      # 停止
./preview.sh build     # 以 CI 的 --strict 模式验证构建
```

## 发布

推送到 `main` 分支即自动触发 GitHub Actions 构建并发布到 GitHub Pages。

```bash
git add -A && git commit -m "docs: ..." && git push
```

详细写作规范见 [写作与发布流程](docs/about/workflow.md)。
