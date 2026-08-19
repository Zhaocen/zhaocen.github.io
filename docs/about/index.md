# 关于本站

## 技术栈

| 组件 | 用途 |
|------|------|
| [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | 静态站点主题与生成器 |
| [GitHub Actions](https://github.com/features/actions) | 自动构建与发布 |
| [GitHub Pages](https://pages.github.com/) | 站点托管 |
| 阿里云 ECS | 写作与本地预览环境 |

## 部署链路

```mermaid
graph TD
    A[阿里云 ECS<br/>编辑 Markdown] -->|docker 预览 :8000| B[实时查看效果]
    A -->|git push| C[GitHub main 分支]
    C -->|Actions 自动构建| D[mkdocs build --strict]
    D --> E[GitHub Pages]
    E --> F[https://zhaocen.github.io/]
```

具体操作步骤见 [写作与发布流程](workflow.md)。

## 联系

- GitHub: [@Zhaocen](https://github.com/Zhaocen)
