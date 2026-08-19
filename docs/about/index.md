# 关于本站

## 技术栈

| 组件 | 用途 |
|------|------|
| [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) | 静态站点主题与生成器 |
| [GitHub Actions](https://github.com/features/actions) | 自动构建与发布 |
| [GitHub Pages](https://pages.github.com/) | 站点托管 |
| Docker | 本地预览环境 |

## 部署链路

```mermaid
graph TD
    A[本地编辑 Markdown] -->|docker 预览| B[实时查看效果]
    A -->|git push| C[GitHub main 分支]
    C -->|Actions 自动构建| D[mkdocs build --strict]
    D --> E[GitHub Pages]
    E --> F[https://zhaocen.github.io/]
```

## 本板块内容

<div class="grid cards" markdown>

-   __[写作与发布流程](workflow.md)__

    ---

    从新建文章到上线的完整操作步骤，含常用 Markdown 写法速查。

-   __[容器与部署](docker.md)__

    ---

    Docker 常用操作，以及本站预览环境的搭建方式。

</div>

## 联系

- GitHub: [@Zhaocen](https://github.com/Zhaocen)
