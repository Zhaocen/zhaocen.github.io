# 在官方 Material for MkDocs 镜像基础上补齐本项目用到的插件
FROM squidfunk/mkdocs-material:latest

# 使用阿里云 PyPI 镜像源，避免国内直连 pypi.org 缓慢
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
 && pip config set global.trusted-host mirrors.aliyun.com

# git-revision-date-localized: 页面底部的最后更新时间
# jieba: 中文搜索分词
RUN pip install --no-cache-dir --break-system-packages \
        mkdocs-git-revision-date-localized-plugin \
        jieba \
    || pip install --no-cache-dir \
        mkdocs-git-revision-date-localized-plugin \
        jieba
