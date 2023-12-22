# embedchain

## git repository

> https://github.com/leesenlen/embedchain.git


## poetry 安装

```shell
# 安装poetry
curl -sSL https://install.python-poetry.org | python -

# https://juejin.cn/post/6999405667261874183
# poetry 文档
# poetry env info
# poetry run python app.py
poetry --version
```

## 使用poetry创建虚拟环境

```shell
# Please DO NOT use pip or conda to install the dependencies. Instead, use poetry:
# `--all-extras` 参数用于安装项目的所有附加依赖项。附加依赖项是在 `pyproject.toml` 文件中定义的
poetry install --all-extras
# or
poetry install --with dev

#activate
poetry shell
```

## Install ipykernel for Jupyter Notebook

```shell
# poetry run python myscript.py

# 安装ipykernel
poetry add ipykernel
# Jupyter Notebook kernel
# poetry run python -m ipykernel install --user --name=my_poetry_kernel  

# vscode 选择对应的环境.venv
```