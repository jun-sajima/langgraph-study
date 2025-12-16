# uvの公式イメージをベースに使用
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# 必要なパッケージをインストール（sudo, git等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    && rm -rf /var/lib/apt/lists/*

# 非rootユーザー（vscode）を作成し、sudo権限を付与
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid 1000 --create-home vscode \
    && echo "vscode ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

# 作業ディレクトリを設定
WORKDIR /workspace

# バイトコードコンパイルを有効化（起動高速化）
ENV UV_COMPILE_BYTECODE=1

# マウントボリューム用にコピーモードを使用（リンクではなく）
ENV UV_LINK_MODE=copy

# uvが自動実行されないようにエントリーポイントをリセット
ENTRYPOINT []

# 非rootユーザーに切り替え
USER vscode
