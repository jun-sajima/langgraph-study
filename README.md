# LangGraph Study

LangGraphの学習用プロジェクトです。marimo（リアクティブPythonノートブック）とuv（Python パッケージマネージャ）を使用して、LangGraphの機能を学習・実験します。

## 技術スタック

- **[LangGraph](https://github.com/langchain-ai/langgraph)**: LangChainベースのステートフルなマルチアクター アプリケーション構築フレームワーク
- **[marimo](https://marimo.io/)**: リアクティブなPythonノートブック環境。Jupyter Notebookの代替として、よりモダンで保守性の高いノートブックを提供
- **[uv](https://github.com/astral-sh/uv)**: Rustで書かれた高速なPythonパッケージマネージャ。pip/pip-tools/virtualenvの代替

## セットアップ

### DevContainerを使用する場合（推奨）

1. VSCodeで プロジェクトを開く
2. コマンドパレット（`Ctrl+Shift+P` / `Cmd+Shift+P`）から `Dev Containers: Reopen in Container` を実行
3. コンテナが起動したら、依存関係をインストール:
   ```bash
   uv sync
   ```

### ローカル環境にセットアップする場合

1. uvをインストール:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. 依存関係をインストール:
   ```bash
   uv sync
   ```

3. 環境変数を設定:
   ```bash
   cp .env.example .env
   # .envファイルにOpenAI APIキーなど必要な情報を記入
   ```

## 使い方

### marimoノートブックの起動

marimoは、リアクティブなPythonノートブック環境です。セルの変更が自動的に依存するセルに反映されます。

```bash
# marimoを起動（ファイル選択画面が表示される）
uv run marimo edit

# または特定のファイルを直接開く
uv run marimo edit src/marimo_test.py

# 特定のポートで起動する場合
uv run marimo edit --host 0.0.0.0 --port 2718

# 読み取り専用アプリとして起動
uv run marimo run src/marimo_test.py
```

起動後、ブラウザで表示されるURL（デフォルトは `http://localhost:2718`）にアクセスします（DevContainer環境では自動的にポートフォワードされます）。

### Pythonスクリプトの実行

```bash
# uvを使用してスクリプトを実行
uv run python src/work_1.py
```

### テストの実行

```bash
uv run pytest
```

## プロジェクト構造

```
.
├── .devcontainer/          # VSCode DevContainer設定
│   ├── devcontainer.json   # DevContainer設定ファイル
│   ├── docker-compose.yml  # Docker Compose設定
│   └── Dockerfile          # コンテナイメージ定義
├── src/                    # ソースコードディレクトリ
│   ├── marimo_test.py      # marimoのテストノートブック
│   ├── work_1.py           # 学習用スクリプト1
│   ├── work_2.py           # 学習用スクリプト2
│   ├── work_3.py           # 学習用スクリプト3
│   └── work_4.py           # 学習用スクリプト4
├── product.py              # プロダクションコード
├── test_product.py         # テストコード
├── pyproject.toml          # プロジェクト設定・依存関係定義
├── uv.lock                 # ロックファイル（依存関係の固定）
└── .env.example            # 環境変数のサンプル
```

## 主な依存関係

- `langchain-openai`: OpenAI APIとLangChainの統合
- `langgraph`: ステートフルなマルチアクターアプリケーション構築
- `marimo`: リアクティブノートブック環境
- `matplotlib`: グラフ描画
- `numpy`: 数値計算
- `pytest`: テストフレームワーク
- `python-dotenv`: 環境変数管理

## Tips

### uvの基本コマンド

```bash
# 依存関係をインストール/更新
uv sync

# パッケージを追加
uv add <package-name>

# パッケージを削除
uv remove <package-name>

# スクリプトを実行
uv run python <script.py>

# uvで管理されたツールを実行
uv run <tool-name>
```

### marimoの特徴

- **リアクティブ実行**: セルの変更が自動的に依存するセルに伝播
- **Pythonファイルとして保存**: `.py`ファイルとして保存されるため、Git管理が容易
- **自動依存解析**: セル間の依存関係を自動的に解析
- **モジュールとして実行可能**: ノートブックをPythonモジュールとして実行可能

### DevContainerの特徴

- **一貫した開発環境**: チーム全体で同じ開発環境を共有
- **簡単なセットアップ**: VSCodeで開くだけで環境構築完了
- **ポートフォワード**: コンテナ内のポート（2718）が自動的にローカルホストにフォワード
- **VSCode拡張機能**: Python、Ruff、marimoなどの拡張機能が自動インストール

## DeepWikiでコードを読む

このプロジェクトは、DeepWikiを使って視覚的にコードを探索することができます。

GitHubのURLを以下のように変換することで、DeepWiki化できます：

```
https://github.com/<username>/<repository>
↓
https://deepwiki.com/<username>/<repository>
```

DeepWikiを使うと、コードベースの構造を視覚的に理解しやすくなります。

## ライセンス

MIT License
