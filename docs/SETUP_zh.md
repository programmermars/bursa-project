# 中文安装与使用说明

## 一、台式机（RTX 2060）

1. 安装 Python 3.11 和 Git。
2. 安装 Ollama：https://ollama.com ，装好后在终端运行：
   ```
   ollama pull nomic-embed-text
   ollama pull qwen2.5:7b
   ```
3. 下载项目并安装：
   ```
   git clone https://github.com/programmermars/bursa-project.git
   cd bursa-project
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev]"
   copy .env.example .env
   ```
4. 把年报 PDF 放进 `data\raw\`，文件名格式是 `公司_年份_xxx.pdf`，例如 `KLK_2025_annual_report.pdf`。
5. 依次运行：
   ```
   rag doctor        # 检查环境
   rag ingest        # 建索引
   streamlit run app.py
   ```

## 二、Dell 笔记本（没有独立显卡）

两种做法选一个：

- **用 Groq 免费额度（推荐）**：到 https://console.groq.com 注册，拿免费 key（不用绑卡）。在 `.env` 里设：
  ```
  LLM_PROVIDER=groq
  GROQ_API_KEY=你的key
  ```
  向量化仍用 Ollama 的 `nomic-embed-text`，模型很小，CPU 跑得动。
- **连台式机的 Ollama**：台式机开 Tailscale，运行 `set OLLAMA_HOST=0.0.0.0` 再 `ollama serve`。笔记本 `.env` 里设 `OLLAMA_URL=http://台式机的Tailscale IP:11434`。

## 三、评估（作品最重要的部分）

1. 打开 `eval/questions.csv`，写 20–30 道题。每行包括：编号、公司名、问题、答案所在页码（多页用分号隔开）。
2. 加 3–5 道年报里没有答案的题，页码留空，用来测模型会不会乱编。
3. 运行：
   ```
   rag eval              # 只评估检索，不调用大模型，免费
   rag eval --with-llm   # 再评估引用是否正确、会不会说"找不到"
   ```
4. 结果在 `eval/results.md`，把表格贴进 README 的 Evaluation 部分。

页码要写 PDF 阅读器显示的页数，不是纸面印的页码。

## 四、改完后上传 GitHub

```
git add .
git commit -m "Add evaluation results"
git push
```

## 五、常见问题

| 问题 | 解决 |
|---|---|
| `Could not reach Ollama` | 没开 Ollama，运行 `ollama serve` |
| `Index was built with EMBED_BACKEND=...` | 换了向量模型，要重新 `rag ingest` |
| Groq 报 rate limit | 免费额度每分钟有限，等一分钟，或改用 Ollama |
| 显存不够 | 换成 `OLLAMA_LLM_MODEL=qwen2.5:3b` |
