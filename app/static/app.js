const statusEl = document.getElementById("status");
const messagesEl = document.getElementById("messages");
const fileInput = document.getElementById("file-input");
const uploadBtn = document.getElementById("upload-btn");
const reindexBtn = document.getElementById("reindex-btn");
const chatForm = document.getElementById("chat-form");
const questionInput = document.getElementById("question-input");

function setStatus(message) {
  statusEl.textContent = message;
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  let payload = null;

  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    const detail = payload?.detail || `请求失败: ${response.status}`;
    throw new Error(detail);
  }

  return payload ?? {};
}

function buildCitation(citation, index) {
  const wrapper = document.createElement("details");
  wrapper.className = "citation";
  if (index === 0) {
    wrapper.open = true;
  }

  const summary = document.createElement("summary");
  const bits = [`[${index + 1}] ${citation.file_name}`];
  if (citation.page_label) {
    bits.push(`第 ${citation.page_label} 页`);
  }
  if (typeof citation.score === "number") {
    bits.push(`相关度 ${citation.score.toFixed(3)}`);
  }
  summary.textContent = bits.join(" · ");
  wrapper.appendChild(summary);

  const body = document.createElement("div");
  body.className = "citation-body";

  const snippet = document.createElement("p");
  snippet.textContent = citation.snippet;
  body.appendChild(snippet);

  if (citation.file_path) {
    const path = document.createElement("p");
    path.className = "citation-path";
    path.textContent = citation.file_path;
    body.appendChild(path);
  }

  wrapper.appendChild(body);
  return wrapper;
}

function addMessage(role, content, citations = []) {
  const item = document.createElement("article");
  item.className = `message ${role}`;

  const text = document.createElement("div");
  text.className = "message-text";
  text.textContent = content;
  item.appendChild(text);

  if (role === "assistant" && citations.length) {
    const refs = document.createElement("section");
    refs.className = "citations";

    const title = document.createElement("h3");
    title.textContent = `引用 (${citations.length})`;
    refs.appendChild(title);

    citations.forEach((citation, index) => {
      refs.appendChild(buildCitation(citation, index));
    });
    item.appendChild(refs);
  }

  messagesEl.appendChild(item);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    setStatus("先选择要上传的文件");
    return;
  }

  const formData = new FormData();
  for (const file of fileInput.files) {
    formData.append("files", file);
  }

  uploadBtn.disabled = true;
  setStatus("正在上传文档...");

  try {
    const payload = await requestJson("/api/documents/upload", {
      method: "POST",
      body: formData,
    });
    setStatus(payload.message || "上传完成");
  } catch (error) {
    setStatus(error.message);
  } finally {
    uploadBtn.disabled = false;
  }
});

reindexBtn.addEventListener("click", async () => {
  reindexBtn.disabled = true;
  setStatus("正在重建索引...");

  try {
    const payload = await requestJson("/api/index/rebuild", { method: "POST" });
    setStatus(payload.message || "索引完成");
  } catch (error) {
    setStatus(error.message);
  } finally {
    reindexBtn.disabled = false;
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    return;
  }

  addMessage("user", question);
  questionInput.value = "";
  setStatus("正在回答...");
  questionInput.disabled = true;

  try {
    const payload = await requestJson("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: question }),
    });
    addMessage("assistant", payload.answer || "请求失败", payload.citations || []);
    setStatus("完成");
  } catch (error) {
    addMessage("assistant", error.message);
    setStatus(error.message);
  } finally {
    questionInput.disabled = false;
    questionInput.focus();
  }
});
