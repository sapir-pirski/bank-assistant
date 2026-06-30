const form = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const messages = document.querySelector("#messages");
const sources = document.querySelector("#sources");
const statusText = document.querySelector("#statusText");
const sourceMeta = document.querySelector("#sourceMeta");
const sendButton = document.querySelector("#sendButton");
const sessionId = getSessionId();

function getSessionId() {
  const key = "oneZeroChatSessionId";
  const existing = window.localStorage.getItem(key);
  if (existing) {
    return existing;
  }

  const generated =
    window.crypto && window.crypto.randomUUID
      ? window.crypto.randomUUID()
      : `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  window.localStorage.setItem(key, generated);
  return generated;
}

function setStatus(text) {
  statusText.textContent = text;
}

function addMessage(role, text, options = {}) {
  const item = document.createElement("div");
  item.className = `message ${role}`;
  if (options.markdown) {
    item.appendChild(renderMarkdown(text));
  } else {
    item.textContent = text;
  }
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function renderMarkdown(text) {
  const root = document.createElement("div");
  root.className = "markdown-content";
  const lines = String(text || "").split(/\r?\n/);
  let paragraph = [];
  let list = null;
  let orderedList = null;

  function flushParagraph() {
    if (paragraph.length === 0) {
      return;
    }
    const p = document.createElement("p");
    p.innerHTML = renderInlineMarkdown(paragraph.join(" "));
    root.appendChild(p);
    paragraph = [];
  }

  function closeLists() {
    list = null;
    orderedList = null;
  }

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      flushParagraph();
      closeLists();
      continue;
    }

    const heading = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      flushParagraph();
      closeLists();
      const level = Math.min(heading[1].length + 2, 5);
      const element = document.createElement(`h${level}`);
      element.innerHTML = renderInlineMarkdown(heading[2]);
      root.appendChild(element);
      continue;
    }

    const bullet = trimmed.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      flushParagraph();
      orderedList = null;
      if (!list) {
        list = document.createElement("ul");
        root.appendChild(list);
      }
      const li = document.createElement("li");
      li.innerHTML = renderInlineMarkdown(bullet[1]);
      list.appendChild(li);
      continue;
    }

    const numbered = trimmed.match(/^\d+[.)]\s+(.+)$/);
    if (numbered) {
      flushParagraph();
      list = null;
      if (!orderedList) {
        orderedList = document.createElement("ol");
        root.appendChild(orderedList);
      }
      const li = document.createElement("li");
      li.innerHTML = renderInlineMarkdown(numbered[1]);
      orderedList.appendChild(li);
      continue;
    }

    closeLists();
    paragraph.push(trimmed);
  }

  flushParagraph();
  return root;
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderSources(items) {
  sources.replaceChildren();
  if (!items || items.length === 0) {
    sourceMeta.textContent = "No sources returned";
    return;
  }

  sourceMeta.textContent = `${items.length} retrieved chunks`;
  for (const item of items) {
    const root = document.createElement("article");
    root.className = "source-item";

    const title = document.createElement("div");
    title.className = "source-title";

    const sourceName = document.createElement("span");
    sourceName.textContent = `[${item.id}] ${item.source}`;

    const score = document.createElement("span");
    score.className = "source-score";
    score.textContent = `rel ${Number(item.relevance || 0).toFixed(2)}`;

    const heading = document.createElement("div");
    heading.className = "source-heading";
    heading.textContent = item.heading || "Document";

    const retrievedFor = document.createElement("div");
    retrievedFor.className = "source-retrieved-for";
    retrievedFor.textContent = item.retrieved_for ? `For: ${item.retrieved_for}` : "";

    const preview = document.createElement("p");
    preview.className = "source-preview";
    preview.textContent = item.preview || "";

    title.append(sourceName, score);
    root.append(title, heading);
    if (item.retrieved_for) {
      root.append(retrievedFor);
    }
    root.append(preview);
    sources.appendChild(root);
  }
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  input.value = "";
  input.style.height = "";
  sendButton.disabled = true;
  setStatus("Thinking");

  try {
    const data = await postJson("/api/chat", { message, session_id: sessionId });
    addMessage("assistant", data.answer || "No answer returned.", { markdown: true });
    renderSources(data.sources || []);
    const score = data.quality_score && data.quality_score.overall_score ? data.quality_score.overall_score : "-";
    const classification = data.classification || "unknown";
    const tokens = data.trace && data.trace.token_usage ? data.trace.token_usage.all_total_tokens : "-";
    setStatus(`${classification} / score ${score} / tokens ${tokens} / ${sessionId.slice(0, 8)}`);
  } catch (error) {
    addMessage("system", error.message);
    setStatus("Error");
  } finally {
    sendButton.disabled = false;
    input.focus();
  }
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});
