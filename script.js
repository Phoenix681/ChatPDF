const API_URL = "http://127.0.0.1:8000";

async function handleUpload() {
  const fileInput = document.getElementById("pdfFile");
  const status = document.getElementById("uploadStatus");
  const btn = document.getElementById("uploadBtn");

  if (!fileInput.files[0]) return alert("Please select a PDF file first.");

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.innerText = "Processing pipeline active...";
  status.style.color = "#6366f1";
  btn.disabled = true;

  try {
    const response = await fetch(`${API_URL}/upload-pdf`, {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (result.status === "success") {
      status.style.color = "#059669";
      status.innerHTML = `${result.message} <span class="collection-tag">📚 Collection: ${result.collection}</span>`;
    } else {
      status.style.color = "#dc2626";
      status.innerText = result.message;
    }
  } catch (error) {
    status.style.color = "#dc2626";
    status.innerText =
      "Pipeline execution failed. Is the FastAPI server running?";
  } finally {
    btn.disabled = false;
  }
}

async function handleChat() {
  const query = document.getElementById("chatQuery").value.trim();
  const results = document.getElementById("chat-results");
  const status = document.getElementById("chatStatus");

  if (!query) return;

  const formData = new FormData();
  formData.append("question", query);

  status.style.color = "#6366f1";
  status.innerText = "Retrieving context...";

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (result.status === "success") {
      const pagesHtml =
        result.pages && result.pages.length > 0
          ? `<div class="page-ref-box">
             <span class="page-ref-label">&#128196; Referenced Pages</span>
             ${result.pages.map((p) => `<span class="page-badge">Page ${p}</span>`).join("")}
           </div>`
          : "";

      results.innerHTML += `
        <p><strong>Q:</strong> ${query}</p>
        <div><strong>A:</strong> ${marked.parse(result.answer)}</div>
        ${pagesHtml}
        <hr>`;
      results.scrollTop = results.scrollHeight;
      document.getElementById("chatQuery").value = "";
      status.innerText = "";
    } else {
      status.style.color = "#dc2626";
      status.innerText = result.answer;
      return;
    }
  } catch (error) {
    status.style.color = "#dc2626";
    status.innerText = "Query retrieval failed. Is the FastAPI server running?";
  }
}

// Allow pressing Enter to submit chat query
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("chatQuery").addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleChat();
  });
});