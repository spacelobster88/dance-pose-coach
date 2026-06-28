// Tiny, dependency-free Markdown → HTML renderer.
//
// The project ships no runtime UI dependencies, and coaching output is
// constrained (headings, bold/italic, inline code, ordered/unordered lists,
// paragraphs). We escape first, then apply a small set of block + inline rules.
// This is not a general Markdown engine — it covers exactly what the coaching
// prompt asks providers to emit.

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

/** Apply inline formatting (bold, italic, code) to already-escaped text. */
function inline(s: string): string {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>")
    .replace(/_([^_]+)_/g, "<em>$1</em>");
}

/** Render a constrained subset of Markdown to safe HTML. */
export function renderMarkdown(md: string): string {
  const lines = escapeHtml(md).split(/\r?\n/);
  const html: string[] = [];
  let listType: "ul" | "ol" | null = null;
  let para: string[] = [];

  const flushPara = () => {
    if (para.length) {
      html.push(`<p>${inline(para.join(" "))}</p>`);
      para = [];
    }
  };
  const closeList = () => {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (line.trim() === "") {
      flushPara();
      closeList();
      continue;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushPara();
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }

    const ol = /^\s*\d+\.\s+(.*)$/.exec(line);
    if (ol) {
      flushPara();
      if (listType !== "ol") {
        closeList();
        listType = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${inline(ol[1])}</li>`);
      continue;
    }

    const ul = /^\s*[-*]\s+(.*)$/.exec(line);
    if (ul) {
      flushPara();
      if (listType !== "ul") {
        closeList();
        listType = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }

    // Plain text: accumulate into the current paragraph (lists end here).
    closeList();
    para.push(line.trim());
  }

  flushPara();
  closeList();
  return html.join("\n");
}
