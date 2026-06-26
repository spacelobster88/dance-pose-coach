// Post-run improvement report panel (issue #9).
//
// Renders the structured ImprovementReport from report.ts into two parts:
//   1. "Biggest opportunities" — the top-N worst limb×segment pairs across the
//      whole run, each with a time range, a numeric error, and a concrete fix.
//   2. A per-segment table — one row per time window showing its worst limb and
//      correction. Clicking a row seeks both videos to that moment for review.
//
// Pure DOM rendering; the seek action is delegated to a callback so this module
// stays independent of the player.

import type { ImprovementReport, LimbSegmentScore } from "../pose/report";

export interface ReportPanelCallbacks {
  /** Seek both videos to this reference time (seconds) for side-by-side review. */
  onSeek?: (startSec: number) => void;
}

export class ReportPanel {
  private readonly root: HTMLElement;

  constructor(container: HTMLElement) {
    this.root = container;
  }

  /** Empty the panel and hide its hosting section. */
  clear(): void {
    this.root.replaceChildren();
    const section = this.root.closest("section");
    if (section) section.hidden = true;
  }

  /** Render a report; reveals the hosting section. */
  render(report: ImprovementReport, cb: ReportPanelCallbacks = {}): void {
    this.root.replaceChildren();
    const section = this.root.closest("section");
    if (section) section.hidden = false;

    if (report.frameCount === 0 || report.topOpportunities.length === 0) {
      const empty = document.createElement("p");
      empty.className = "rp-empty";
      empty.textContent =
        report.frameCount === 0
          ? "No comparison data was recorded — play a routine to the end to get a report."
          : "Nice — no limb drifted far enough from the reference to flag. Try a higher strictness for finer feedback.";
      this.root.appendChild(empty);
      return;
    }

    this.root.appendChild(this.renderOpportunities(report, cb));
    this.root.appendChild(this.renderSegments(report, cb));
  }

  private renderOpportunities(
    report: ImprovementReport,
    cb: ReportPanelCallbacks,
  ): HTMLElement {
    const block = document.createElement("div");
    block.className = "rp-block";

    const h = document.createElement("h4");
    h.className = "rp-subhead";
    h.textContent = "Biggest opportunities";
    block.appendChild(h);

    const list = document.createElement("ol");
    list.className = "rp-opps";
    for (const o of report.topOpportunities) {
      const li = document.createElement("li");
      li.className = "rp-opp";
      li.tabIndex = 0;
      li.title = "Click to replay this moment side by side";

      const top = document.createElement("div");
      top.className = "rp-opp-top";
      const when = document.createElement("span");
      when.className = "rp-time";
      when.textContent = o.label;
      const limb = document.createElement("span");
      limb.className = "rp-limb";
      limb.textContent = o.limbLabel;
      const err = document.createElement("span");
      err.className = "rp-err";
      err.textContent = `${o.meanErrorDeg.toFixed(0)}°`;
      err.title = "Mean bone-direction error over this segment";
      top.append(when, limb, err);

      const fix = document.createElement("div");
      fix.className = "rp-fix";
      fix.textContent = o.correction;

      li.append(top, fix);
      this.wireSeek(li, o.startSec, cb);
      list.appendChild(li);
    }
    block.appendChild(list);
    return block;
  }

  private renderSegments(
    report: ImprovementReport,
    cb: ReportPanelCallbacks,
  ): HTMLElement {
    const block = document.createElement("div");
    block.className = "rp-block";

    const h = document.createElement("h4");
    h.className = "rp-subhead";
    h.textContent = "Segment-by-segment";
    block.appendChild(h);

    const table = document.createElement("div");
    table.className = "rp-table";

    const head = document.createElement("div");
    head.className = "rp-trow rp-thead";
    for (const [txt, cls] of [
      ["When", "rp-c-when"],
      ["Worst limb", "rp-c-limb"],
      ["Error", "rp-c-err"],
      ["How to fix", "rp-c-fix"],
    ] as const) {
      const c = document.createElement("span");
      c.className = cls;
      c.textContent = txt;
      head.appendChild(c);
    }
    table.appendChild(head);

    for (const seg of report.segments) {
      const worst: LimbSegmentScore | null = seg.worst;
      const row = document.createElement("div");
      row.className = "rp-trow";
      row.tabIndex = 0;
      row.title = "Click to replay this moment side by side";

      const when = document.createElement("span");
      when.className = "rp-c-when rp-time";
      when.textContent = seg.label;

      const limb = document.createElement("span");
      limb.className = "rp-c-limb";
      limb.textContent = worst ? worst.limbLabel : "—";

      const err = document.createElement("span");
      err.className = "rp-c-err";
      err.textContent = worst ? `${worst.meanErrorDeg.toFixed(0)}°` : "—";

      const fix = document.createElement("span");
      fix.className = "rp-c-fix";
      fix.textContent = worst ? worst.correction : "On the reference — nice.";

      row.append(when, limb, err, fix);
      this.wireSeek(row, seg.startSec, cb);
      table.appendChild(row);
    }

    block.appendChild(table);
    return block;
  }

  private wireSeek(
    el: HTMLElement,
    startSec: number,
    cb: ReportPanelCallbacks,
  ): void {
    if (!cb.onSeek) return;
    el.classList.add("rp-seekable");
    const go = () => cb.onSeek?.(startSec);
    el.addEventListener("click", go);
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        go();
      }
    });
  }
}
