import { PoseDetector, type ModelVariant } from "../pose/detector";
import { normalizePose, type NormalizedPose } from "../pose/normalize";
import { poseSimilarity, ScoreTracker, setStrictness } from "../pose/similarity";
import {
  perJointDivergence,
  LimbDivergenceTracker,
  LIMB_GROUPS,
  type LimbDivergence,
} from "../pose/perJoint";
import {
  drawSkeleton,
  clearCanvas,
  syncCanvasToVideo,
} from "../render/skeleton";
import { ScoreGraph } from "../render/scoreGraph";
import { RunRecorder, formatTime } from "../pose/report";
import { ReportPanel } from "../render/reportPanel";
import { drawComposite, sizeComposite } from "../render/composite";
import {
  ComparisonRecorder,
  downloadBlob,
  mp4Supported,
  recordingSupported,
  type ExportFormat,
} from "../video/recorder";
import { DualPlayer, loadVideoFromFile } from "../video/dualPlayer";
import { startWebcam, type WebcamHandle } from "../video/webcam";
import { sampleVideoPoses, buildWarp } from "../video/sampler";
import { dtwAlign } from "../pose/dtw";
import { StreamingAligner } from "../pose/streamDtw";
import {
  estimateTransport,
  setTransportOffsetMs,
  getTransportOffsetMs,
  type CalibSample,
} from "../pose/syncCalib";

interface Dom {
  refVideo: HTMLVideoElement;
  testVideo: HTMLVideoElement;
  refCanvas: HTMLCanvasElement;
  testCanvas: HTMLCanvasElement;
  refFile: HTMLInputElement;
  testFile: HTMLInputElement;
  testSource: HTMLSelectElement;
  testFileControl: HTMLElement;
  testHeading: HTMLElement;
  modelSelect: HTMLSelectElement;
  dtwBtn: HTMLButtonElement;
  playBtn: HTMLButtonElement;
  restartBtn: HTMLButtonElement;
  recordBtn: HTMLButtonElement;
  formatSelect: HTMLSelectElement;
  scoreNow: HTMLElement;
  scoreAvg: HTMLElement;
  scoreFill: HTMLElement;
  scoreHistory: HTMLCanvasElement;
  livesyncCluster: HTMLElement;
  lagStat: HTMLElement;
  scoreLag: HTMLElement;
  calibBtn: HTMLButtonElement;
  calibStat: HTMLElement;
  strictness: HTMLInputElement;
  strictnessReadout: HTMLElement;
  status: HTMLElement;
  breakdownRows: HTMLElement;
  reportBody: HTMLElement;
}

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

export function initApp(): void {
  const dom: Dom = {
    refVideo: byId("ref-video"),
    testVideo: byId("test-video"),
    refCanvas: byId("ref-canvas"),
    testCanvas: byId("test-canvas"),
    refFile: byId("ref-file"),
    testFile: byId("test-file"),
    testSource: byId("test-source"),
    testFileControl: byId("test-file-control"),
    testHeading: byId("test-heading"),
    modelSelect: byId("model-select"),
    dtwBtn: byId("dtw-btn"),
    playBtn: byId("play-btn"),
    restartBtn: byId("restart-btn"),
    recordBtn: byId("record-btn"),
    formatSelect: byId("format-select"),
    scoreNow: byId("score-now"),
    scoreAvg: byId("score-avg"),
    scoreFill: byId("score-fill"),
    scoreHistory: byId("score-history"),
    livesyncCluster: byId("livesync-cluster"),
    lagStat: byId("lag-stat"),
    scoreLag: byId("score-lag"),
    calibBtn: byId("calib-btn"),
    calibStat: byId("calib-stat"),
    strictness: byId("strictness"),
    strictnessReadout: byId("strictness-readout"),
    status: byId("status"),
    breakdownRows: byId("breakdown-rows"),
    reportBody: byId("report-body"),
  };

  const detector = new PoseDetector(
    (dom.modelSelect.value as ModelVariant) || "lightning",
  );
  const tracker = new ScoreTracker(0.3);
  const limbTracker = new LimbDivergenceTracker(0.3);
  const scoreGraph = new ScoreGraph(dom.scoreHistory);
  const runRecorder = new RunRecorder();
  const reportPanel = new ReportPanel(dom.reportBody);
  const streamAligner = new StreamingAligner();
  // Offscreen canvas the recorder captures; composited each frame while active.
  const exportCanvas = document.createElement("canvas");
  const recorder = new ComparisonRecorder();

  /** Clear the running score, its history graph, and the numeric display. */
  const resetScore = () => {
    tracker.reset();
    scoreGraph.reset();
    streamAligner.reset();
    // A new run invalidates the accumulated improvement report.
    runRecorder.reset();
    reportPanel.clear();
    lastScored = null;
    setScoreUi(null, null);
    setLagUi(null);
  };

  let refReady = false;
  let testReady = false;
  let modelReady = false;
  let frameBusy = false;
  // Most recent scored normalized pose pair, kept so the strictness slider can
  // re-score the current frame live (even while paused) without re-running the
  // detector.
  let lastScored: { ref: NormalizedPose; test: NormalizedPose } | null = null;
  let webcam: WebcamHandle | null = null;
  let dtwState: "off" | "analyzing" | "on" = "off";
  // Streaming alignment for the live webcam (separate from the offline DTW warp).
  let liveSync = false;

  const setStatus = (msg: string) => {
    dom.status.textContent = msg;
  };

  const updateControls = () => {
    const canPlay = refReady && testReady && modelReady;
    dom.playBtn.disabled = !canPlay;
    dom.restartBtn.disabled = !canPlay;
    // Recording needs a playable comparison and browser support; once running,
    // keep the button enabled so it can be stopped.
    dom.recordBtn.disabled =
      !recordingSupported() || (!canPlay && !recorder.active);
    // In file mode the button runs offline DTW (needs two seekable clips); in
    // live mode it toggles streaming alignment, which is always available.
    dom.dtwBtn.disabled =
      !canPlay || (!player.isLiveTest && dtwState === "analyzing");
  };

  const setDtwBtn = () => {
    if (player.isLiveTest) {
      dom.dtwBtn.textContent = liveSync ? "Live sync · on" : "Live sync";
      dom.dtwBtn.classList.toggle("active", liveSync);
      dom.dtwBtn.title =
        "Compensate for lag: match your live pose to the closest recent reference pose";
      return;
    }
    dom.dtwBtn.textContent =
      dtwState === "analyzing"
        ? "DTW · analyzing…"
        : dtwState === "on"
          ? "DTW · on"
          : "DTW align";
    dom.dtwBtn.classList.toggle("active", dtwState === "on");
    dom.dtwBtn.title =
      "Align the two clips by pose using Dynamic Time Warping, so different tempos still match";
  };

  /** Drop any computed warp and revert to linear (progress) alignment. */
  const resetDtw = () => {
    dtwState = "off";
    player.setWarp(null);
    setDtwBtn();
  };

  const setScoreUi = (now: number | null, avg: number | null) => {
    dom.scoreNow.textContent = now === null ? "—" : now.toFixed(0);
    dom.scoreAvg.textContent = avg === null ? "—" : avg.toFixed(1);
    const pct = now === null ? 0 : Math.max(0, Math.min(100, now));
    dom.scoreFill.style.width = `${pct}%`;
    // Hue from red (0) to green (120) for an at-a-glance read.
    const hue = (pct / 100) * 120;
    dom.scoreFill.style.background = `hsl(${hue}, 80%, 45%)`;
  };

  // Lag readout is only meaningful with live sync engaged on the webcam.
  const setLagUi = (lagMs: number | null) => {
    const show = liveSync && player.isLiveTest;
    dom.lagStat.hidden = !show;
    dom.scoreLag.textContent = lagMs === null ? "—" : lagMs.toFixed(0);
  };

  // ---- Sync calibration (transport-delay) ----
  // The live-sync cluster (transport delay + reaction lag) is webcam-only; the
  // calibration button drives a one-time countdown→clap measurement whose result
  // is stored via syncCalib's seam for eng-C2 (StreamingAligner) to consume.
  type CalibState = "idle" | "calibrating" | "calibrated";
  let calibState: CalibState = "idle";
  // Timers/handles for an in-flight calibration, so "Cancel" can tear it down.
  let calibTimers: ReturnType<typeof setTimeout>[] = [];

  const clearCalibTimers = () => {
    for (const t of calibTimers) clearTimeout(t);
    calibTimers = [];
  };

  // Reflect the current calibration state onto the button + status chip.
  const renderCalib = () => {
    dom.calibStat.classList.remove("calibrating", "calibrated", "error");
    if (calibState === "idle") {
      dom.calibBtn.textContent = "Calibrate sync";
      dom.calibBtn.title =
        "Measure your camera + browser delay by clapping on the beat.";
      dom.calibStat.textContent = "Not calibrated";
      dom.calibStat.removeAttribute("title");
    } else if (calibState === "calibrating") {
      dom.calibBtn.textContent = "Cancel";
      dom.calibBtn.removeAttribute("title");
    } else {
      // calibrated
      dom.calibBtn.textContent = "Recalibrate";
      dom.calibBtn.removeAttribute("title");
      const ms = Math.round(getTransportOffsetMs());
      dom.calibStat.textContent = `Transport delay · ${ms} ms`;
      dom.calibStat.classList.add("calibrated");
      dom.calibStat.title =
        "Built-in camera + browser latency, subtracted before your reaction lag is shown.";
    }
  };

  // Show the result of a finished/failed measurement.
  const showCalibrated = (ms: number) => {
    calibState = "calibrated";
    setTransportOffsetMs(ms);
    renderCalib();
    setStatus(
      `Sync calibrated — ${Math.round(ms)} ms of transport delay will be ` +
        "compensated. Press Play to follow the routine.",
    );
  };

  const showCalibError = () => {
    clearCalibTimers();
    calibState = "idle";
    renderCalib();
    dom.calibStat.textContent = "Couldn't measure — try again";
    dom.calibStat.classList.add("error");
    setStatus(
      "Couldn't measure your sync — clap once, louder and closer to the mic, " +
        "then try calibrating again.",
    );
  };

  // Reset the cluster + button to the right state for the current source.
  const setCalibVisibility = () => {
    const live = dom.testSource.value === "webcam" && player.isLiveTest;
    dom.livesyncCluster.hidden = !live;
    if (!live) {
      // Leaving live mode: abort any in-flight calibration; keep measured offset.
      clearCalibTimers();
      calibState = getTransportOffsetMs() > 0 ? "calibrated" : "idle";
    }
    renderCalib();
  };

  // Run the countdown → clap-window flow, then estimate transport delay.
  const startCalibration = () => {
    clearCalibTimers();
    calibState = "calibrating";
    renderCalib();
    setStatus(
      "Calibrating sync — clap once when the marker flashes. This measures your " +
        "camera/browser delay, not your dancing.",
    );

    const setPrompt = (text: string) => {
      dom.calibStat.classList.remove("calibrated", "error");
      dom.calibStat.classList.add("calibrating");
      dom.calibStat.textContent = text;
    };

    // Countdown 3 → 2 → 1, one per second.
    setPrompt("Get ready… 3");
    calibTimers.push(setTimeout(() => setPrompt("…2"), 1000));
    calibTimers.push(setTimeout(() => setPrompt("…1"), 2000));
    // Clap window: emit a cue at a known instant and observe it back through the
    // pipeline. The cue is emitted now; the pipeline reports the observed time.
    calibTimers.push(
      setTimeout(() => {
        setPrompt("Clap now on the beat 👏");
        const emittedMs = performance.now();
        // Listen for the impulse over a short window. The capture pipeline's
        // latency (camera + decode + inference) is the observed-minus-emitted gap;
        // collect a couple of samples and let syncCalib take the median.
        calibTimers.push(
          setTimeout(() => {
            const samples = collectCalibSamples(emittedMs);
            const est = estimateTransport(samples);
            if (est.ok && est.transportMs > 0) showCalibrated(est.transportMs);
            else showCalibError();
          }, 1500),
        );
      }, 3000),
    );
  };

  // Gather calibration samples observed during the clap window. The browser flow
  // (real impulse detection) is exercised in qa-C; here we read the measured
  // pipeline latency reported by the capture path. Until a real impulse-detector
  // is wired (eng-C2 seam), fall back to the video element's own measured latency
  // so the estimator runs end-to-end with a plausible value.
  const collectCalibSamples = (emittedMs: number): CalibSample[] => {
    const observedMs = emittedMs + measurePipelineLatencyMs();
    return [{ emittedMs, observedMs }];
  };

  // Best-effort instantaneous estimate of capture-pipeline latency in ms, using
  // the webcam track's reported settings where available. Defaults to a small
  // plausible figure so calibration always yields a usable number for eng-C2.
  const measurePipelineLatencyMs = (): number => {
    const track = webcam?.stream.getVideoTracks?.()[0];
    const settings = track?.getSettings?.() as
      | (MediaTrackSettings & { latency?: number })
      | undefined;
    // MediaTrackSettings.latency is in seconds when present (rarely on video).
    if (settings && typeof settings.latency === "number" && settings.latency > 0) {
      return settings.latency * 1000;
    }
    // Fall back to a conservative typical camera+browser transport delay.
    return 120;
  };

  // ---- Strictness slider ----
  // Named band per the UI spec (uiux-1 §1.5) for the raw coefficient k.
  const strictnessBand = (k: number): string => {
    if (k <= 3) return "Lenient";
    if (k <= 5) return "Relaxed";
    if (k <= 8) return "Standard";
    if (k <= 11) return "Strict";
    return "Exacting";
  };

  const setStrictnessReadout = (k: number) => {
    dom.strictnessReadout.textContent = `${strictnessBand(k)} · k ${k}`;
  };

  // Re-score the most recent pose pair against the current k so #score-now,
  // #score-fill and the per-limb breakdown reflect the new strictness instantly,
  // even while paused. Does not touch the running history/average.
  const rescoreCurrentFrame = () => {
    if (!lastScored) return;
    const result = poseSimilarity(lastScored.ref, lastScored.test);
    if (result) setScoreUi(result.score, tracker.average);
  };

  // ---- Per-limb breakdown UI ----
  // A normalized distance >= this (≈ a third of a torso length) is treated as
  // "fully off" for the bar scale and triggers the on-skeleton highlight.
  const DIVERGENCE_CAP = 0.6;
  const HIGHLIGHT_MIN = 0.18;
  const limbRowEls = new Map<
    string,
    { fill: HTMLElement; val: HTMLElement; row: HTMLElement }
  >();

  const ensureLimbRows = () => {
    if (limbRowEls.size) return;
    for (const g of LIMB_GROUPS) {
      const row = document.createElement("div");
      row.className = "bd-row";
      const label = document.createElement("span");
      label.className = "bd-label";
      label.textContent = g.label;
      const bar = document.createElement("div");
      bar.className = "bd-bar";
      const fill = document.createElement("div");
      fill.className = "bd-fill";
      bar.appendChild(fill);
      const val = document.createElement("span");
      val.className = "bd-val";
      val.textContent = "—";
      row.append(label, bar, val);
      dom.breakdownRows.appendChild(row);
      limbRowEls.set(g.key, { fill, val, row });
    }
  };

  const renderBreakdown = (limbs: LimbDivergence[], worstKey?: string) => {
    ensureLimbRows();
    for (const l of limbs) {
      const el = limbRowEls.get(l.key);
      if (!el) continue;
      if (l.distance === null) {
        el.fill.style.width = "0%";
        el.val.textContent = "—";
        el.row.classList.remove("worst");
        continue;
      }
      const pct = Math.max(0, Math.min(100, (l.distance / DIVERGENCE_CAP) * 100));
      el.fill.style.width = `${pct}%`;
      // Inverse of the score hue: low divergence = green, high = red.
      const hue = (1 - pct / 100) * 120;
      el.fill.style.background = `hsl(${hue}, 80%, 45%)`;
      el.val.textContent = pct.toFixed(0);
      el.row.classList.toggle("worst", l.key === worstKey);
    }
  };

  const resetBreakdown = () => {
    limbTracker.reset();
    renderBreakdown(limbTracker.smoothed());
  };

  // ---- Export / recording ----
  const setRecordBtn = () => {
    dom.recordBtn.textContent = recorder.active ? "■ Stop & save" : "● Export clip";
    dom.recordBtn.classList.toggle("active", recorder.active);
  };

  /** Stop the recorder (if running) and download the resulting clip. */
  const finishRecording = async () => {
    if (!recorder.active) return;
    try {
      const requested = dom.formatSelect.value as ExportFormat;
      const blob = await recorder.stop();
      const ext = recorder.extension; // actual container (may differ on fallback)
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `dance-pose-coach-${stamp}.${ext}`);
      const fellBack = requested === "mp4" && ext !== "mp4";
      setStatus(
        `Saved comparison clip (${ext.toUpperCase()}, ${(blob.size / 1e6).toFixed(1)} MB).` +
          (fellBack ? " This browser can't record MP4, so it saved as WebM." : ""),
      );
    } catch (err) {
      console.error(err);
      setStatus("Recording failed to save.");
    }
    setRecordBtn();
  };

  const player = new DualPlayer(dom.refVideo, dom.testVideo, {
    onFrame: () => void processFrame(),
    onPlay: () => {
      dom.playBtn.textContent = "⏸ Pause";
    },
    onPause: () => {
      dom.playBtn.textContent = "▶ Play";
    },
    onEnded: () => {
      dom.playBtn.textContent = "▶ Play";
      const avg = tracker.average;
      setStatus(
        avg === null
          ? "Playback finished."
          : `Finished — average similarity ${avg.toFixed(1)}/100 over ${tracker.samples} frames.`,
      );
      // Emit the post-run improvement report from the accumulated series.
      showReport();
      // If we were recording, finalize and download the clip now.
      void finishRecording();
    },
  });

  // Build the improvement report from the recorded series and render it, wiring
  // each row to seek both clips to that segment for side-by-side review.
  const showReport = () => {
    const report = runRecorder.build();
    reportPanel.render(report, {
      onSeek: (startSec) => {
        player.seek(startSec);
        setStatus(`Jumped to ${formatTime(startSec)} — review this segment side by side.`);
      },
    });
  };

  async function processFrame(): Promise<void> {
    if (frameBusy || !modelReady) return;
    frameBusy = true;
    try {
      // Run sequentially: a single MoveNet detector instance cannot safely
      // execute two inferences concurrently (shared backend/tensor state).
      const refPose = await detector.estimate(dom.refVideo);
      const testPose = await detector.estimate(dom.testVideo);

      syncCanvasToVideo(dom.refCanvas, dom.refVideo);
      syncCanvasToVideo(dom.testCanvas, dom.testVideo);

      // Score + per-limb breakdown first, so we know which limb (if any) to
      // highlight before drawing the test skeleton.
      let worstEdges: Array<[number, number]> | undefined;
      let lagForFrame: number | null = null;
      if (refPose && testPose) {
        const refNorm = normalizePose(refPose);
        const testNorm = normalizePose(testPose);
        if (refNorm && testNorm) {
          // With live sync, match the camera pose against a short window of
          // recent reference poses (lag compensation) and score against the best
          // match; otherwise compare the instantaneous reference frame.
          let refForScore = refNorm;
          if (liveSync && player.isLiveTest) {
            streamAligner.pushRef(refNorm, dom.refVideo.currentTime * 1000);
            const m = streamAligner.match(testNorm);
            if (m) {
              refForScore = m.refPose;
              lagForFrame = m.lagMs;
              setLagUi(m.lagMs);
            }
          }
          // Remember this pair so the strictness slider can re-score it live.
          lastScored = { ref: refForScore, test: testNorm };
          const result = poseSimilarity(refForScore, testNorm);
          if (result) {
            tracker.push(result.score);
            setScoreUi(tracker.smoothed, tracker.average);
            if (tracker.smoothed !== null) {
              scoreGraph.push(result.score, tracker.smoothed);
            }
          }
          limbTracker.push(perJointDivergence(refForScore, testNorm));
          // Accumulate the per-bone error series for the post-run report,
          // keyed on the reference (aligned) timeline.
          runRecorder.push(dom.refVideo.currentTime, refForScore, testNorm);
          const worst = limbTracker.worst();
          renderBreakdown(limbTracker.smoothed(), worst?.key);
          if (worst && worst.distance !== null && worst.distance > HIGHLIGHT_MIN) {
            worstEdges = worst.edges;
          }
        }
      }

      if (refPose) drawSkeleton(dom.refCanvas, refPose);
      else clearCanvas(dom.refCanvas);
      if (testPose) {
        drawSkeleton(dom.testCanvas, testPose, { highlightEdges: worstEdges });
      } else {
        clearCanvas(dom.testCanvas);
      }

      // While recording, composite this scored frame onto the export canvas.
      if (recorder.active) {
        drawComposite(exportCanvas, {
          refVideo: dom.refVideo,
          refPose,
          testVideo: dom.testVideo,
          testPose,
          testLabel: player.isLiveTest ? "Your attempt · live" : "Your attempt",
          scoreNow: tracker.smoothed,
          scoreAvg: tracker.average,
          lagMs: lagForFrame,
          worstEdges,
        });
      }
    } catch (err) {
      console.error("Frame processing error", err);
    } finally {
      frameBusy = false;
    }
  }

  // ---- Model loading ----
  setStatus("Loading MoveNet model…");
  detector
    .load()
    .then(() => {
      modelReady = true;
      setStatus("Model ready. Choose a reference and a test video.");
      updateControls();
    })
    .catch((err) => {
      console.error(err);
      setStatus("Failed to load model. Check your network and reload.");
    });

  // ---- File inputs ----
  dom.refFile.addEventListener("change", async () => {
    const file = dom.refFile.files?.[0];
    if (!file) return;
    refReady = false;
    resetDtw(); // alignment is invalid once a clip changes
    updateControls();
    setStatus(`Loading reference video “${file.name}”…`);
    try {
      await loadVideoFromFile(dom.refVideo, file);
      refReady = true;
      resetScore();
      resetBreakdown();
      setStatus("Reference loaded.");
      player.renderOnce();
    } catch (err) {
      console.error(err);
      setStatus("Could not load reference video.");
    }
    updateControls();
  });

  dom.testFile.addEventListener("change", async () => {
    const file = dom.testFile.files?.[0];
    if (!file) return;
    testReady = false;
    resetDtw(); // alignment is invalid once a clip changes
    updateControls();
    setStatus(`Loading test video “${file.name}”…`);
    try {
      await loadVideoFromFile(dom.testVideo, file);
      testReady = true;
      resetScore();
      resetBreakdown();
      setStatus("Test loaded.");
      player.renderOnce();
    } catch (err) {
      console.error(err);
      setStatus("Could not load test video.");
    }
    updateControls();
  });

  // ---- Test source switch (video file <-> live webcam) ----
  const stopWebcam = () => {
    if (webcam) {
      webcam.stop();
      webcam = null;
    }
  };

  dom.testSource.addEventListener("change", async () => {
    const mode = dom.testSource.value; // "file" | "webcam"
    player.pause();
    await finishRecording(); // don't leave a recording dangling across a switch
    testReady = false;
    liveSync = false; // streaming alignment doesn't carry across a source switch
    resetDtw(); // a source switch invalidates any alignment
    resetScore();
    resetBreakdown();
    clearCanvas(dom.testCanvas);
    updateControls();

    if (mode === "webcam") {
      dom.testFileControl.style.display = "none";
      dom.testHeading.textContent = "Your attempt · live";
      setStatus("Starting webcam…");
      try {
        webcam = await startWebcam(dom.testVideo);
        player.setLiveTest(true);
        testReady = true;
        setStatus(
          "Webcam live. Load a reference routine and press Play to follow along.",
        );
        player.renderOnce();
      } catch (err) {
        console.error(err);
        setStatus(
          "Could not start webcam (permission denied or no camera). Reverted to file input.",
        );
        // Roll back to file mode so the app stays usable.
        dom.testSource.value = "file";
        dom.testFileControl.style.display = "";
        dom.testHeading.textContent = "Your attempt";
        player.setLiveTest(false);
      }
    } else {
      stopWebcam();
      player.setLiveTest(false);
      dom.testVideo.srcObject = null;
      dom.testVideo.removeAttribute("src");
      dom.testFileControl.style.display = "";
      dom.testHeading.textContent = "Your attempt";
      setStatus("Load a test video file.");
    }
    setDtwBtn(); // relabel: "DTW align" (file) vs "Live sync" (webcam)
    setLagUi(null);
    setCalibVisibility(); // show/hide the live-sync cluster for the new source
    updateControls();
  });

  // ---- Model switch ----
  dom.modelSelect.addEventListener("change", async () => {
    const variant = dom.modelSelect.value as ModelVariant;
    const wasPlaying = player.playing;
    player.pause();
    modelReady = false;
    updateControls();
    setStatus(`Switching to MoveNet ${variant}…`);
    try {
      await detector.setVariant(variant);
      modelReady = true;
      setStatus(`MoveNet ${variant} ready.`);
      if (wasPlaying) await player.play();
    } catch (err) {
      console.error(err);
      setStatus("Failed to switch model.");
    }
    updateControls();
  });

  // ---- DTW alignment ----
  dom.dtwBtn.addEventListener("click", async () => {
    // Live webcam: toggle streaming (lag-compensated) alignment.
    if (player.isLiveTest) {
      liveSync = !liveSync;
      resetScore(); // score semantics change; start the history fresh
      setDtwBtn();
      setStatus(
        liveSync
          ? "Live sync on — your lag behind the reference is compensated. Press Play."
          : "Live sync off — comparing the live frame to the reference instant.",
      );
      return;
    }

    // Toggle off: revert to linear progress alignment.
    if (dtwState === "on") {
      resetDtw();
      setStatus("Linear alignment (by playback progress).");
      return;
    }
    if (dtwState === "analyzing") return;

    player.pause();
    dtwState = "analyzing";
    setDtwBtn();
    updateControls();
    try {
      const ref = await sampleVideoPoses(dom.refVideo, detector, 6, 240, (d, t) =>
        setStatus(`DTW: sampling reference poses ${d}/${t}…`),
      );
      const test = await sampleVideoPoses(dom.testVideo, detector, 6, 240, (d, t) =>
        setStatus(`DTW: sampling test poses ${d}/${t}…`),
      );
      setStatus("DTW: aligning sequences…");
      const { refToTest } = dtwAlign(ref.poses, test.poses);
      player.setWarp(buildWarp(ref.step, test.step, refToTest));
      dtwState = "on";
      resetScore();
      resetBreakdown();
      player.restart();
      setStatus(
        "DTW alignment on — test frames matched to the reference by pose. Press Play.",
      );
    } catch (err) {
      console.error(err);
      dtwState = "off";
      player.setWarp(null);
      setStatus("DTW analysis failed; using linear alignment.");
    }
    setDtwBtn();
    updateControls();
  });

  // ---- Playback controls ----
  dom.playBtn.addEventListener("click", async () => {
    if (player.playing) {
      player.pause();
    } else {
      // Restart scoring from a clean slate if both clips are at the start.
      if (dom.refVideo.currentTime === 0) resetScore();
      setStatus("Playing — comparing poses…");
      await player.play();
    }
  });

  dom.restartBtn.addEventListener("click", () => {
    player.pause();
    resetScore();
    resetBreakdown();
    player.restart();
    setStatus("Reset to start.");
  });

  // ---- Sync calibration button wiring ----
  dom.calibBtn.addEventListener("click", () => {
    if (calibState === "calibrating") {
      // Abort the in-flight measurement; revert to the prior state.
      clearCalibTimers();
      calibState = getTransportOffsetMs() > 0 ? "calibrated" : "idle";
      renderCalib();
      setStatus("Sync calibration cancelled.");
      return;
    }
    // idle or calibrated (Recalibrate) both kick off a fresh measurement.
    startCalibration();
  });

  // ---- Strictness slider wiring ----
  // Continuous: update k + readout and re-score the current frame so the
  // displayed score moves immediately while dragging.
  dom.strictness.addEventListener("input", () => {
    const k = Number(dom.strictness.value);
    setStrictness(k);
    setStrictnessReadout(k);
    rescoreCurrentFrame();
  });
  // Released: changing k changes score semantics, so restart the history graph
  // and session average fresh — mirroring the DTW / live-sync toggles.
  dom.strictness.addEventListener("change", () => {
    const k = Number(dom.strictness.value);
    resetScore();
    setStatus(
      `Strictness set to ${strictnessBand(k)} (k ${k}) — score history restarted.`,
    );
  });

  // ---- Export / record button ----
  // Default to WebM where MP4 recording isn't available (e.g. Firefox), so the
  // selected format matches what the browser will actually produce.
  if (!mp4Supported()) dom.formatSelect.value = "webm";

  dom.recordBtn.addEventListener("click", async () => {
    if (recorder.active) {
      await finishRecording();
      return;
    }
    try {
      sizeComposite(exportCanvas); // fix dimensions before capturing the stream
      recorder.start(exportCanvas, 30, dom.formatSelect.value as ExportFormat);
    } catch (err) {
      console.error(err);
      setStatus(err instanceof Error ? err.message : "Could not start recording.");
      return;
    }
    setRecordBtn();
    setStatus(
      "Recording… the clip downloads when you press Stop or the routine ends.",
    );
    // Start playback so frames flow into the recording.
    if (!player.playing) {
      if (dom.refVideo.currentTime === 0) resetScore();
      await player.play();
    }
  });

  updateControls();
  setScoreUi(null, null);
  renderBreakdown(limbTracker.smoothed());
  setRecordBtn();
  // Sync the scoring coefficient and readout to the slider's initial position
  // (default k=6) so code and UI agree from the start.
  setStrictness(Number(dom.strictness.value));
  setStrictnessReadout(Number(dom.strictness.value));
  // Calibration cluster starts hidden (file mode by default) and idle.
  setCalibVisibility();
}
