import { PoseDetector, type ModelVariant } from "../pose/detector";
import { normalizePose } from "../pose/normalize";
import { poseSimilarity, ScoreTracker } from "../pose/similarity";
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
import { drawComposite, sizeComposite } from "../render/composite";
import {
  ComparisonRecorder,
  downloadBlob,
  recordingSupported,
} from "../video/recorder";
import { DualPlayer, loadVideoFromFile } from "../video/dualPlayer";
import { startWebcam, type WebcamHandle } from "../video/webcam";
import { sampleVideoPoses, buildWarp } from "../video/sampler";
import { dtwAlign } from "../pose/dtw";
import { StreamingAligner } from "../pose/streamDtw";

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
  scoreNow: HTMLElement;
  scoreAvg: HTMLElement;
  scoreFill: HTMLElement;
  scoreHistory: HTMLCanvasElement;
  lagStat: HTMLElement;
  scoreLag: HTMLElement;
  status: HTMLElement;
  breakdownRows: HTMLElement;
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
    scoreNow: byId("score-now"),
    scoreAvg: byId("score-avg"),
    scoreFill: byId("score-fill"),
    scoreHistory: byId("score-history"),
    lagStat: byId("lag-stat"),
    scoreLag: byId("score-lag"),
    status: byId("status"),
    breakdownRows: byId("breakdown-rows"),
  };

  const detector = new PoseDetector(
    (dom.modelSelect.value as ModelVariant) || "lightning",
  );
  const tracker = new ScoreTracker(0.3);
  const limbTracker = new LimbDivergenceTracker(0.3);
  const scoreGraph = new ScoreGraph(dom.scoreHistory);
  const streamAligner = new StreamingAligner();
  // Offscreen canvas the recorder captures; composited each frame while active.
  const exportCanvas = document.createElement("canvas");
  const recorder = new ComparisonRecorder();

  /** Clear the running score, its history graph, and the numeric display. */
  const resetScore = () => {
    tracker.reset();
    scoreGraph.reset();
    streamAligner.reset();
    setScoreUi(null, null);
    setLagUi(null);
  };

  let refReady = false;
  let testReady = false;
  let modelReady = false;
  let frameBusy = false;
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
      const blob = await recorder.stop();
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      downloadBlob(blob, `dance-pose-coach-${stamp}.webm`);
      setStatus(`Saved comparison clip (${(blob.size / 1e6).toFixed(1)} MB).`);
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
      // If we were recording, finalize and download the clip now.
      void finishRecording();
    },
  });

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
          const result = poseSimilarity(refForScore, testNorm);
          if (result) {
            tracker.push(result.score);
            setScoreUi(tracker.smoothed, tracker.average);
            if (tracker.smoothed !== null) {
              scoreGraph.push(result.score, tracker.smoothed);
            }
          }
          limbTracker.push(perJointDivergence(refForScore, testNorm));
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

  // ---- Export / record button ----
  dom.recordBtn.addEventListener("click", async () => {
    if (recorder.active) {
      await finishRecording();
      return;
    }
    try {
      sizeComposite(exportCanvas); // fix dimensions before capturing the stream
      recorder.start(exportCanvas);
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
}
