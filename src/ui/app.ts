import { PoseDetector, type ModelVariant } from "../pose/detector";
import { normalizePose } from "../pose/normalize";
import { poseSimilarity, ScoreTracker } from "../pose/similarity";
import {
  drawSkeleton,
  clearCanvas,
  syncCanvasToVideo,
} from "../render/skeleton";
import { DualPlayer, loadVideoFromFile } from "../video/dualPlayer";
import { startWebcam, type WebcamHandle } from "../video/webcam";

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
  playBtn: HTMLButtonElement;
  restartBtn: HTMLButtonElement;
  scoreNow: HTMLElement;
  scoreAvg: HTMLElement;
  scoreFill: HTMLElement;
  status: HTMLElement;
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
    playBtn: byId("play-btn"),
    restartBtn: byId("restart-btn"),
    scoreNow: byId("score-now"),
    scoreAvg: byId("score-avg"),
    scoreFill: byId("score-fill"),
    status: byId("status"),
  };

  const detector = new PoseDetector(
    (dom.modelSelect.value as ModelVariant) || "lightning",
  );
  const tracker = new ScoreTracker(0.3);

  let refReady = false;
  let testReady = false;
  let modelReady = false;
  let frameBusy = false;
  let webcam: WebcamHandle | null = null;

  const setStatus = (msg: string) => {
    dom.status.textContent = msg;
  };

  const updateControls = () => {
    const canPlay = refReady && testReady && modelReady;
    dom.playBtn.disabled = !canPlay;
    dom.restartBtn.disabled = !canPlay;
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

      if (refPose) drawSkeleton(dom.refCanvas, refPose);
      else clearCanvas(dom.refCanvas);
      if (testPose) drawSkeleton(dom.testCanvas, testPose);
      else clearCanvas(dom.testCanvas);

      if (refPose && testPose) {
        const refNorm = normalizePose(refPose);
        const testNorm = normalizePose(testPose);
        if (refNorm && testNorm) {
          const result = poseSimilarity(refNorm, testNorm);
          if (result) {
            tracker.push(result.score);
            setScoreUi(tracker.smoothed, tracker.average);
          }
        }
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
    updateControls();
    setStatus(`Loading reference video “${file.name}”…`);
    try {
      await loadVideoFromFile(dom.refVideo, file);
      refReady = true;
      tracker.reset();
      setScoreUi(null, null);
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
    updateControls();
    setStatus(`Loading test video “${file.name}”…`);
    try {
      await loadVideoFromFile(dom.testVideo, file);
      testReady = true;
      tracker.reset();
      setScoreUi(null, null);
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
    testReady = false;
    tracker.reset();
    setScoreUi(null, null);
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

  // ---- Playback controls ----
  dom.playBtn.addEventListener("click", async () => {
    if (player.playing) {
      player.pause();
    } else {
      // Restart scoring from a clean slate if both clips are at the start.
      if (dom.refVideo.currentTime === 0) tracker.reset();
      setStatus("Playing — comparing poses…");
      await player.play();
    }
  });

  dom.restartBtn.addEventListener("click", () => {
    player.pause();
    tracker.reset();
    setScoreUi(null, null);
    player.restart();
    setStatus("Reset to start.");
  });

  updateControls();
  setScoreUi(null, null);
}
