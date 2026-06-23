/**
 * Webcam capture for the live-follow ("your attempt") source.
 *
 * Attaches a getUserMedia MediaStream to the test <video> element and resolves
 * once the stream is producing frames. The rest of the pipeline (MoveNet →
 * normalize → similarity → score) consumes the <video> exactly the same way
 * whether its frames come from an uploaded file or the live camera.
 */

export interface WebcamHandle {
  readonly stream: MediaStream;
  /** Stop all tracks and detach the stream from the video element. */
  stop(): void;
}

/**
 * Start the webcam and bind it to `video`. Throws on unsupported browsers or
 * when the user denies camera permission (callers should surface this and fall
 * back to file input).
 */
export async function startWebcam(
  video: HTMLVideoElement,
): Promise<WebcamHandle> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Webcam is not supported in this browser.");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user",
    },
    audio: false,
  });

  // Detach any previous file source so the element switches cleanly to live.
  const prevUrl = video.dataset.objectUrl;
  if (prevUrl) {
    URL.revokeObjectURL(prevUrl);
    delete video.dataset.objectUrl;
  }
  video.removeAttribute("src");
  video.srcObject = stream;

  await new Promise<void>((resolve, reject) => {
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error("Failed to start the webcam stream."));
    };
    const cleanup = () => {
      video.removeEventListener("loadeddata", onLoaded);
      video.removeEventListener("error", onError);
    };
    video.addEventListener("loadeddata", onLoaded);
    video.addEventListener("error", onError);
  });

  // A muted live stream autoplays; ignore the rare rejection.
  await video.play().catch(() => undefined);

  return {
    stream,
    stop() {
      for (const track of stream.getTracks()) track.stop();
      if (video.srcObject === stream) video.srcObject = null;
    },
  };
}
