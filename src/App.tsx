import { useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

const OBJECTS = [
  "person",
  "bottle",
  "chair",
  "cup",
  "book",
  "laptop",
  "cell phone",
  "keyboard",
  "mouse",
  "remote",
];

type Prediction = {
  class: string;
  score: number;
  bbox: number[];
};

function speak(message: string) {
  window.speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(message);
  utterance.rate = 0.95;
  utterance.pitch = 1;
  utterance.volume = 1;

  window.speechSynthesis.speak(utterance);
}

function getDirection(bbox: number[], imageWidth: number) {
  const [x, , width] = bbox;
  const centerX = x + width / 2;

  if (centerX < imageWidth / 3) {
    return "on your left";
  }

  if (centerX > (imageWidth * 2) / 3) {
    return "on your right";
  }

  return "ahead";
}

function getDistance(bbox: number[], imageWidth: number) {
  const [, , width] = bbox;
  const ratio = width / imageWidth;

  if (ratio < 0.2) {
    return "far";
  }

  if (ratio < 0.5) {
    return "near";
  }

  return "very close";
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [selectedObject, setSelectedObject] = useState("bottle");
  const [status, setStatus] = useState("Loading model...");
  const [cameraStarted, setCameraStarted] = useState(false);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [scanning, setScanning] = useState(false);

  useEffect(() => {
    async function loadModel() {
      try {
        setStatus("Loading object detection model...");
        await tf.ready();

        const loadedModel = await cocoSsd.load();

        setModel(loadedModel);
        setStatus("Model loaded. Start camera.");
        speak("Model loaded. Start camera.");
      } catch (error) {
        console.error("MODEL LOAD ERROR:", error);
        setStatus("Error loading model. Check console.");
        speak("Error loading object detection model.");
      }
    }

    loadModel();
  }, []);

  async function startCamera() {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setStatus("Camera is not supported in this browser.");
        speak("Camera is not supported in this browser.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraStarted(true);
      setStatus(`Finding ${selectedObject}. Press scan now.`);
      speak(`Finding ${selectedObject}. Press scan now.`);
    } catch (error) {
      console.error("CAMERA ERROR:", error);
      setStatus("Camera permission denied or unavailable.");
      speak("Camera permission denied or unavailable.");
    }
  }

  async function scanNow() {
    try {
      if (!model) {
        setStatus("Model is not ready yet.");
        speak("Model is not ready yet.");
        return;
      }

      if (!videoRef.current || !canvasRef.current) {
        setStatus("Camera is not ready yet.");
        speak("Camera is not ready yet.");
        return;
      }

      const video = videoRef.current;
      const canvas = canvasRef.current;
      const context = canvas.getContext("2d");

      if (!context) {
        setStatus("Could not read camera image.");
        speak("Could not read camera image.");
        return;
      }

      setScanning(true);
      setStatus("Scanning...");
      speak("Scanning.");

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const results = (await model.detect(video)) as Prediction[];

      console.log("PREDICTIONS:", results);

      setPredictions(results);

      const confidentResults = results.filter((item) => item.score >= 0.5);

      const target = confidentResults.find(
        (item) => item.class.toLowerCase() === selectedObject.toLowerCase()
      );

      if (target) {
        const direction = getDirection(target.bbox, canvas.width);
        const distance = getDistance(target.bbox, canvas.width);
        const confidence = Math.round(target.score * 100);

        const message = `${selectedObject} detected ${direction}. It is ${distance}. Confidence ${confidence} percent.`;

        setStatus(message);
        speak(message);
      } else {
        const seenObjects = confidentResults.map((item) => item.class).join(", ");

        const message =
          seenObjects.length > 0
            ? `${selectedObject} not found. I can see ${seenObjects}. Move your phone slowly.`
            : `${selectedObject} not found. Move your phone slowly.`;

        setStatus(message);
        speak(message);
      }

      setScanning(false);
    } catch (error) {
      console.error("SCAN ERROR:", error);
      setStatus("Scan failed. Check console.");
      speak("Scan failed. Try again.");
      setScanning(false);
    }
  }

  function testFeedback() {
    const message = `${selectedObject} detected ahead. It is near.`;
    setStatus(message);
    speak(message);
  }

  return (
    <main className="app">
      <h1>Assistive Object Detection</h1>

      <p className="subtitle">
        Select an object, start the camera, then scan.
      </p>

      <section className="object-list">
        {OBJECTS.map((item) => (
          <button
            key={item}
            className={selectedObject === item ? "selected" : ""}
            onClick={() => {
              setSelectedObject(item);
              speak(`Selected ${item}`);
              setStatus(`Selected object: ${item}`);
            }}
          >
            {item}
          </button>
        ))}
      </section>

      <section className="camera-section">
        <video
          ref={videoRef}
          className="video"
          playsInline
          muted
          autoPlay
        />
        <canvas ref={canvasRef} className="hidden-canvas" />
      </section>

      <section className="controls">
        <button onClick={startCamera} disabled={!model}>
          Start Camera
        </button>

        <button onClick={scanNow} disabled={!cameraStarted || scanning}>
          {scanning ? "Scanning..." : "Scan Now"}
        </button>

        <button onClick={testFeedback}>Test Audio</button>
      </section>

      <p className="status">{status}</p>

      <section className="detections">
        <h2>Detected Objects</h2>

        {predictions.length === 0 ? (
          <p>No detections yet.</p>
        ) : (
          predictions.map((item, index) => (
            <p key={index}>
              {item.class} — {Math.round(item.score * 100)}%
            </p>
          ))
        )}
      </section>
    </main>
  );
}