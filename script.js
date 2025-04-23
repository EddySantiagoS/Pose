let detector, model;
const video = document.getElementById("video");
const output = document.getElementById("output");

async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 480, height: 360 },
      audio: false,
    });
    video.srcObject = stream;
    return new Promise((resolve) => {
      video.onloadedmetadata = () => resolve(video);
    });
  } catch (err) {
    console.error("No se pudo acceder a la cámara:", err);
    output.innerText = "Error: Cámara no disponible.";
  }
}

async function loadModels() {
  try {
    // Carga modelo de clasificación
    model = await tf.loadLayersModel('modelo_posturas_web/model.json');
    console.log("Modelo cargado correctamente");

    // Carga MoveNet
    detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, {
      modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING
    });
    console.log("Detector MoveNet creado");
  } catch (err) {
    console.error("Error al cargar modelos:", err);
    output.innerText = "Error al cargar modelos.";
  }
}

function extractKeypoints(keypoints) {
  return keypoints
    .slice(0, 17)
    .map(k => [k.x / 480, k.y / 360])  // Normalizado
    .flat();
}

function classifyPose(keypointsArray) {
  const input = tf.tensor2d([keypointsArray]);
  const prediction = model.predict(input);
  const classIndex = prediction.argMax(-1).dataSync()[0];
  const classes = ["Brazos arriba", "En cuclillas", "Pose de yoga", "Sentado", "pie"];
  output.innerText = `Postura detectada: ${classes[classIndex]}`;
}

async function detectPose() {
  const poses = await detector.estimatePoses(video);
  if (poses.length > 0 && poses[0].keypoints) {
    const keypoints = extractKeypoints(poses[0].keypoints);
    classifyPose(keypoints);
  }
  requestAnimationFrame(detectPose);
}

async function main() {
  await setupCamera();
  await loadModels();
  output.innerText = "Detectando...";
  detectPose();
}

main();
