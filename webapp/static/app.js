// DEVICE DETECTION & SUPPORT
import DeviceDetector from "https://cdn.skypack.dev/device-detector-js@2.2.10";
// Usage: testSupport({client?: string, os?: string}[])
// Client and os are regular expressions.
// See: https://cdn.jsdelivr.net/npm/device-detector-js@2.2.10/README.md for
// legal values for client and os
testSupport([{ client: "Chrome" }]);
function testSupport(supportedDevices) {
    const deviceDetector = new DeviceDetector();
    const detectedDevice = deviceDetector.parse(navigator.userAgent);
    let isSupported = false;
    for (const device of supportedDevices) {
        if (device.client !== undefined) {
            const re = new RegExp(`^${device.client}$`);
            if (!re.test(detectedDevice.client.name)) {
                continue;
            }
        }
        if (device.os !== undefined) {
            const re = new RegExp(`^${device.os}$`);
            if (!re.test(detectedDevice.os.name)) {
                continue;
            }
        }
        isSupported = true;
        break;
    }
    if (!isSupported) {
        alert(`This demo, running on ${detectedDevice.client.name}/${detectedDevice.os.name}, ` +
            `is not well supported at this time, continue at your own risk.`);
    }
}

// GLOBAL VARIABLES THAT STORE NAMES
var nameProvided = null
var nameMatch = null

// IMPORTANT CONSTANTS
/* UI Related */
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
const canvasCtx = canvasElement.getContext("2d");
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
    spinner.style.display = "none";
};
const controls = window;
const drawingUtils = window;
const mpFaceMesh = window;
const config = {
    locateFile: (file) => {
        return (`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh@` +
            `${mpFaceMesh.VERSION}/${file}`);
    }
};
const fpsControl = new controls.FPS();
/* For landmark subsetting */
const subset_idxs = [127, 234, 93, 215, 172, 136, 150, 176, 152, 400, 379,
    365, 367, 433, 366, 447, 372, 70, 63, 105, 66, 107, 336,
    296, 334, 293, 276, 168, 197, 195, 4, 240, 97, 2, 326,
    290, 33, 160, 158, 133, 153, 144, 362, 385, 386, 249, 373,
    380, 61, 39, 37, 11, 267, 269, 291, 321, 314, 17, 85, 181,
    78, 82, 13, 402, 308, 402, 14, 87]

// UTILITY FUNCTIONS FOR UI
function showDialog() {
    document.getElementById('dialog').style.display = 'block';
}
function submitName() {
    nameProvided = document.getElementById('nameInput').value;
    document.getElementById('dialog').style.display = 'none';
}
// UTILITY FUNCTIONS FOR DRAWING
function getBoundingBox(subsetLandmarks, width, height, margin=0.15){
    let bb = {
        minX: Math.round(Math.min(...subsetLandmarks.map(pt => pt.x * width))),
        minY: Math.round(Math.min(...subsetLandmarks.map(pt => pt.y * height))),
        maxX: Math.round(Math.max(...subsetLandmarks.map(pt => pt.x * width))),
        maxY: Math.round(Math.max(...subsetLandmarks.map(pt => pt.y * height)))
    };
    if (margin > 0){
        const marginX = Math.round(margin * (bb.maxX - bb.minX));
        const marginY = Math.round(margin * (bb.maxY - bb.minY));

        bb.minX -= marginX;
        bb.minY -= marginY;
        bb.maxX += marginX;
        bb.maxY += marginY;
    }

    return bb
}
function drawBoundingBox(canvasCtx, bb, style){
    canvasCtx.strokeStyle = style.color;
    canvasCtx.lineWidth = style.lineWidth;
    canvasCtx.strokeRect(bb.minX, bb.minY, bb.maxX-bb.minX, bb.maxY-bb.minY);
}
function drawName(canvasCtx, displayName, boundingBox, style){
    if (displayName != null){
        canvasCtx.fillStyle = style.color;
        canvasCtx.font = style.font;
        canvasCtx.fillText(displayName, boundingBox.minX,
                           boundingBox.maxY+30);
    }
}

// FUNCTION TO IDENTIFY A FACE W/ AJAX
function identify(imgDataURL, landmarks, boundingbox){
    let dataToSend = {
        frame_enc: imgDataURL,
        bb:boundingbox,
        landmarks: landmarks,
        name:nameProvided
    };
    fetch('/identify/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(dataToSend)
    })
    .then(response => response.json())
    .then(data => {
        nameMatch = null
        if ('displayName' in data){
            nameMatch = data.displayName
            if (nameProvided != data.nameProvided){
                nameProvided = data.nameProvided;
            }
        }else if ('message' in data){
            console.log('Warning:', data.message);
        }else if ('error' in data){
            console.error('Error:', data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        nameMatch = null
    });
}

// FUNCTION TO GET THE MEDIAPIPE FACEMESH RESULTS
function onResults(results) {
    document.body.classList.add("loaded");
    fpsControl.tick();
    const imageDataURL = results.image.toDataURL('image/png');
    var width = results.image.width;
    var height = results.image.height;
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
    if (results.multiFaceLandmarks) {
        for (const fullLandmarks of results.multiFaceLandmarks) {
            var subsetLandmarks = subset_idxs.map(index => fullLandmarks[index]);
            drawingUtils.drawLandmarks(canvasCtx, subsetLandmarks, {color:"#30FF30", fillColor:"#30FF30", radius:2, lineWidth:0});
            var subsetLandmarksNorm = subsetLandmarks.map(landmark => {
                const x = Math.round(landmark.x * width);
                const y = Math.round(landmark.y * height);
                return [x, y];
            });
            var bb = getBoundingBox(subsetLandmarks, width, height);
            var boundingBoxNorm = [bb.minX, bb.minY, bb.maxX, bb.maxY]
            var boundingBox = getBoundingBox(subsetLandmarks, canvasElement.width, canvasElement.height);
            identify(imageDataURL, subsetLandmarksNorm, boundingBoxNorm)
            drawBoundingBox(canvasCtx, boundingBox, {color:"#30FF30", lineWidth:2});
            drawName(canvasCtx, nameMatch, boundingBox, {color:"#30FF30", font:"30px Arial"})
        }

    }
    canvasCtx.restore();
}

// FOR CONTROLING FACEMESH
const faceMesh = new mpFaceMesh.FaceMesh(config);
const solutionOptions = {
    maxNumFaces: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
    selfieMode: true,
    enableFaceGeometry: false,
    refineLandmarks: true
};
faceMesh.setOptions(solutionOptions);
faceMesh.onResults(onResults);

new controls.ControlPanel(controlsElement, solutionOptions)
    .add([
    new controls.StaticText({ title: "Face Recognition" }),
    fpsControl,
    new controls.SourcePicker({
        onFrame: async (input, size) => {
            const aspect = size.height / size.width;
            let width, height;
            if (window.innerWidth > window.innerHeight) {
                height = window.innerHeight;
                width = height / aspect;
            }
            else {
                width = window.innerWidth;
                height = width * aspect;
            }
            canvasElement.width = width;
            canvasElement.height = height;
            await faceMesh.send({ image: input });
        }
    })
]).on((x) => {
    const options = x;
    faceMesh.setOptions(options);
});

// REGISTER EVENT LISTENING FOR REGISTERING NAME FUNCTIONALITY
document.getElementById('enterName').addEventListener('click', submitName);
document.getElementById('registerInVectorDB').addEventListener('click', showDialog);