import yolo from 'tfjs-yolo';
import * as tf from '@tensorflow/tfjs';

var player = document.getElementById('player');
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');

var appliedFilters = document.getElementById('appliedFilters');
var ctxAppliedFilters = appliedFilters.getContext('2d');

let myYolo;

(async function main() {
  try {
  myYolo = await yolo.v2tiny();
  
  var handleSuccess = function(stream) {
    player.srcObject = stream;
  };

  player.addEventListener('playing', function() {
    console.log(player.videoWidth);
    console.log(player.videoHeight);

    canvas.width = player.videoWidth;
    canvas.height = player.videoHeight;

    appliedFilters.width = player.videoWidth;
    appliedFilters.height = player.videoHeight;

  }, false);

  navigator.mediaDevices.getUserMedia({ audio: false, video: true })
      .then(handleSuccess)



  async function drawFrame(video) {
    
    ctxAppliedFilters.drawImage(video, 0, 0);
    await run(video);
    //var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    setTimeout(function () {
      drawFrame(video);
    }, 10);
  }

  drawFrame(player);
  } catch(e) {
    console.error(e);
  }
})();

async function run(video) {
  console.log("Start with tensors: " + tf.memory().numTensors);
  
  const boxes = await myYolo(appliedFilters);
  ctx.clearRect(0,0,canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0); //myYolo.predict(webcam, { scoreThreshold: threshold });
  console.log(boxes.length);
  boxes.map((box) => {
    console.log("drawing box " + box.toString());
    ctx.lineWidth = 2;
    ctx.fillStyle = "red";
    ctx.strokeStyle = "red";
    ctx.rect(box["left"], box["top"], box["width"], box["height"]);
    ctx.fillText(box["class"], box["left"] + 5, box["top"] + 10);
    ctx.stroke();
  });
  console.log("End with tensors: " + tf.memory().numTensors);
}


const identityKernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0,]];
const boxBlur3Kernel = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]];
const gaussianBlur3Kernel = [[1/16, 2/16, 1/16], [2/16, 4/16, 2/16], [1/16, 2/16, 1/16]];
const boxBlur5Kernel = [[1/25, 1/25, 1/25, 1/25, 1/25],[1/25, 1/25, 1/25, 1/25, 1/25],[1/25, 1/25, 1/25, 1/25, 1/25],[1/25, 1/25, 1/25, 1/25, 1/25],[1/25, 1/25, 1/25, 1/25, 1/25]];
const edgeDetectKernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]];

let boxBlur20Kernel = new Array(21);
for (let i=0; i<21; i++) {
  boxBlur20Kernel[i] = new Array(21);
  for(let j=0; j<21; j++) {
    boxBlur20Kernel[i][j] = 1/441;
  }
}


const kernel = boxBlur20Kernel;
const kernelHeight = kernel.length;
const kernelWidth = kernel[0].length;

function filter(data, box){


  for (let y = box["left"]; y < box["left"] + box["width"]; y++){
      for(let x = box["top"]; x < box["top"] + box["height"]; x++){
 

        if((y+kernelHeight)>player.videoHeight || (x+kernelWidth)>player.videoWidth)
          continue;

        let red = 0, green=0, blue=0, alpha=255;

        for(let ky = 0; ky < kernelHeight; ky++){
          for(let kx = 0; kx < kernelWidth; kx++){

            let kernelOffsetPixelIndex = ((y+ky)*player.videoWidth + (x+kx))*4;

            red += imageData.data[kernelOffsetPixelIndex] * kernel[ky][kx];
            green += imageData.data[kernelOffsetPixelIndex + 1] * kernel[ky][kx];
            blue += imageData.data[kernelOffsetPixelIndex + 2] * kernel[ky][kx];
          }
        }

        let pixelIndex = ((y+Math.floor(kernelHeight/2))*player.videoWidth + x + Math.floor(kernelWidth/2))*4;

        //console.log(red)
        data[pixelIndex] = red;
        data[pixelIndex+1] = green;
        data[pixelIndex+2] = blue;
        data[pixelIndex+3] = alpha;
      }
    }

}