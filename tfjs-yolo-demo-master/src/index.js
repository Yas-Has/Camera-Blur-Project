import yolo from 'tfjs-yolo';
import * as tf from '@tensorflow/tfjs';

var player = document.getElementById('player');
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
var playerWidth = player.videoWidth;
var playerHeight = player.videoHeight;

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
    ctx.drawImage(video, 0, 0);
    var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    var imageDataFiltered = ctxAppliedFilters.createImageData(appliedFilters.width, appliedFilters.height);
    
    for(let i = 0; i < imageData.data.length; i++){
      imageDataFiltered.data[i] = imageData.data[i];
    }

    imageDataFiltered = await run(video, imageData, imageDataFiltered);
    console.log(imageData.data)
    console.log(imageDataFiltered.data)
    //var imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    ctxAppliedFilters.putImageData(imageDataFiltered,0,0);
    setTimeout(function () {
      drawFrame(video);
    }, 10);
  } 

  drawFrame(player);
  } catch(e) {
    console.error(e);
  }
})();

async function run(video, imageData, imageDataFiltered) {
  console.log("Start with tensors: " + tf.memory().numTensors);
  
  const boxes = await myYolo(canvas);
  ctx.clearRect(0,0,canvas.width, canvas.height);
 //myYolo.predict(webcam, { scoreThreshold: threshold });
  console.log(boxes);

  
  boxes.map((box) => {
    ctx.drawImage(video, 0, 0);
    console.log("drawing box " + box.toString());
    imageDataFiltered = filter(imageData, imageDataFiltered, box)
    
    // ctx.lineWidth = 2;
    // ctx.fillStyle = "red";
    // ctx.strokeStyle = "red";
    // ctx.rect(box["left"], box["top"], box["width"], box["height"]);
    // ctx.fillText(box["class"], box["left"] + 5, box["top"] + 10);
    // ctx.stroke();
  });
  console.log("End with tensors: " + tf.memory().numTensors);
  return imageDataFiltered;
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

function filter(data, data2, box){
  
 

    let left = parseInt(box["left"])
    let width = parseInt(box["width"])
    let top = parseInt(box["top"])
    let height = parseInt(box["height"])
  

  console.log("filtering")

  
  for (let y = top; y < top + height; y++){
      for(let x = left; x < left + width; x++){
  
  // for (let y = 10; y < 100; y++){
  //   for(let x = 0; x < 100; x++){
      //console.log(x + ", " + y)

      if((y+kernelHeight)>player.videoHeight || (x+kernelWidth)>player.videoWidth)
        continue;

      let red = 0, green=0, blue=0, alpha=255;

      for(let ky = 0; ky < kernelHeight; ky++){
        for(let kx = 0; kx < kernelWidth; kx++){

          let kernelOffsetPixelIndex = ((y+ky)*player.videoWidth + (x+kx))*4;

          red += data.data[kernelOffsetPixelIndex] * kernel[ky][kx];
          green += data.data[kernelOffsetPixelIndex + 1] * kernel[ky][kx];
          blue += data.data[kernelOffsetPixelIndex + 2] * kernel[ky][kx];
        }
      }

      let pixelIndex = ((y+Math.floor(kernelHeight/2))*player.videoWidth + x + Math.floor(kernelWidth/2))*4;

      //console.log(red)
      data2.data[pixelIndex] = red;
      data2.data[pixelIndex+1] = green;
      data2.data[pixelIndex+2] = blue;
      data2.data[pixelIndex+3] = alpha;
      // data2.data[y*640*4 + x*4] = 255;
      // data2.data[y*640*4 + x*4 + 1] = 255;
      // data2.data[y*640*4 + x*4 + 2] = 255;
      // data2.data[y*640*4 + x*4 + 3] = 255;
    }
  }
  

    // for(let i = 0; i < data2.data.length; i++){
    //   data2.data[i] = 200;
    // }

    // for(let i = 0; i< data.data.length; i++){
    //   if(data.data[i] != data2.data[i]){
    //     console.log("working");
    //   }
    // }
    
    
    return data2;

}

// function collide(boxArray, unBlurredIndex){
//   let rect1 = boxArray[unBlurredIndex];
//   for (var i = 0; i < boxArray.length; i++) {
//     let rect2 = boxArray[i]
//     if(i != unBlurredIndex){
//       if(rect1["left"] < rect2["left"] + rect2["width"] && rect1["left"] + rect1["width"] > rect2["width"] && rect1["top"] < rect2["top"] + rect2["height"] && rect1["top"] + rect1["height"] > rect2["top"]){
//         return true;
//       }

//       //rect1.x < rect2.x + rect2.w && rect1.x + rect1.w > rect2.x && rect1.y < rect2.y + rect2.h && rect1.h + rect1.y > rect2.y
//     }
//   }
// }