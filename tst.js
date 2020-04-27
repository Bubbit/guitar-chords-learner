const tf = require('@tensorflow/tfjs');
const dft = require('fft-js').dft
const fs = require('fs');
const speechCommands = require('@tensorflow-models/speech-commands');

global.fetch = require("node-fetch");

const NUM_FRAMES = 3
const INPUT_SHAPE = [NUM_FRAMES, 232, 1]
const earOptions = {
  overlapFactor: 0.999,
  includeSpectrogram: true,
  invokeCallbackOnNoiseAndUnknown: true
}

const chords = ['a'];

const normalizeAudio = (a) => {
  const maxA = Math.max(...a);
  return a.map(x => x / maxA);
}

const flattenAudioSamples = tensors => {
  const size = tensors[0].length;
  console.log(size);
  console.log(tensors.length);
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));

  return result;
}

const createSet = (start, end) => {
  const trainingSet = [];
  chords.forEach((chord, index) => {
    for (let i = start; i < end; i++) {
      data = fs.readFileSync(`assets/${chord}/${chord}${i}.wav`);
      const bufferNewSamples = new Float32Array(data);

      for (let j = 0; j < 5; j += 1024) {
        const phasors = dft(normalizeAudio(bufferNewSamples.slice(j, j + 1024)));
        console.log(phasors);
      //   const phasors = dft(bufferNewSamples.slice(j, j + 1024));
      //   // console.log(phasors);
        // trainingSet.push({ chord: index, data: phasors });
      }
    }
  });

  return trainingSet;
}

const think = async (model, trainingSet) => {
  console.log(trainingSet[0]);
  const ys = tf.data.array(trainingSet.map(({ chord }) => chord));
  const xs = tf.data.array(trainingSet.map(({ data }) => data));
  const ds = tf.data.zip({ xs, ys }).shuffle(100).batch(32);

  await await model.fitDataset(ds, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
      }
    }
  });

  tf.dispose([xs, ys]);

  return model;
}

const run = async () => {
  // const recognizer = speechCommands.create('BROWSER_FFT');

  // await recognizer.ensureModelLoaded();
  // // Inspect the input shape of the recognizer's underlying tf.Model.
  // console.log(recognizer.modelInputShape());
  // // You will get something like [null, 43, 232, 1].
  // // - The first dimension (null) is an undetermined batch dimension.
  // // - The second dimension (e.g., 43) is the number of audio frames.
  // // - The third dimension (e.g., 232) is the number of frequency data points in
  // //   every frame (i.e., column) of the spectrogram
  // // - The last dimension (e.g., 1) is fixed at 1. This follows the convention of
  // //   convolutional neural networks in TensorFlow.js and Keras.

  // // Inspect the sampling frequency and FFT size:
  // console.log(recognizer.params().sampleRateHz);
  // console.log(recognizer.params().fftSize);

  // let model = tf.sequential();

  // model.add(tf.layers.depthwiseConv2d({
  //   depthMultiplier: 8,
  //   kernelSize: [NUM_FRAMES, 4],
  //   activation: 'relu',
  //   inputShape: INPUT_SHAPE
  // }));

  // model.add(tf.layers.maxPooling2d({
  //   poolSize: [1, 2],
  //   strides: [2, 2]
  // }));

  // model.add(tf.layers.flatten());
  // model.add(tf.layers.dense({ units: 6, activation: 'softmax' }));

  // model.compile({
  //   optimizer: tf.train.adam(0.01),
  //   loss: 'categoricalCrossentropy',
  //   metrics: ['accuracy']
  // });

  const trainingSet = createSet(1, 51);
  // model = think(model, trainingSet);
}

run();

// fs.readFile('assets/a/a2.wav', (err, data) => {
//   if (err) throw err;
//   console.log(tf.tensor(new Float32Array(data)));
// });