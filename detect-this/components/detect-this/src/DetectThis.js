import { LitElement, html, css } from 'lit-element';

const NUM_FRAMES = 3
const INPUT_SHAPE = [NUM_FRAMES, 232, 1]
const earOptions = {
  overlapFactor: 0.999,
  includeSpectrogram: true,
  invokeCallbackOnNoiseAndUnknown: true
}

export class DetectThis extends LitElement {
  static get properties() {
    return {
      title: { type: String },
      trainingsAccuracy: { type: Number },
    };
  }

  constructor() {
    super();

    this.brain = null;
    this.knowledge = [];
    this.trainingsAccuracy = 0;
    this.predictions = [0, 0, 0, 0, 0, 0];
    this.ears = speechCommands.create('BROWSER_FFT');
  }

  normalizeAudio = (a, mean = -100, std = 100) => a.map(x => (x - mean) / std)
  flattenAudioSamples = tensors => {
    const size = tensors[0].length
    const result = new Float32Array(tensors.length * size)
    tensors.forEach((arr, i) => result.set(arr, i * size))

    return result
  }

  async turnRecorderOn() {
    await this.ears.ensureModelLoaded()

    this.brain = tf.sequential();

    this.brain.add(tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 4],
      activation: 'relu',
      inputShape: INPUT_SHAPE
    }));

    this.brain.add(tf.layers.maxPooling2d({
      poolSize: [1, 2],
      strides: [2, 2]
    }));

    this.brain.add(tf.layers.flatten());
    this.brain.add(tf.layers.dense({ units: 6, activation: 'softmax' }));

    this.brain.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    console.log('recorder is on');
  }

  learn(chord) {
    if (this.brain) {
      if (!this.ears.isListening()) {
        this.ears.listen(async ({ spectrogram }) => {
          const { frameSize, data } = spectrogram;

          const audioWaves = this.normalizeAudio(data.subarray(-frameSize * NUM_FRAMES));

          this.knowledge.push({ type: chord, audioWaves });

          console.log(`${this.knowledge.length} examples collected`);

        }, earOptions);
      } else {
        this.ears.stopListening();
        console.log('stopped');
      }
    } else {
      console.error('Brain not on');
    }
  }

  updateTrainingStats(epoch, logs) {
    this.trainingsAccuracy = (logs.acc * 100).toFixed(2);
  }

  async think() {
    if (this.knowledge.length) {
      const ys = tf.oneHot(this.knowledge.map(({ type }) => type), 6);
      const flattened = this.flattenAudioSamples(this.knowledge.map(({ audioWaves }) => audioWaves));
      const xs = tf.tensor(flattened, [this.knowledge.length, ...INPUT_SHAPE]);

      await this.brain.fit(xs, ys, {
        batchSize: 16,
        epochs: 20,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            console.log(`Epoch ${epoch}: loss=${logs.loss}, accuracy=${logs.acc}`);
            this.trainingsAccuracy = logs.acc;
          }
        }
      });

      tf.dispose([xs, ys]);
    } else {
      console.error('Need to collect knowledge before I can think')
    }
  }

  listen() {
    if (this.brain) {
      if (!this.ears.isListening()) {
        this.ears.listen(async ({ spectrogram }) => {
          const { frameSize, data } = spectrogram

          const audioWaves = this.normalizeAudio(data.subarray(-frameSize * NUM_FRAMES))

          const whatRobotHeard = tf.tensor(audioWaves, [1, ...INPUT_SHAPE])
          const whatRobotPredicts = this.brain.predict(whatRobotHeard)

          const labels = await whatRobotPredicts.data()

          labels.map((confidence, label) => {
            console.log(`Confidence: ${confidence} for label: ${label}`);
            this.predictions[label] = confidence;
          });

          this.requestUpdate();

          tf.dispose([audioWaves, whatRobotHeard, whatRobotPredicts])

        }, earOptions)
      } else {
        this.ears.stopListening();
        console.log('stopped');
      }
    } else {
      console.error('Empty brain can\'t predict.');
    }
  }

  render() {
    return html`
      <main>
        <button @click=${this.turnRecorderOn}>Start Recorder</button>
        <button @click=${() => this.learn(0)}>Learn A</button>
        <button @click=${() => this.learn(1)}>Learn D</button>
        <button @click=${() => this.learn(2)}>Learn E</button>
        <button @click=${() => this.learn(3)}>Learn G</button>
        <button @click=${() => this.learn(4)}>Learn C</button>
        <button @click=${() => this.learn(5)}>Learn Noise</button>
        <button @click=${this.think}>Think</button>
        <button @click=${this.listen}>Listen</button>
      </main>

      <div>A: ${this.predictions[0]}</div>
      <div>D: ${this.predictions[1]}</div>
      <div>E: ${this.predictions[2]}</div>
      <div>G: ${this.predictions[3]}</div>
      <div>C: ${this.predictions[4]}</div>
      <div>Noise: ${this.predictions[5]}</div>

      <p class="app-footer">
        Accuracy: ${this.trainingsAccuracy}%
      </p>
    `;
  }

  static get styles() {
    return [
      css`
        :host {
          min-height: 100vh;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: flex-start;
          font-size: calc(10px + 2vmin);
          color: #1a2b42;
          max-width: 960px;
          margin: 0 auto;
        }

        header {
          width: 100%;
          background: #fff;
          border-bottom: 1px solid #ccc;
        }

        header ul {
          display: flex;
          justify-content: space-between;
          min-width: 400px;
          margin: 0 auto;
          padding: 0;
        }

        header ul li {
          display: flex;
        }

        header ul li a {
          color: #ccc;
          text-decoration: none;
          font-size: 18px;
          line-height: 36px;
        }

        header ul li a:hover,
        header ul li a.active {
          color: #000;
        }

        main {
          flex-grow: 1;
        }

        .app-footer {
          color: #a8a8a8;
          font-size: calc(12px + 0.5vmin);
          align-items: center;
        }

        .app-footer a {
          margin-left: 5px;
        }
      `,
    ];
  }
}
