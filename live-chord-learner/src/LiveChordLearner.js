import { LitElement, html, css } from 'lit-element';
import * as tf from '@tensorflow/tfjs'
import * as speechCommands from '@tensorflow-models/speech-commands'

const NUM_FRAMES = 3
const INPUT_SHAPE = [NUM_FRAMES, 232, 1]
const earOptions = {
  overlapFactor: 0.999,
  includeSpectrogram: true,
  invokeCallbackOnNoiseAndUnknown: true
}

export class LiveChordLearner extends LitElement {
  static get properties() {
    return {
      title: { type: String },
    };
  }

  constructor() {
    super();

    this.brain = null;
    this.knowledge = [];
    // ears = speechCommands.create('BROWSER_FFT');
  }

  turnRecorderOn() {
    //  await robot.ears.ensureModelLoaded()

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
    this.brain.add(tf.layers.dense({ units: 4, activation: 'softmax' }));

    this.brain.compile({
      optimizer: tf.train.adam(0.01),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    // this.initiateBody()
  }

  render() {
    return html`

      <main>
        <button @click=${this.turnRecorderOn}>Start Recorder</button>
      </main>

      <p class="app-footer">
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
