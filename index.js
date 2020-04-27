const AudioContext = require('web-audio-api').AudioContext
const fs = require('fs');
const { exec } = require('child_process');
const np = require('numjs');
const plot = require('plotter').plot;
const fftjs = require('fft-js').fft;

const context = new AudioContext;
let pcmdata;

const soundfile = "assets/a/a1.wav"

decodeSoundFile(soundfile);

const note_references = [16.35, 17.32, 18.35, 19.45, 20.60, 21.83, 23.12, 24.50, 25.96, 27.50, 29.14, 30.87];

function nearestPow2( aSize ){
  return Math.pow( 2, Math.floor( Math.log( aSize ) / Math.log( 2 ) ) ); 
}

function m_func(l, p, frames) {
  //M(l) = round(12 * log_2( (f_s*l)/(N*f_ref) ) ) % 12
  let a = samplerate * l;
  let b = frames * note_references[p];
  //M(l) = round(12 * log_2( (a)/(b) ) ) % 12
  let c = 12 * Math.log2(a / b);
  console.log(c);
  let d = np.round(c);
  let e = np.mod(d, 12);
  //#print "Result: ", e
//#raw_input()
  return e;
}

function pcp(fft, p) {
  let r = 0

  fft.forEach((l) => {
    let result = m_func(l[0], p, fft.length);
    if(result === p) {
      r += 1;
    }
  });

  return r;
}

function calculate_PCP(fft_results) {
  const resultBins = [];

  for(let p = 0; p < 12; p++) {
    resultBins[p] = pcp(fft_results, p);
  }

  return resultBins;
}


function decodeSoundFile(soundfile){
  console.log("decoding wav file ", soundfile, " ..... ")
  fs.readFile(soundfile, function(err, buf) {
    if (err) throw err
    context.decodeAudioData(buf, function(audioBuffer) {
      pcmdata = np.float32(audioBuffer.getChannelData(0));
      samplerate = audioBuffer.sampleRate; // store sample rate
  
      plot({
        data:		pcmdata.tolist(),
        filename:	'before_normalize.png'
      });
      //normalize
      pcmdata = np.divide(pcmdata, pcmdata.max());

      console.log(pcmdata);
      plot({
        data:		pcmdata.tolist(),
        filename:	'after_normalize.png'
      });

      console.log(pcmdata.tolist().length);

      const phasors = fftjs(pcmdata.slice([nearestPow2(pcmdata.tolist().length)]).tolist());

      console.log(calculate_PCP(phasors));
      // plot({
      //   data:		phasors,
      //   filename:	'after_fft.png'
      // });
    }, function(err) { throw err })
  })
}
