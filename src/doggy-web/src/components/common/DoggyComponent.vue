<template>
  <div class="imagenet">
    <v-alert outline color="error" icon="priority_high" :value="!hasWebGL">
      Note: this browser does not support WebGL 2 or the features necessary to run in GPU mode.
    </v-alert>
    <div class="ui-container">
      <v-layout row justify-center class="input-label">
        Enter a valid image URL or select an image from the dropdown:
      </v-layout>
      <v-layout row wrap justify-center align-center>
        <v-flex xs7 md5>
          <v-text-field
            v-model="imageURLInput"
            :disabled="modelLoading || modelInitializing"
            label="enter image url"
            @keyup.native.enter="onImageURLInputEnter"
          ></v-text-field>
        </v-flex>
        <v-flex xs1 class="input-label text-xs-center">or</v-flex>
        <v-flex xs5 md3>
          <v-select
            v-model="imageURLSelect"
            :disabled="modelLoading || modelInitializing"
            :items="imageURLSelectList"
            label="select image"
            max-height="750"
          ></v-select>
        </v-flex>
        <v-flex xs2 md2 class="controls">
          <v-switch label="use GPU"
                    v-model="useGPU"
                    :disabled="modelLoading || modelInitializing || modelRunning || !hasWebGL"
                    color="primary"
          ></v-switch>
        </v-flex>
      </v-layout>
      <v-layout row wrap justify-center class="image-panel elevation-1">
        <div v-if="imageLoadingError" class="error-message">Error loading URL</div>
        <v-flex sm5 md3 class="canvas-container">
          <canvas id="input-canvas"
                  :width="imageSize"
                  :height="imageSize"
          ></canvas>
          <transition name="fade">
            <div v-show="showVis">
              <canvas id="visualization-canvas"
                      :width="imageSize"
                      :height="imageSize"
                      :style="{ opacity: colormapSelect === 'transparency' ? 1 : colormapAlpha }"
              ></canvas>
            </div>
          </transition>
        </v-flex>
        <v-flex sm6 md4 class="output-container">
          <div>
            <div v-if="imageLoading || modelRunning">
              <v-progress-circular indeterminate color="primary"/>
            </div>

            <div class="inference-time">
              <span>inference time: </span>
              <span v-if="inferenceTime > 0" class="inference-time-value">{{ inferenceTime.toFixed(1)
                }} ms </span>
              <span v-if="inferenceTime > 0"
                    class="inference-time-value">({{ (1000 / inferenceTime).toFixed(1) }} fps)</span>
              <span v-else>-</span>
            </div>
            <div v-for="i in [0, 1, 2, 3, 4]" :key="i"
                 class="output-class"
                 :class="{ predicted: i === 0 && outputClasses[i].probability.toFixed(2) > 0 }"
            >
              <div class="output-label">{{ outputClasses[i].name }}</div>
              <div class="output-bar"
                   :style="{width: `${Math.round(100 * outputClasses[i].probability)}px`, background: `rgba(27, 188, 155, ${outputClasses[i].probability.toFixed(2)})` }"
              ></div>
              <div class="output-value">{{ Math.round(100 * outputClasses[i].probability) }}%</div>
            </div>
          </div>
        </v-flex>
      </v-layout>
    </div>
  </div>
</template>

<script>
  import loadImage from 'blueimp-load-image'
  import ndarray from 'ndarray'
  import ops from 'ndarray-ops'
  import resample from 'ndarray-resample'
  import {imagenetUtils} from '../../utils'
  import {IMAGE_URLS} from '../../data/sample-image-urls'
  import {COLORMAPS} from '../../data/colormaps'
  import ModelStatus from './ModelStatus'
  import {Model} from 'keras-js'

  export default {
    props: {
      modelName: {type: String, required: true},
      hasWebGL: {type: Boolean, required: true},
      imageSize: {type: Number, required: true},
      visualizations: {type: Array, required: true},
      preprocess: {type: Function, required: true}
    },

    components: {ModelStatus},

    created() {
      // store module on component instance as non-reactive object
      this.model = new Model({
        filepaths: {
          model: 'trained/model.json',
          weights: 'trained/model_weights.buf',
          metadata: 'trained/model_metadata.json'
        },
        gpu: this.hasWebGL,
        headers: {},
        filesystem: false,
        transferLayerOutputs: false,
        pauseAfterLayerCalls: false,
        visualizations: []

      })
    },

    async mounted() {
      await this.model.ready();
      this.modelLoading = false;
      this.modelInitializing = false;
      this.model.visMap = {};
      await this.$nextTick()
      this.modelLayersInfo = this.model.modelLayersInfo
    },

    beforeDestroy() {
      this.model.cleanup()
      this.model.events.removeAllListeners()
    },

    data() {
      const visualizationSelect = this.visualizations[0]
      const visualizationSelectList = [{text: 'None', value: 'None'}]
      if (['squeezenet_v1.1', 'inception_v3', 'densenet121'].includes(this.modelName)) {
        visualizationSelectList.push({text: 'Class Activation Mapping', value: 'CAM'})
      }

      return {
        useGPU: this.hasWebGL,
        modelLoading: true,
        modelLoadingProgress: 0,
        modelInitializing: true,
        modelInitProgress: 0,
        modelLayersInfo: [],
        modelRunning: false,
        inferenceTime: null,
        imageURLInput: 'https://static9.depositphotos.com/1594920/1171/i/950/depositphotos_11718919-stock-photo-belgian-shepherd-dog-puppy-5.jpg',
        imageURLSelect: null,
        imageURLSelectList: IMAGE_URLS,
        imageLoading: false,
        imageLoadingError: false,
        visualizationSelect,
        visualizationSelectList,
        colormapSelect: 'transparency',
        colormapSelectList: COLORMAPS,
        colormapAlpha: 0.7,
        showVis: false,
        output: null,
        loaded: false
      }
    },

    computed: {
      outputClasses() {
        if (!this.output) {
          const empty = [];
          for (let i = 0; i < 5; i++) {
            empty.push({name: '-', probability: 0});
          }
          return empty;
        }

        const classes = ["jumping", "laying", "rolling", "sitting", "standing"];
        const result = [];
        for (let i = 0; i < 5; i++) {
          result.push({name: classes[i], probability: this.output[i]});
        }
        return result;
      }
    },

    watch: {
      imageURLSelect(newVal) {
        this.imageURLInput = newVal
        this.loadImageToCanvas(newVal)
      },
      useGPU(newVal) {
        this.model.toggleGPU(newVal)
      },
      visualizationSelect(newVal) {
        if (newVal === 'None') {
          this.showVis = false
        } else {
          this.updateVis(this.outputClasses[0].index)
        }
      },
      colormapSelect() {
        this.updateVis(this.outputClasses[0].index)
      }
    },

    methods: {
      onImageURLInputEnter(e) {
        this.imageURLSelect = null
        this.loadImageToCanvas(e.target.value)
      },
      loadImageToCanvas(url) {
        if (!url) {
          this.clearAll()
          return
        }

        this.imageLoading = true
        loadImage(
          url,
          img => {
            if (img.type === 'error') {
              this.imageLoadingError = true
              this.imageLoading = false
            } else {
              // load image data onto input canvas
              const ctx = document.getElementById('input-canvas').getContext('2d')
              ctx.drawImage(img, 0, 0)
              this.imageLoadingError = false
              this.imageLoading = false
              this.modelRunning = true
              // model predict
              this.$nextTick(function () {
                setTimeout(() => {
                  this.runModel()
                }, 10)
              })
            }
          },
          {
            maxWidth: this.imageSize,
            maxHeight: this.imageSize,
            cover: true,
            crop: true,
            canvas: true,
            crossOrigin: 'Anonymous'
          }
        )
      },
      runModel() {
        const ctx = document.getElementById('input-canvas').getContext('2d');
        const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
        const flattenData = new Float32Array(imageData.data);

        const preprocessedData = this.preprocess(imageData);

        const inputName = "input_1";
        const outputName = "dense_2";

        const inputData = {[inputName]: preprocessedData};
        const startTime = Date.now();
        this.model.predict(inputData).then(outputData => {
          this.loaded = true;
          this.inferenceTime = Date.now() - startTime;
          this.output = outputData[outputName];
          this.modelRunning = false;
        })
      },

      clearAll() {
        this.modelRunning = false
        this.inferenceTime = null
        this.imageURLInput = null
        this.imageURLSelect = null
        this.imageLoading = false
        this.imageLoadingError = false
        this.output = null

        const ctx = document.getElementById('input-canvas').getContext('2d')
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)
      }
    }
  }
</script>

<style lang="postcss" scoped>
  @import '../../variables.css';

  .ui-container {
    font-family: var(--font-monospace);
    margin-bottom: 30px;
  }

  .input-label {
    font-family: var(--font-cursive);
    font-size: 16px;
    color: var(--color-lightgray);
    text-align: left;
    user-select: none;
    cursor: default;
  }

  .controls {
    width: 100px;
    margin-left: 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
  }

  .image-panel {
    padding: 30px 20px;
    background-color: whitesmoke;
    position: relative;

  }

  .error-message {
    color: var(--color-error);
    font-size: 12px;
    position: absolute;
    top: 5px;
    left: 5px;
  }

  .visualization {
    margin-right: 20px;
    position: relative;

  }
  .colormap-alpha {
    position: relative;

  }
  label {
    position: absolute;
    color: var(--color-darkgray);
    font-size: 10px;
  }

  .visualization-instruction {
    position: absolute;
    bottom: 10px;
    left: 0;
    font-size: 12px;
    color: var(--color-lightgray);
  }

  .canvas-container {
    position: relative;
    margin: 0 20px;

  }
  #input-canvas {
    background: #eeeeee;
  }

  #visualization-canvas {
    pointer-events: none;
    position: absolute;
    top: 0;
    left: 0;
  }


  .output-container {
    margin-top: 10px;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;

  }
  .inference-time {
    align-self: center;
    font-family: var(--font-monospace);
    font-size: 14px;
    color: var(--color-lightgray);
    margin-bottom: 10px;

  }
  .inference-time-value {
    color: var(--color-green);
  }

  .output-class {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: center;
    padding: 6px 0;

  }
  .output-label {
    text-align: right;
    width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: var(--font-monospace);
    font-size: 16px;
    color: var(--color-darkgray);
    padding: 0 6px;
    border-right: 2px solid var(--color-green-lighter);
  }

  .output-bar {
    height: 8px;
    transition: width 0.2s ease-out;
  }

  .output-value {
    text-align: left;
    margin-left: 5px;
    font-family: var(--font-monospace);
    font-size: 14px;
    color: var(--color-lightgray);
  }

  .output-class.predicted {
  }

  .output-label {
    color: var(--color-green);
    border-left-color: var(--color-green);
  }

  .output-value {
    color: var(--color-green);
  }


  /* vue transition `fade` */
  .fade-enter-active,
  .fade-leave-active {
    transition: opacity 0.5s;
  }

  .fade-enter,
  .fade-leave-to {
    opacity: 0;
  }
</style>
