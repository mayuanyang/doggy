<template>
  <div class="demo">
    <doggy-component
      modelName="doggy_model"
      :hasWebGL="true"
      :imageSize="299"
      :visualizations="['CAM']"
      :preprocess="preprocess"
    ></doggy-component>
  </div>
</template>

<script>
import ndarray from 'ndarray'
import ops from 'ndarray-ops'
import DoggyComponent from '../common/DoggyComponent'

const MODEL_FILEPATH_PROD = '/demos/data/doggy/example.bin'
const MODEL_FILEPATH_DEV = '/demos/data/doggy/example.bin'

export default {
  props: ['hasWebGL'],

  components: { DoggyComponent },

  data() {
    return {};
  },

  methods: {
    preprocess(imageData) {
      const { data, width, height } = imageData;

      // data processing
      // see https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py
      const dataTensor = ndarray(new Float32Array(data), [width, height, 4])
      const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [width, height, 3])

      ops.divseq(dataTensor, 255)
      ops.subseq(dataTensor, 0.5)
      ops.mulseq(dataTensor, 2)
      ops.assign(dataProcessedTensor.pick(null, null, 0), dataTensor.pick(null, null, 0))
      ops.assign(dataProcessedTensor.pick(null, null, 1), dataTensor.pick(null, null, 1))
      ops.assign(dataProcessedTensor.pick(null, null, 2), dataTensor.pick(null, null, 2))

      const preprocessedData = dataProcessedTensor.data
      return preprocessedData
    }
  }
}
</script>

<style scoped lang="postcss">

</style>
