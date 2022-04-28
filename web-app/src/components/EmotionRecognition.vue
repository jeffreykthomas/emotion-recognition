<template>
  <div class="camera2">
    <div class="overlay-spinner" v-if="loading"></div>
    <MDBSpinner class="spin" size="lg" v-if="loading"></MDBSpinner>
    <section v-if="detection">
      <MDBRow>
        <MDBCol>
          <h3 class="pt-5">Emotion Detection</h3>
        </MDBCol>
      </MDBRow>
      <MDBRow class="pt-3" v-show="!video">
        <MDBCol>
          <MDBBtn @click="video = true; photo = false; startWebcam()">
            Use Camera
          </MDBBtn>
          <h6 class="pt-3">OR</h6>
        </MDBCol>
      </MDBRow>
      <MDBRow v-if="video && !photo">
        <MDBCol>
          <p>Webcam feed</p>
          <video id="videoInput" ref="videoInput" class="vid" playsinline></video>
          <canvas id="canvasOutput" ref="canvasOutput"></canvas>
          <MDBBtn @click="detectEmotions" class="d-block mx-auto mb-3">
            <span v-if="!streaming">Start</span>
            <span v-if="streaming">Stop</span>
          </MDBBtn>
        </MDBCol>
      </MDBRow>
      <MDBRow v-show="!streaming && !photo">
        <MDBCol>
          <h6 class="pt-3">OR</h6>
          <MDBBtn @click="video = false; photo = true;" class="my-3">
            Upload Photo
          </MDBBtn>
        </MDBCol>
      </MDBRow>

      <MDBRow v-if="photo && !video" center class="pt-3">
        <MDBCol md="4">
          <MDBFile v-model="uploadedPhoto" accept=".jpg, .jpeg, .png" label="Upload Photo" @change="loadImageFromUpload"></MDBFile>
        </MDBCol>
      </MDBRow>
      <MDBRow v-show="photo" class="pt-3">
        <MDBCol>
          <img id="inputImg" ref="inputImg" src="" class="img-fluid" alt="uploaded-img" v-show="showPhoto" />
          <canvas id="photoCanvas" ref="photoCanvas"></canvas>
        </MDBCol>
      </MDBRow>
    </section>
    <section v-if="generate">
      <MDBRow center>
        <MDBCol md="8">
          <h3 class="pt-5">Face Generation</h3>
        </MDBCol>
      </MDBRow>
      <MDBRow>
        <MDBCol>
          <h4>Choose Emotion</h4>
        </MDBCol>
      </MDBRow>
      <MDBRow class="d-none d-md-flex">
        <MDBCol>
          <MDBBtnGroup class="shadow-0">
            <MDBBtn outline="primary" @click="generateFace(0)">Anger</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(2)">Disgust</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(3)">Fear</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(4)">Happiness</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(5)">Neutral</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(6)">Sadness</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(7)">Surprise</MDBBtn>
          </MDBBtnGroup>
        </MDBCol>
      </MDBRow>
      <MDBRow class="d-md-none">
        <MDBCol>
          <MDBBtnGroup vertical class="shadow-0">
            <MDBBtn outline="primary" @click="generateFace(0)">Anger</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(2)">Disgust</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(3)">Fear</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(4)">Happiness</MDBBtn>
          </MDBBtnGroup>
          <MDBBtnGroup vertical class="ms-3">
            <MDBBtn outline="primary" @click="generateFace(5)">Neutral</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(6)">Sadness</MDBBtn>
            <MDBBtn outline="primary" @click="generateFace(7)">Surprise</MDBBtn>
          </MDBBtnGroup>
        </MDBCol>
      </MDBRow>
      <MDBRow class="pt-3">
        <MDBCol>
          <canvas id="generateCanvas" ref="generateCanvas" width="200" height="200"></canvas>
        </MDBCol>
      </MDBRow>
    </section>
    <section v-if="transform">
      <MDBRow center>
        <MDBCol lg="4">
          <h3 class="pt-5">Emotion Transformation</h3>
          <p>Upload a photo with <strong>smiling</strong> faces (up to 4)</p>
          <MDBFile v-model="uploadedTransformPhoto" accept=".jpg, .jpeg, .png" @change="loadImageFromUpload"></MDBFile>
        </MDBCol>
      </MDBRow>
      <MDBRow class="pt-5" v-show="uploadedTransformPhoto.length">
        <MDBCol>
          <h4>Now choose an emotion</h4>
        </MDBCol>
      </MDBRow>
      <MDBRow v-show="uploadedTransformPhoto.length" center>
        <MDBCol class="d-none d-md-inline">
          <MDBBtnGroup class="shadow-0">
            <MDBBtn outline="primary" @click="changeEmotion(0)">Anger</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(1)">Disgust</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(2)">Fear</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(3)">Neutral</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(4)">Sadness</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(5)">Surprise</MDBBtn>
          </MDBBtnGroup>
        </MDBCol>
        <MDBCol class="d-md-none">
          <MDBBtnGroup vertical class="shadow-0">
            <MDBBtn outline="primary" @click="changeEmotion(0)">Anger</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(1)">Disgust</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(2)">Fear</MDBBtn>
          </MDBBtnGroup>
          <MDBBtnGroup vertical class="shadow-0">
            <MDBBtn outline="primary" @click="changeEmotion(3)">Neutral</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(4)">Sadness</MDBBtn>
            <MDBBtn outline="primary" @click="changeEmotion(5)">Surprise</MDBBtn>
          </MDBBtnGroup>
        </MDBCol>
      </MDBRow>
<!--      <MDBRow class="d-md-none pt-5">-->
<!--        <MDBCol>-->
<!--          <MDBBtnGroup vertical class="shadow-0">-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(0)">Anger</MDBBtn>-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(1)">Disgust</MDBBtn>-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(2)">Fear</MDBBtn>-->
<!--          </MDBBtnGroup>-->
<!--          <MDBBtnGroup vertical class="ms-3">-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(3)">Neutral</MDBBtn>-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(4)">Sadness</MDBBtn>-->
<!--            <MDBBtn outline="primary" @click="changeEmotion(5)">Surprise</MDBBtn>-->
<!--          </MDBBtnGroup>-->
<!--        </MDBCol>-->
<!--      </MDBRow>-->
      <MDBRow class="pt-5">
        <MDBCol>
          <img id="inputTransformImg" ref="inputTransformImg" src="" class="img-fluid" alt="uploaded-img" v-show="showTransformPhoto" />
          <canvas id="transformCanvas1" ref="transformCanvas1"></canvas>
          <canvas id="transformCanvas2" ref="transformCanvas2"></canvas>
          <canvas id="transformCanvas3" ref="transformCanvas3"></canvas>
          <canvas id="transformCanvas4" ref="transformCanvas4"></canvas>
        </MDBCol>
      </MDBRow>
    </section>
  </div>
</template>

<script>
import { MDBBtn, MDBCol, MDBFile, MDBRow, MDBSpinner, MDBBtnGroup } from 'mdb-vue-ui-kit'
import {ref} from 'vue'

export default {
  name: "EmotionRecognition",
  components: {
    MDBRow,
    MDBCol,
    MDBBtn,
    MDBFile,
    MDBSpinner,
    MDBBtnGroup
  },
  props: ['generate', 'transform', 'detection'],
  setup() {
    const canvasOutput = ref(null)
    const photoCanvas = ref(null)
    const transformCanvas1 = ref(null)
    const transformCanvas2 = ref(null)
    const transformCanvas3 = ref(null)
    const transformCanvas4 = ref(null)
    const generateCanvas = ref(null)
    const uploadedPhoto = ref([])
    const uploadedTransformPhoto = ref([])
    return {
      canvasOutput,
      photoCanvas,
      transformCanvas1,
      transformCanvas2,
      transformCanvas3,
      transformCanvas4,
      generateCanvas,
      uploadedPhoto,
      uploadedTransformPhoto
    }
  },
  data() {
    return {
      photo: true,
      video: false,
      streaming: false,
      emotionChosen: false,
      files: [],
      model: '',
      time: 0,
      fps: 0,
      tensors: 0,
      showPhoto: false,
      showTransformPhoto: false,
      loading: false,
      activeBtn: ''
    }
  },
  async created() {
  },
  async mounted() {
    window.addEventListener('resize', this.setCanvasSize)
    if (this.video) {
      await this.$faceapi.nets.tinyFaceDetector.loadFromUri('/models')
      await this.detectEmotions()
      this.$refs.videoInput.addEventListener('resize', this.setCanvasSize)
    } else if (this.photo || this.transform) {
      await this.$faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
    }
  },
  beforeUnmount() {
    console.log('before unmount')
    if(this.video){
      this.streaming = false
      this.$refs.videoInput.removeEventListener('resize', this.setCanvasSize)
      navigator.mediaDevices.getUserMedia({video: true, audio: false})
          .then(function (stream) {
            stream.getTracks().forEach(track => {
              track.stop()
              console.log('webcam stopped')
            })
          })
          .catch(function (err) {
            console.log("An error occurred! " + err)
          });
    }
  },
  unmounted() {
    window.removeEventListener('resize', this.setCanvasSize)
  },
  methods: {
    setCanvasSize(){
      let canvas = HTMLCanvasElement
      let source = HTMLMediaElement
      if(this.video){
        canvas = this.$refs.canvasOutput
        source = this.$refs.videoInput
      } else {
        canvas = this.$refs.photoCanvas
        source = this.$refs.inputImg
      }

      let xPos = source.getBoundingClientRect().left
      let yPos = source.getBoundingClientRect().top
      canvas.style.left = xPos + 'px'
      canvas.style.top = yPos + 'px'
      canvas.width = source.getBoundingClientRect().width
      canvas.height = source.getBoundingClientRect().height
    },
    loadImageFromUpload(){
      if(this.uploadedPhoto !== [] && this.detection) {
        setTimeout(() => {
          this.$refs.inputImg.src = URL.createObjectURL(this.uploadedPhoto[0])
          this.showPhoto = true
          setTimeout(() => {
            this.detectEmotions()
          }, 100)
        }, 1)
      } else if (this.uploadedTransformPhoto !== []) {
        let canvasArray = [
          this.$refs.transformCanvas1,
          this.$refs.transformCanvas2,
          this.$refs.transformCanvas3,
          this.$refs.transformCanvas4
        ]
        canvasArray.forEach(canvas => {
          if(canvas){
            const context = canvas.getContext('2d');
            context.clearRect(0, 0, canvas.width, canvas.height);
          }
        })
        console.log('upload started')
        setTimeout(() => {
          this.$refs.inputTransformImg.src = URL.createObjectURL(this.uploadedTransformPhoto[0])
          this.showTransformPhoto = true
        }, 1)
      }
    },
    drawFaces(canvas, data, emotion){
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      // draw title
      ctx.font = 'small-caps 28px "Roboto"'
      if(data.length === 0){
        ctx.fillText('No faces detected. Try a photo with larger faces', canvas.width/4, canvas.height/2)
      }
      data.forEach((person, i) => {
        // draw box around each face
        ctx.lineWidth = 3
        ctx.strokeStyle = 'deepskyblue'
        ctx.fillStyle = 'deepskyblue'
        ctx.globalAlpha = 0.8
        ctx.beginPath()
        ctx.rect(person.box.x, person.box.y, person.box.width, person.box.height)
        ctx.stroke()
        ctx.globalAlpha = 1

        ctx.fillStyle = 'white'
        ctx.textAlign = 'center'
        ctx.fillText(`${emotion[i]}`, (person.box.bottomLeft.x + person.box.bottomRight.x) / 2, person.box.bottom + 20)
      })
    },
    async startWebcam(){
      await this.$faceapi.nets.tinyFaceDetector.loadFromUri('/models')
      await this.detectEmotions()
      this.$refs.videoInput.addEventListener('resize', this.setCanvasSize)
    },
    async generateFace(i) {
      this.loading = true
      const tf = this.$tf
      const model = await tf.loadGraphModel('/models/conditional_gan/model.json')
      let label_array = [0, 1, 2, 3, 4, 5, 6, 7]
      let label = tf.oneHot(tf.tensor1d([label_array[i]], 'int32'), 8)
      let noise = tf.randomNormal([1, 128])
      let noiseAndLabel = tf.concat([noise, label], 1)
      let generatedImage = model.predict(noiseAndLabel)
      let reshapedImage = generatedImage.reshape([64, 64, 3])
      // let normalized = tf.add(reshapedImage, tf.scalar(1))
      // let furtherNormalized = tf.div(normalized, tf.scalar(2))
      let resizedImage = tf.image.resizeBilinear(reshapedImage, [200, 200])
      let canvas = this.$refs.generateCanvas
      tf.browser.toPixels(resizedImage, canvas)
      this.loading = false
      noise.dispose()
      noiseAndLabel.dispose()
      generatedImage.dispose()
      reshapedImage.dispose()
      resizedImage.dispose()
    },
    async detectEmotions() {
      this.loading = true
      let tensorSource = HTMLMediaElement
      let canvas = HTMLCanvasElement
      const video = this.$refs.videoInput
      if(this.video){
        this.streaming = !this.streaming
        const video = this.$refs.videoInput
        canvas = this.$refs.canvasOutput
        tensorSource = this.$refs.videoInput

        navigator.mediaDevices.getUserMedia({video: true, audio: false})
            .then(function (stream) {
              if(!vm.streaming){
                video.srcObject = null
                stream.getTracks().forEach(track => {
                  track.stop()
                })
              } else {
                video.srcObject = stream;
                video.play()
              }
            })
            .catch(function (err) {
              console.log("An error occurred! " + err)
            });
        console.log('webcam started')
      } else {
        canvas = this.$refs.photoCanvas
        tensorSource = this.$refs.inputImg
      }

      const tf = this.$tf
      let vm = this
      const faceapi = this.$faceapi

      const model = await tf.loadGraphModel('/models/recognizer/model.json')

      const minConfidenceFace = 0.5

      let faceapiOptions
      if(vm.video){
        faceapiOptions = new faceapi.TinyFaceDetectorOptions({
          minConfidenceFace
        })
      } else {
        faceapiOptions = new faceapi.SsdMobilenetv1Options({
          minConfidenceFace
        })
      }

      let emotion_dict = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Neutral',
        5: 'Sadness',
        6: 'Surprise'
      }
      let forwardTimes = []

      function updateTimeStats(timeInMs) {
        forwardTimes = [timeInMs].concat(forwardTimes).slice(0, 30)
        const avgTimeInMs = forwardTimes.reduce((total, t) => total + t) / forwardTimes.length
        vm.time = `${Math.round(avgTimeInMs)} ms`
        vm.fps = `${faceapi.utils.round(1000 / avgTimeInMs)}`
      }
      this.setCanvasSize()

      async function onPlay() {
        const ts = Date.now()
        vm.tensors = tf.engine().memory().numTensors

        const mediaTensor = await tf.browser.fromPixels(tensorSource)
        const detections = await faceapi.detectAllFaces(mediaTensor, faceapiOptions)

        await faceapi.extractFaceTensors(mediaTensor, detections)
            .then(faceImages => {
              vm.loading = false
              var emotions = []
              for (const faceImage of faceImages) {
                const scaledImage = tf.tidy(() => {
                  const scalar = tf.scalar(255)
                  const result = tf.image.resizeBilinear(faceImage, [128, 128])
                  const expandResult = result.expandDims(0)
                  const returnedResult = expandResult.div(scalar)
                  scalar.dispose()
                  expandResult.dispose()
                  result.dispose()
                  return returnedResult
                })
                const prediction = model.predict(scaledImage)
                const emotionResult = tf.unstack(prediction)[0]
                const topEmotion = tf.argMax(emotionResult)
                emotions.push(emotion_dict[topEmotion.arraySync()])
                emotionResult.dispose()
                prediction.dispose()
                topEmotion.dispose()
                faceImage.dispose()
                scaledImage.dispose()
              }
              vm.drawFaces(canvas, detections, emotions)
            })
        mediaTensor.dispose()

        updateTimeStats(Date.now() - ts)

        if (vm.streaming) {
          setTimeout(() => onPlay())
        }
      }

      // schedule the first one.
      if(this.video){
        video.addEventListener('loadeddata', async function () {
          await onPlay()
        }, false)
      } else {
        await onPlay()
      }
    },
    async changeEmotion(i){
      this.loading = true
      let tensorSource = this.$refs.inputTransformImg

      const tf = this.$tf
      let vm = this
      const faceapi = this.$faceapi
      const weightFiles = [
          '/models/cyclegan/happy-anger/model.json',
          '/models/cyclegan/happy-disgust/model.json',
          '/models/cyclegan/happy-fear/model.json',
          '/models/cyclegan/happy-neutral/model.json',
          '/models/cyclegan/happy-sadness/model.json',
          '/models/cyclegan/happy-surprise/model.json'
      ]

      const gan = await tf.loadGraphModel(weightFiles[i])

      const minConfidenceFace = 0.5
      let faceapiOptions = new faceapi.SsdMobilenetv1Options({
        minConfidenceFace
      })
      let canvasArray = [
        this.$refs.transformCanvas1,
        this.$refs.transformCanvas2,
        this.$refs.transformCanvas3,
        this.$refs.transformCanvas4
      ]

      const mediaTensor = await tf.browser.fromPixels(tensorSource)
      const detections = await faceapi.detectAllFaces(mediaTensor, faceapiOptions)

      await faceapi.extractFaceTensors(mediaTensor, detections)
          .then(faceImages => {
            vm.loading = false
            faceImages.forEach((faceImage, i) => {
              const scaledImage = tf.tidy(() => {
                const scalar = tf.scalar(127.5)
                const result = tf.image.resizeBilinear(faceImage, [128, 128])
                const expandResult = result.expandDims(0)
                const returnedResult = expandResult.div(scalar)
                const refinedResult = tf.sub(returnedResult, tf.scalar(1))
                scalar.dispose()
                expandResult.dispose()
                result.dispose()
                returnedResult.dispose()
                return refinedResult
              })

              const faceHeight = detections[i].box.height
              const faceWidth = detections[i].box.width

              const facePositionX = detections[i].box.x
              const facePositionY = detections[i].box.y

              let xPos = tensorSource.getBoundingClientRect().left
              let yPos = tensorSource.getBoundingClientRect().top
              let canvas = canvasArray[i]
              canvas.style.left = (xPos + facePositionX) + 'px'
              canvas.style.top = (yPos + facePositionY) + 'px'
              canvas.height = faceHeight
              canvas.width = faceWidth

              const prediction = gan.predict(scaledImage)
              const reshapedImage = prediction.reshape([128, 128, 3])
              const normalized = tf.add(reshapedImage, tf.scalar(1))
              const furtherNormalized = tf.div(normalized, tf.scalar(2))
              const resizedImage = tf.image.resizeBilinear(furtherNormalized, [Math.round(faceHeight), Math.round(faceWidth)])

              tf.browser.toPixels(resizedImage, canvas)
              vm.loading = false

              prediction.dispose()
              faceImage.dispose()
              scaledImage.dispose()
              reshapedImage.dispose()
              resizedImage.dispose()
              normalized.dispose()
              furtherNormalized.dispose()
            })
          })
      mediaTensor.dispose()
      gan.dispose()
    },
  }
}
</script>

<style scoped>

.vid {
  border: 1px solid black;
  margin: auto;
  max-height: 100%;
  width: 100%;
}
@media (min-width: 768px) {
  .vid {
    height: 480px;
    width: 640px;
  }
}
#canvasOutput {
  position: absolute;
}
#photoCanvas {
  position: absolute;
  z-index: 1999;
}
#transformCanvas1, #transformCanvas2, #transformCanvas3, #transformCanvas4 {
  position: absolute;
}
.overlay-spinner {
  position:fixed;
  width: 100%;
  height: 100%;
  top:0;
  left:0;
  right:0;
  bottom:0;
  background-color:rgba(0, 0, 0, 0.35);
  z-index:9998;
  color:white;
}
.spin {
  width: 50px;
  height: 50px;
  position: absolute;
  top: 25%;
  left: 50%;
  margin-left: -25px;
  margin-top: -25px;
  z-index: 9999;
}
</style>