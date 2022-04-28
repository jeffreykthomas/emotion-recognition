import 'mdb-vue-ui-kit/css/mdb.min.css'
import { createApp } from 'vue'
import App from './App.vue'
import * as faceapi from '@vladmandic/face-api'

const app = createApp(App)
app.config.globalProperties.$tf = faceapi.tf
app.config.globalProperties.$faceapi = faceapi

app.mount('#app')
