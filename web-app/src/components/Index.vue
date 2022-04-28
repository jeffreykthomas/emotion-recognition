<template>
  <div class="hello">
    <MDBContainer class="px-0 px-md-3">
      <MDBRow center v-show="!detectImage && !generateImage && !transformImage">
        <MDBCol md="4">
          <h1>Welcome to the <br>Emotion Detection App</h1>
        </MDBCol>
      </MDBRow>
      <MDBRow center>
        <MDBCol lg="4" class="text-center" v-show="!detectImage && !generateImage  && !transformImage">
          <MDBBtn class="d-block mx-auto mt-3" rounded @click="detectionClick">Emotion Detection</MDBBtn>
          <MDBBtn class="d-block mx-auto my-3" rounded @click="generateClick">Generate A Face</MDBBtn>
          <MDBBtn class="d-block mx-auto my-3" rounded @click="transformClick">Transform A Face</MDBBtn>
        </MDBCol>
      </MDBRow>
      <MDBRow center class="pt-5" v-show="detectImage || generateImage || transformImage">
        <MDBTabs v-model="activeTabId" v-on:hide="hideTab" v-on:show="showTab">
          <!-- Tabs navs -->
          <MDBTabNav fill tabsClasses="mb-3">
            <MDBTabItem tabId="detection" href="detection">Emotion Detection</MDBTabItem>
            <MDBTabItem tabId="generate" href="generate">Generate A Face</MDBTabItem>
            <MDBTabItem tabId="transform" href="transform">Transform A Face</MDBTabItem>
          </MDBTabNav>
          <!-- Tabs navs -->
          <!-- Tabs content -->
          <MDBTabContent>
            <MDBTabPane tabId="detection"><EmotionRecognition :detection=true v-if="detectImage"></EmotionRecognition></MDBTabPane>
            <MDBTabPane tabId="generate"><EmotionRecognition :generate=true v-if="generateImage"></EmotionRecognition></MDBTabPane>
            <MDBTabPane tabId="transform"><EmotionRecognition :transform=true v-if="transformImage"></EmotionRecognition></MDBTabPane>
          </MDBTabContent>
          <!-- Tabs content -->
        </MDBTabs>
      </MDBRow>
      <MDBRow>
        <MDBCol>
          <p>Disclaimer: No data is recorded or saved</p>
        </MDBCol>
      </MDBRow>
    </MDBContainer>
  </div>
</template>

<script>
import {
  MDBContainer,
  MDBRow,
  MDBCol,
  MDBBtn,
  MDBTabs,
  MDBTabNav,
  MDBTabContent,
  MDBTabItem,
  MDBTabPane } from 'mdb-vue-ui-kit'
import EmotionRecognition from "@/components/EmotionRecognition";
import { ref } from 'vue';
export default {
  name: 'RootIndex',
  components: {
    MDBContainer,
    MDBRow,
    MDBCol,
    MDBBtn,
    EmotionRecognition,
    MDBTabs,
    MDBTabNav,
    MDBTabContent,
    MDBTabItem,
    MDBTabPane
  },
  data(){
    return {
      detectImage: false,
      generateImage: false,
      transformImage: false
    }
  },
  setup() {
    const activeTabId = ref('ex1-1');
    return {
      activeTabId,
    };
  },
  methods: {
    detectionClick(){
      this.activeTabId = 'detection'
      this.useCamera = true
    },
    generateClick(){
      this.activeTabId = 'generate'
      this.generateImage = true
    },
    transformClick(){
      this.activeTabId = 'transform'
      this.transformImage = true
    },
    hideTab(event){
      switch (event.target.id) {
        case 'tab-detection':
          this.detectImage = false
          break
        case 'tab-generate':
          this.generateImage = false
          break
        case 'tab-transform':
          this.transformImage = false
      }
    },
    showTab(event){
      switch (event.target.id) {
        case 'tab-detection':
          this.detectImage = true
          break
        case 'tab-generate':
          this.generateImage = true
          break
        case 'tab-transform':
          this.transformImage = true
      }
    }
  },
  created(){

  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
