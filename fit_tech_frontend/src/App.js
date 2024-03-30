import logo from './logo.svg';
import './App.css';
import {useState, useEffect, useRef} from "react"; 
import WebCam from "react-webcam"; 

import Header from './constants/header';
import Product from "./product"; 

import header_img from "./media/imgs/header_img.png"
//pg1:
const Motive_words_latter = () =>{
  return(
    <div class = "row" id = "motive_words">
      <h1><a href = "#pg2">Develop</a></h1>
      <h1><a href = "#pg2">Apply</a></h1>
      <h1><a href = "#pg2">Learn</a></h1>
    </div>
  )
}
function App() {
  return (
    <div className="App">
      <Header/>
      <div id = "pg1">
        <div class = 'column'> 
        <div class = "row" id = "motive_sentence_and_header_img">
        <img src = {header_img} id = "header_img" />
        <h1 id = "motive_sentence">learn & get fit with tech</h1>
      </div>
      <Motive_words_latter/>
      </div>
      </div>

      <div id = "pg2">
     <h1>pg2</h1>
      </div>

    </div>
  );
}

export default App;
