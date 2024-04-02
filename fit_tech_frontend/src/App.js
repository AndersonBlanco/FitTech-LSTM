import logo from './logo.svg';
import './App.css';
import {useState, useEffect, useRef} from "react"; 
import WebCam from "react-webcam"; 
import Header from './constants/header';
import Product from "./product"; 
import { useInView } from 'react-intersection-observer';

import header_img from "./media/imgs/header_img.png"

import {set_item, get_item} from "./tools/localStorage";

import try_out_img1 from "./media/imgs/try_out_img1.png"; 
import try_out_img2 from "./media/imgs/try_out_img2.png"; 
import try_out_img3 from "./media/imgs/try_out_img3.png"; 
import { MotionAnimate } from 'react-motion-animate'; 

import good_jab from "./media/imgs/good_jab.png";
import bad_rest from "./media/imgs/bad_rest.png";
import good_uppercut from "./media/imgs/good_uppercut.png";

//import {motion} from "framer-motion" //excelent animation library as well. 
//pg1:

import Footer from "./constants/footer"; 

const Motive_words_latter = () =>{

  return(
    <div className = "row" id = "motive_words">
         <h1><a onClick={function(){set_item("CURRENT_PAGE", "learn")}} href = "#pg2">Learn</a></h1>
         <h1><a onClick={function(){set_item("CURRENT_PAGE", "learn")}} href = "#pg3">Apply</a></h1>
      <h1><a onClick={function(){set_item("CURRENT_PAGE", "learn")}} href = "#pg4">Develop</a></h1>
     
   
    </div>
  )
}


function App() { 
 //set_item("CURRENT_PAGE", "landing")
 

  return (
    <div className="App">
    <Header/>
      <div id = "pg1">
        <div className = 'column'> 
        <div className = "row" id = "motive_sentence_and_header_img">
        <img src = {header_img} id = "header_img" />
        <h1 id = "motive_sentence">learn & get fit with tech</h1>
      </div>
      <Motive_words_latter/>
      </div>
      </div>

      <div className='pg' id = "pg2">
        <div className = "row" id = "learn_row">
        <div className = "pg_img_text_column" id ="learn_img_text_column">
       <img src = {try_out_img1} id = "learn_img"/>
       <h1 className = "pg_text_title" id = "learn_title">Learn</h1>
      </div>
      <h2 className = "pg_text" id ="learn_text">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut vitae posuere lectus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Aenean venenatis, ipsum id maximus luctus, orci lorem vulputate tellus, at venenatis sem felis tincidunt massa. Donec hendrerit sem ut est auctor vehicula. Nam in mattis sem. Cras vitae nisl tincidunt, commodo mi a, euismod ex. Duis rutrum suscipit nisi, ut molestie ligula aliquet id. Proin ullamcorper blandit dolor, at luctus dui iaculis non. Fusce viverra pulvinar aliquet. Aenean tincidunt tellus vel aliquet sollicitudin. Curabitur in varius ex. Curabitur vel risus porttitor, congue sem at, lacinia lectus. Vivamus.</h2>
      </div>
      </div>

      <div className = "pg" id = "pg3">
        <div className = "row" id = "apply_row">
        <div className = "pg_img_text_column" id ="pg3_img_text_column">
       <img src = {try_out_img1} id = "apply_img"/>
       <h1 className = "pg_text_title" id = "">Apply</h1>
      </div>
      <h2 className = "pg_text" id ="apply_text">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut vitae posuere lectus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Aenean venenatis, ipsum id maximus luctus, orci lorem vulputate tellus, at venenatis sem felis tincidunt massa. Donec hendrerit sem ut est auctor vehicula. Nam in mattis sem. Cras vitae nisl tincidunt, commodo mi a, euismod ex. Duis rutrum suscipit nisi, ut molestie ligula aliquet id. Proin ullamcorper blandit dolor, at luctus dui iaculis non. Fusce viverra pulvinar aliquet. Aenean tincidunt tellus vel aliquet sollicitudin. Curabitur in varius ex. Curabitur vel risus porttitor, congue sem at, lacinia lectus. Vivamus.</h2>
      </div>
      </div>


      <div className = "pg" id = "pg4">
        <div className = "row" id = "develop_img_row">
       <img src = {bad_rest} id = "develop_img"/>
       <img src = {good_jab} id = "develop_img"/>
       <img src = {good_uppercut} id = "develop_img"/>
      </div>

<div className='column' id = "develop_title_text_column">
            <h1 className = "pg_text_title" id = "develop_title">Develop</h1>
      <h2 className = "" id ="develop_text">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut vitae posuere lectus. Interdum et malesuada fames ac ante ipsum primis in faucibus. Aenean venenatis, ipsum id maximus luctus, orci lorem vulputate tellus, at venenatis sem felis tincidunt massa. Donec hendrerit sem ut est auctor vehicula. Nam in mattis sem. Cras vitae nisl tincidunt, commodo mi a, euismod ex. Duis rutrum suscipit nisi, ut molestie ligula aliquet id. Proin ullamcorper blandit dolor, at luctus dui iaculis non. Fusce viverra pulvinar aliquet. Aenean tincidunt tellus vel aliquet sollicitudin. Curabitur in varius ex. Curabitur vel risus porttitor, congue sem at, lacinia lectus. Vivamus.</h2>
    </div>
      </div>


      <div id = "test_out">    
      <div className = "row" id = "img_row">
        <MotionAnimate speed={.3} ease={"easeInOut"} delay={5} animation={"scrollFadeIn"}> <img src = {try_out_img2} /> </MotionAnimate> 
        <MotionAnimate speed={.5} ease={"easeInOut"} delay = {7} animation={"scrollFadeIn"}> <img src = {try_out_img3} /></MotionAnimate>
        <MotionAnimate speed={.7} ease={"easeInOut"} delqy={5} animation={"scrollFadeIn"}>  <img src = {try_out_img1} /></MotionAnimate>

     
      
        
    </div>

    <MotionAnimate ease={"easeInOut"} animation={"fade"}>
          <button id = "try_out_button">Test It Out!</button>
    </MotionAnimate>
    
        </div>

  
        <Footer/>
    </div>
  );
}

export default App;
