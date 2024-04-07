import logo from './logo.svg';
import './App.css';
import React, {useState, useEffect, useRef} from "react"; 
import WebCam from "react-webcam"; 
import Header from './constants/header';
import { BrowserRouter, Routes, Route } from "react-router-dom"; 
import Footer from "./constants/footer"; 
import Main from './pages/main';
import Product from "./pages/product";
import NoPageFound from './pages/NoPageFound';
 

//import * as mediapipe from "@mediapipe/holistic";


function App() { 
 
  

  //set_item("CURRENT_PAGE", "landing")
 const [CURRENTPage, setCurrentPage] = useState(Main); 
 /*
 <div className="App">
 <Header/>
<Main/>
     <Footer/>
 </div>
/**/
//GT request to python test flask file 
 
  return (
     <div className='App'>
    <BrowserRouter> 
    <Header/>
                <Routes> 
                    <Route exact path="/" element={<Main />} /> 
                    <Route exact path="/product" element={<Product />} /> 
                    <Route path="*" element={<NoPageFound />} /> 
                </Routes> 
                <Footer/>
            </BrowserRouter> 
     </div>
  );
}

export default App;
