import './App.css';
import {useState, useEffect, useRef} from "react"; 
import WebCam from "react-webcam"; 

function App() {
    const webCam_ref = useRef(null); 
  
    const [time, setTime] = useState(0),
          [_break, set_break] = useState(time == 5? true : false),
          [debugtext, steDebugText] = useState(`${_break}`),
          [img_rgb_array, set_img_rgb_array] = useState([]),
          [img_src, set_img_src] = useState(" "),
          [imgData, set_imgData] = useState(" "); 
  
  const delay = (s) => new Promise((resolve) => setTimeout(resolve, s)); 
  
  const Canvas = () =>{
    const canvas_ref = useRef(null);
  
    useEffect(() =>{
    const canvas = canvas_ref.current; 
    const ctx = canvas.getContext("2d", {willReadFrequently: true}); 
  
    const img = new Image();
    img.src = img_src; 
    img.onload = () =>{
      ctx.drawImage(img, 0,0, canvas.width, canvas.height); 
    };
   let ctx_data = ctx.getImageData(0, 0, 340, 250 )
  //window.localStorage.setItem("imgData", (ctx_data));
  set_imgData(ctx_data.data); 
   //console.log(ctx_data)
   //console.log(img_rgb_array.length)
    }, [imgData])
  
    return <canvas ref = {canvas_ref} width={340} height = {250} />
  }
  
  useEffect(() =>{  
    var storeFrame = setInterval(async () =>{
    if(time == 40){
      await delay(500) //delay 5s
      setTime(0); //reset timer
    }else{
      setTime(time+1);
      //console.log("Stored frame")
      let _base64 = webCam_ref.current.getScreenshot(); 
      set_img_src(_base64); 
      //console.log(window.localStorage.getItem("imgData"));
      set_img_rgb_array([...img_rgb_array, imgData])
  
      console.log(img_rgb_array)
    }
  }, 1000)//every 1s 
  
  
  return () =>{
    clearInterval(storeFrame); 
  }
   
  }, [time, img_rgb_array, img_src, webCam_ref]);
  
   
  
  
   
    return (
      <div className="App">
      
        <h2>DebugText: {debugtext}</h2>
        <h2>Time: {time}</h2>
  
        <WebCam
        ref = {webCam_ref}
        height={500}
        width={500}
        mirrored
        />
  
     <div style = {{opacity:0}}><Canvas /></div>
        
  
        
      </div>
    );
  }
  
  export default App;
  