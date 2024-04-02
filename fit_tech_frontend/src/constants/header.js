import React, {useState} from "react";
import "../App.css"; 
import { get_item } from "../tools/localStorage";
import CloseIcon from "../pages/media/svgs/close_icon.js"; 
import { Link,  } from "react-router-dom"; 

export default function Header(){
     //onscroll windows event

   const [universal_scroll_pos, set_universal_scroll_pos] = useState(window.scrollY/window.innerHeight);
   window.onscroll = () =>{
    set_universal_scroll_pos(window.scrollY/window.innerHeight);
    console.log(universal_scroll_pos)
   }

   
   const [header_Text_Color, set_header_Text_Color] = useState(0.82 < universal_scroll_pos < 1.44? "red" : "blue");  
   const [toggle, setToggle] = useState(false); 
   let start = 1.02, end = 3.97; //was 4.08

   const color_theme_limits = (start, end, c1, c2) =>{
    return (start <= universal_scroll_pos && universal_scroll_pos <= end? c1 : c2)
  }
   const NavButton = () =>{
     let dot = <div className = "dot" style = {{backgroundColor: color_theme_limits(start, end, "white", "black")}}></div>;
     return(
 
       <div id = "nav_button" onClick = {() => setToggle(true)}>
       {dot}
       {dot}
       {dot}
       {dot}
       {dot}
       {dot}
       {dot}
       {dot}
       {dot}
       </div>
   
     )
    }

  //style = {{color: get_item("CURRENT_PAGE")=="learn"? "black" : "white"}}

  const AsideNav = () =>{
    return(
      <div id = "opaque_overlay" style = {{display: toggle? "absolute" : "none"}}>
        <div id = "aside_nav">
        <CloseIcon 
        onClick = {() => {setToggle(false)}}
        height = {"1.5rem"} 
        id = "svg_close_icon" 
        style = {{
          transform: "rotateZ(180deg)",
          position: "relative",
          left: "-3.5rem",
          top: "1.5rem",
          cursor: "url('./media/imgs/boxing_glove.png'),pointer"
        }}
     
        />
        <ul>         
          <li><Link to="/">Home</Link></li>
          <li><a href = "#test_out">Try Out Our Prpoduct</a></li>
          <li><Link to="/product">Knowledge</Link></li>
          <li><Link to="/product">Guded Workout</Link></li>
        </ul>
      </div>
      </div>

    )
  }
    return (
        <div id = "header" >            
        <AsideNav/>
            <NavButton/>
            <h1 id = "header_title" style = {{color: color_theme_limits(start, end, "white","black") }}>FitTech</h1>
            <button id = "auth_in" style = {{color: color_theme_limits(start, end, "black", "white"), backgroundColor: color_theme_limits(start, end, "white", "black") }}>signup/login</button>

        </div>
    )
}