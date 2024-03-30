import React, {useState} from "react";
import "../App.css"; 

export default function Header(){
   const Nav_Button = () =>{
     let dot = <div className = "dot"></div>;
     let arr = new Array(9).fill(0); 
     return(
       <div id = "nav_button">
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
  
    return(
        <div id = "header">
            <Nav_Button/>
            <h1 id = "header_title">FitTech</h1>
            <button id = "auth_in">signup/login</button>
        </div>
    )
}