import "../App.css"; 
import React, {useState} from "react";
import { MotionAnimate } from 'react-motion-animate'; 
import instagram_img from "../pages/media/imgs/instagram.png";
import x_twitter_img from "../pages/media/imgs/x_twitter.png";

import facebook_img from "../pages/media/imgs/facebook.png";
export default function Footer(){
    const leftList= (
        <div className = "column" id = "stay_updated">
            <h1>Stay Updated</h1>
            
            <ul id = "ul">
                <li>@fittech - instagram <img src = {instagram_img} /></li>
                <li>@fittech <img src = {x_twitter_img} /> </li>
                <li>@fittech - facebook <img src = {facebook_img} /></li>
            </ul>
        </div>
    );

    const middleList= (
        <div className = "column" id = "middle_list">
            <ul id = "ul">
                <li>user privacy</li>
                <li>terms of use</li>
                <li>knowledge bowl</li>
            </ul>
        </div>
    );


    
    const rightList= (
        <div className = "column" id = "discover_more_abt_us">
            <h1>Discover more about us</h1>
            
            <ul id = "ul">
                <li>our mission</li>
                <li>founders</li>
                <li>advertise with us</li>
            </ul>
        </div>
    );

 
    return(
        <div id = "footer">
        <div className="row">
        {leftList}
        {middleList}
        {rightList}
        </div>
            <h1 id = "copyright_sent">@copyright fittech 2024</h1>
        </div>
    )
}