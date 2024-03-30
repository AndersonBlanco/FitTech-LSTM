let time = document.getElementById("time");
let debug_text = document.getElementById("debug_text");

let count = 0; 
setInterval(() =>{
    debug_text.innerText=`${debug_text.innerText} On`
}, 1000)
 