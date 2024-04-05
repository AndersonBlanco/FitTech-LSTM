const express = require("express");

var app = express()
app.use(express.static('build'))
const port = 3000; 

/*
app.get("/",(req, res) =>{
    res.sendFile("../server/build/index.html", )
})

/**/
//app.set("Content-Disposition", "inline")
 app.get("/ml",(req, res) =>{
     res.setHeader("Content-Disposition", "inline")
    res.sendFile("/workspaces/FitTech-LSTM/server/py_files/main.cgi");
    
})

app.listen(port,(e)=>console.log(`Server listening on prt${port}`)) 