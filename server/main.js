const express = require("express");
const {exec} = require("child_process");

exec('python test.py Ander', (err, stdout, stderr)=>{
console.log('stdout: ' + stdout)
console.log(stderr)
});

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
     //res.setHeader("Content-Disposition", "inline")
    res.sendFile("build");
})

app.listen(port,(e)=>console.log(`Server listening on port: ${port}`)) 