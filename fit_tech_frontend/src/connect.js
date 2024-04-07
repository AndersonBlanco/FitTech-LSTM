const {exec} = require("child_process");

function connect(){
    exec('python test.py Ander', (err, stdout, stderr)=>{
console.log('stdout: ' + stdout)
console.log(stderr)
//return stdout
});

}

export {connect}; 