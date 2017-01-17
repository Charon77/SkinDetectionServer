var express = require('express')
var app = express()
var multer  = require('multer')
var exec = require('child_process').exec;

getTimeMSFloat = () => {
    const hrtime = process.hrtime();
    return ( hrtime[0] * 1000000 + hrtime[1] / 1000 );
}

var storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    cb(null, getTimeMSFloat() + '.jpg')
  }
})


var upload = multer({ storage: storage })

app.post('/upload', upload.single('picture'), function (req, res, next) {
  console.log("File uploaded: ", req.file)
  res.send(req.file.path +".txt")
  exec( '(cd ..; python3 skin.py skindetectserver/' + req.file.path + ')' )
  console.log("Running:",  '(cd ..; python3 skin.py skindetectserver/' + req.file.path + ')')
})

app.use('/uploads', express.static('uploads'))

app.listen(80, function () {
  console.log('Example app listening on port 3000!')
})

