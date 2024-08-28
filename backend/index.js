const express = require('express');
const multer = require('multer');
const tensorflow = require('@tensorflow/tfjs-node');

const app = express();
const upload = multer({ dest: './uploads/' });

app.post('/predict', upload.single('image'), (req, res) => {
  const image = req.file;

  // Load the model
  const model = tensorflow.loadLayersModel('/python/train_model.py');

  // Preprocess the image
  const img = tensorflow.tensor3d(image.buffer, [image.height, image.width, 3]);
  img = img.resizeNearestNeighbor([256, 256]);

  // Make a prediction
  const prediction = model.predict(img);
  const disease = prediction.argMax(-1).dataSync()[0];

  res.json({ disease });
});

app.listen(3000, () => {
  console.log('Server listening on port 3000');
});
