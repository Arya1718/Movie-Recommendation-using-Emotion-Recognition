const express = require('express');
const app = express();
const mongoose = require('mongoose');
const favModel = require('./favModel');

const bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static('public'));

mongoose.connect('mongodb://0.0.0.0/favList', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const db = mongoose.connection;

db.on('error', console.error.bind(console, 'MongoDB connection error:'));


app.set('view engine', 'ejs');
app.set('views', __dirname + '/views');

app.get('/predict', (req, res) => {
  favModel.find()
  .then((data) => {
    console.log(data);
    res.render(data);
    // res.render('index', { data: data });
  })
  .catch((err) => {
    console.error(err);
    res.status(500).send('Error retrieving data from MongoDB');
  });
  
});

app.post('/delete', (req, res) => {

    const deleteItemID = req.body.checkbox;
    const name = req.body.list;
  
    console.log(deleteItemID);
  
    favModel.findByIdAndRemove(deleteItemID)
        .then((deletedItem) => {
          console.log('Deleted item:', deletedItem);
          res.redirect('/')
        })
        .catch((err) => {
          console.error('Error while deleting item:', err);
        });
    
    
  
  })

app.listen(4000, () => {
  console.log('Server is running on http://localhost:4000');
});
