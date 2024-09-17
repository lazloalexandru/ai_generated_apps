const express = require('express');
const { OpenAI } = require('openai');
const path = require('path');
const cors = require('cors'); // Import cors
require('dotenv').config();

const app = express();
const port = 3000;

// Set up OpenAI API client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Middleware to parse JSON requests
app.use(express.json());
app.use(cors()); // Enable CORS for all routes

// Serve static files (your frontend files)
app.use(express.static(path.join(__dirname, '/')));

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
