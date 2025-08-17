// temp_projects/test123/server.js
const express = require("express");
const app = express();
app.get("/", (req, res) => res.send("Preview running"));
app.listen(4000, () => console.log("Server on port 4000"));