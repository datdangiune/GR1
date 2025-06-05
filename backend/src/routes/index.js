const express = require('express');
const router = express.Router();
const chatRoutes = require('./chat');

router.use('/', chatRoutes);

module.exports = router;