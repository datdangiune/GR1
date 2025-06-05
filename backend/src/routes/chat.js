const express = require('express');
const router = express.Router();
const ChatController = require('../controllers/ChatController');

// Route nhận câu hỏi và trả về kết quả đã qua kiểm duyệt Gemini
router.post('/chat', ChatController.chat);

module.exports = router;
