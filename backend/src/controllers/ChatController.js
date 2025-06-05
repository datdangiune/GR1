const ChatInteraction = require('../models').ChatInteraction;
const Replicate = require("replicate");
const axios = require("axios");

const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

// Hàm gọi Gemini API (Google Gemini Pro)
async function callGeminiAPI(question, modelAnswer) {
    const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
    const url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY;

    // English, detailed, professional prompt for Gemini
    const prompt = `
You are an expert medical AI assistant responsible for reviewing and refining chatbot answers to user medical questions.
Please carefully check the following chatbot answer for accuracy, clarity, completeness, and safety.
If the answer is already appropriate, simply return it as is.
If the answer is incomplete, unclear, inaccurate, or could be improved, please revise, expand, or clarify it to ensure it is medically sound, clear, and helpful for the user.
Always provide your final answer in English, and make sure it is professional, easy to understand, and safe for the user.

---
User question: ${question}
Chatbot's answer: ${modelAnswer}
---
Final answer (in English, revise if needed, otherwise keep as is):`;

    const body = {
        contents: [
            {
                parts: [
                    { text: prompt }
                ]
            }
        ]
    };

    const response = await axios.post(url, body, {
        headers: { "Content-Type": "application/json" }
    });

    // Gemini trả về ở response.data.candidates[0].content.parts[0].text
    return response.data?.candidates?.[0]?.content?.parts?.[0]?.text || modelAnswer;
}

const ChatController = {
    async chat(req, res) {
        try {
            const { user_question, session_identifier } = req.body;
            if (!user_question) {
                return res.status(400).json({ success: false, message: "Missing user_question" });
            }

            // 1. Gửi tới model của bạn trên Replicate
            const startTimePrimary = Date.now();
            const output = await replicate.run(
                "datdangiune/medical_chatbot2:0705591af1520d60399d8d401bdd71a67acbc91c4d69bc29a0b142cda7dcb143",
                { input: { prompt: user_question } }
            );
            const processing_time_ms_primary = Date.now() - startTimePrimary;

            // output có thể là string hoặc object, tuỳ model
            const primary_model_response = typeof output === "string" ? output : (output.answer || JSON.stringify(output));

            // 2. Gửi qua Gemini để kiểm duyệt/tinh chỉnh
            const startTimeGemini = Date.now();
            const final_bot_response = await callGeminiAPI(user_question, primary_model_response);
            const processing_time_ms_gemini = Date.now() - startTimeGemini;

            // 3. Xác định Gemini có chỉnh sửa không
            const was_refined_by_gemini = (final_bot_response.trim() !== primary_model_response.trim());

            // 4. Lưu vào DB
            await ChatInteraction.create({
                session_identifier,
                timestamp_start: new Date(),
                user_question,
                primary_model_response,
                final_bot_response,
                was_refined_by_gemini,
                model_version_used: "datdangiune/medical_chatbot2:0705591af1520d60399d8d401bdd71a67acbc91c4d69bc29a0b142cda7dcb143",
                gemini_model_version_used: "gemini-2.0-flash",
                processing_time_ms_primary,
                processing_time_ms_gemini
            });

            // 5. Trả về cho user
            res.status(200).json({
                success: true,
                data: {
                    user_question,
                    primary_model_response,
                    final_bot_response,
                    was_refined_by_gemini
                }
            });
        } catch (err) {
            res.status(500).json({
                success: false,
                message: "Error processing chat",
                error: err.message
            });
        }
    }
};

module.exports = ChatController;