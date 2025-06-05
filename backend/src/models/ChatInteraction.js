const { DataTypes } = require('sequelize');
const sequelize = require('../database');

const ChatInteraction = sequelize.define('ChatInteraction', {
    id: {
        type: DataTypes.UUID,
        primaryKey: true,
        defaultValue: DataTypes.UUIDV4,
    },
    session_identifier: {
        type: DataTypes.STRING(255),
        allowNull: true,
    },
    timestamp_start: {
        type: DataTypes.DATE,
        allowNull: false,
        defaultValue: DataTypes.NOW,
    },
    user_question: {
        type: DataTypes.TEXT,
        allowNull: false,
    },
    primary_model_response: {
        type: DataTypes.TEXT,
        allowNull: true,
    },
    final_bot_response: {
        type: DataTypes.TEXT,
        allowNull: true,
    },
    was_refined_by_gemini: {
        type: DataTypes.BOOLEAN,
        allowNull: false,
        defaultValue: false,
    },
    model_version_used: {
        type: DataTypes.STRING(100),
        allowNull: true,
    },
    gemini_model_version_used: {
        type: DataTypes.STRING(100),
        allowNull: true,
    },
    processing_time_ms_primary: {
        type: DataTypes.INTEGER,
        allowNull: true,
    },
    processing_time_ms_gemini: {
        type: DataTypes.INTEGER,
        allowNull: true,
    },
    user_rating: {
        type: DataTypes.INTEGER,
        allowNull: true,
        validate: {
            min: 1,
            max: 5,
        },
    },
    user_text_feedback: {
        type: DataTypes.TEXT,
        allowNull: true,
    },
    feedback_timestamp: {
        type: DataTypes.DATE,
        allowNull: true,
    },
}, {
    tableName: 'chat_interactions',
    timestamps: false,
});

module.exports = ChatInteraction;
