
import { useState, useEffect, useRef } from 'react';
import { ChatMessage, ChatState } from '../types/chat';

const STORAGE_KEY = 'medibot-chat-history';

export const useChat = () => {
  const [chatState, setChatState] = useState<ChatState>({
    messages: [],
    isTyping: false,
    error: null,
    selectedTopic: 'general'
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedMessages = localStorage.getItem(STORAGE_KEY);
    if (savedMessages) {
      try {
        const messages = JSON.parse(savedMessages);
        setChatState(prev => ({ ...prev, messages }));
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    }
  }, []);

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    if (chatState.messages.length > 0) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(chatState.messages));
    }
  }, [chatState.messages]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatState.messages, chatState.isTyping]);

  const generateId = () => Math.random().toString(36).substr(2, 9);

  const generateBotResponse = (userMessage: string, topic: string): string => {
    const responses = {
      general: [
        "I understand your concern. Based on what you've described, I'd recommend consulting with a healthcare professional for a proper evaluation.",
        "Thank you for sharing that with me. While I can provide general information, it's important to speak with a doctor about your specific situation.",
        "I can help provide some general health information, but please remember that this doesn't replace professional medical advice."
      ],
      cardiology: [
        "Heart health is very important. If you're experiencing chest pain or concerning symptoms, please seek immediate medical attention.",
        "For heart-related concerns, I'd strongly recommend speaking with a cardiologist or your primary care physician.",
        "Cardiovascular health involves many factors including diet, exercise, and regular check-ups with your doctor."
      ],
      digestion: [
        "Digestive issues can have many causes. Keeping a food diary and noting symptoms can be helpful information for your doctor.",
        "For persistent digestive problems, it's best to consult with a gastroenterologist or your healthcare provider.",
        "Diet and lifestyle changes often help with digestive health, but medical evaluation may be needed for ongoing issues."
      ],
      respiratory: [
        "Breathing difficulties should always be taken seriously. If you're having trouble breathing, please seek immediate medical care.",
        "Respiratory symptoms can have various causes. A healthcare provider can perform proper tests to determine the best treatment.",
        "For ongoing respiratory concerns, pulmonary function tests and consultation with a specialist may be recommended."
      ],
      nutrition: [
        "A balanced diet is key to good health. Consider consulting with a registered dietitian for personalized nutrition advice.",
        "Nutritional needs vary by individual. Your healthcare provider can help determine the best dietary approach for you.",
        "Good nutrition supports overall health, but specific dietary recommendations should come from qualified professionals."
      ],
      'mental-health': [
        "Mental health is just as important as physical health. If you're struggling, please consider speaking with a mental health professional.",
        "There are many resources available for mental health support. Don't hesitate to reach out to a counselor or therapist.",
        "Taking care of your mental health is important. Professional support can make a significant difference."
      ]
    };

    const topicResponses = responses[topic as keyof typeof responses] || responses.general;
    return topicResponses[Math.floor(Math.random() * topicResponses.length)];
  };

  const sendMessage = async (message: string) => {
    const userMessage: ChatMessage = {
      id: generateId(),
      message,
      isBot: false,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    setChatState(prev => ({
      ...prev,
      messages: [...prev.messages, userMessage],
      isTyping: true,
      error: null
    }));

    try {
      // Simulate API call delay
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

      // Simulate occasional API failures (10% chance)
      if (Math.random() < 0.1) {
        throw new Error('Failed to get response from MediBot. Please try again.');
      }

      const botResponse: ChatMessage = {
        id: generateId(),
        message: generateBotResponse(message, chatState.selectedTopic),
        isBot: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      };

      setChatState(prev => ({
        ...prev,
        messages: [...prev.messages, botResponse],
        isTyping: false
      }));
    } catch (error) {
      setChatState(prev => ({
        ...prev,
        isTyping: false,
        error: error instanceof Error ? error.message : 'An error occurred'
      }));
    }
  };

  const clearConversation = () => {
    setChatState(prev => ({
      ...prev,
      messages: [],
      error: null
    }));
    localStorage.removeItem(STORAGE_KEY);
  };

  const setSelectedTopic = (topic: string) => {
    setChatState(prev => ({ ...prev, selectedTopic: topic }));
  };

  const dismissError = () => {
    setChatState(prev => ({ ...prev, error: null }));
  };

  return {
    ...chatState,
    sendMessage,
    clearConversation,
    setSelectedTopic,
    dismissError,
    chatEndRef
  };
};
