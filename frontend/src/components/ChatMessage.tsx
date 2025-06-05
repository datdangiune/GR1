
import React from 'react';
import { Bot, User } from 'lucide-react';

interface ChatMessageProps {
  message: string;
  isBot: boolean;
  timestamp: string;
  isDark: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isBot, timestamp, isDark }) => {
  return (
    <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
      <div className={`flex max-w-xs lg:max-w-md ${isBot ? 'flex-row' : 'flex-row-reverse'}`}>
        <div className={`flex-shrink-0 ${isBot ? 'mr-3' : 'ml-3'}`}>
          <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
            isBot 
              ? 'bg-blue-500' 
              : isDark ? 'bg-gray-600' : 'bg-gray-300'
          }`}>
            {isBot ? (
              <Bot className="w-5 h-5 text-white" />
            ) : (
              <User className={`w-5 h-5 ${isDark ? 'text-white' : 'text-gray-600'}`} />
            )}
          </div>
        </div>
        
        <div className={`rounded-lg px-4 py-2 shadow-md ${
          isBot
            ? isDark 
              ? 'bg-gray-700 text-white'
              : 'bg-gray-100 text-gray-800'
            : 'bg-blue-500 text-white'
        }`}>
          <p className="text-sm">{message}</p>
          <p className={`text-xs mt-1 ${
            isBot 
              ? isDark ? 'text-gray-400' : 'text-gray-500'
              : 'text-blue-100'
          }`}>
            {timestamp}
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
